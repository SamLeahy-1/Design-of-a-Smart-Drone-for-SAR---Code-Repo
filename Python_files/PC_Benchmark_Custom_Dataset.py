#!/usr/bin/env python3
"""
evaluate_models.py  --  STEP 4 of the offline pipeline (scoring)

Runs your detection models on the frames you labelled in Roboflow and scores
them against your labels.

WHAT IT DOES
  1. Reads the Roboflow COCO export (every split is merged). Class "person"
     = a real target. A class whose name contains "ignore" (e.g. person_ignore)
     = an ignore region: detections there count as neither hit nor false alarm.
     Images with no boxes are negatives (they measure false positives).
  2. Runs each model on every labelled frame at FULL resolution and saves ALL
     detections down to confidence CONF_FLOOR in results/predictions/.
     Nothing is re-run unless the weights, the frames or CONF_FLOOR change.
       mmdet models are run in two modes so you can see the effect of the
       640x640 squash used by the flight scripts:
         "full"   = whole frame handed to the model (what your test bench does)
         "squash" = frame squashed to 640x640 first, boxes scaled back
                    (what V6/V7 do live)
       Ultralytics (YOLO) models only have "full" (the flight scripts also pass
       the full frame).
  3. Scores every model: AP50, AP50:95, and precision / recall / F1 / false
     positives at the flight threshold (0.5) and others, overall and broken
     down by scenario, negatives-only, and altitude. Confidence intervals come
     from resampling whole CLIPS (frames in one clip are not independent).
  4. Writes tables (CSV + a Markdown table to paste into the thesis), a
     threshold sweep, PR curves, and pictures of the false positives / misses.

ENVIRONMENTS
  YOLO and mmdet models live in different virtualenvs. Run this script once in
  each (it skips models whose backend is missing), then run
      python3 evaluate_models.py --score-only
  anywhere: it scores everything that is cached.

USAGE
  python3 evaluate_models.py --check            # dataset stats + model paths, no inference
  python3 evaluate_models.py                    # infer (if needed) + score
  python3 evaluate_models.py --models YOLOv11n_VisDrone,YOLOX-tiny_HERIDAL
  python3 evaluate_models.py --score-only       # re-score from cached detections
  python3 evaluate_models.py --rerun            # ignore the cache, redo inference
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

# ============================== SETTINGS ==============================
PROJECT_DIR  = os.path.expanduser("~/drone_project/DRONE_TEST_VIDS")
DATASET_DIR  = os.path.join(PROJECT_DIR, "roboflow_export1")   # unzipped COCO export folder, or the .zip
FRAMES_DIR   = os.path.join(PROJECT_DIR, "frames")            # extract_frames.py output (optional metadata)
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")
MODELS_BASE  = os.path.expanduser("~/1_AI_Training_Benchmarking")   # same layout on PC and Jetson

# Which models to run. None = every entry in MODEL_REGISTRY below.
MODELS_TO_RUN = None
MMDET_INPUT_MODES = ["full", "squash"]

# Class ids that count as "person", per model. Default is [0] (what the flight
# scripts use). The model's class names are printed when it loads: if a
# VisDrone-style model lists 'pedestrian' AND 'people', set e.g.
#   PERSON_CLASSES = {"YOLOv11n_VisDrone": [0, 1]}
PERSON_CLASSES = {}

CONF_FLOOR           = 0.05      # keep every detection above this; thresholds are applied when scoring
OPERATING_THRESHOLDS = [0.3, 0.5]  # 0.5 = the flight scripts; 0.3 = your test bench
PRIMARY_THRESHOLD    = 0.5       # used for per-image tables and visuals
IOU_MATCH            = 0.5       # a detection is a hit if IoU with a person box >= this
IGNORE_OVERLAP       = 0.5       # detection counts as inside an ignore box if >= this fraction of it is
SAMPLE_FPS           = None      # frames per second of footage the labelled frames represent (None = read from manifests, else 2.0)
ALTITUDE_BINS        = [0, 12, 17, 22, 1000]    # metres, for the altitude breakdown
N_BOOT               = 500       # bootstrap resamples (clips) for confidence intervals
BOOT_SEED            = 0
SAVE_VISUALS         = True
MAX_VISUALS_PER_MODEL = 60
MMDET_DEVICE         = "cuda:0"
YOLO_IMGSZ           = None      # None = Ultralytics default (640), same as the flight scripts
# If Roboflow dropped your empty frames (no "Mark Null"), list the clips you
# have FULLY labelled here: frames of those clips that are missing from the
# export are treated as empty (negative) images.
ASSUME_UNLISTED_FRAMES_EMPTY_FOR_CLIPS = []

# Which camera/source each clip came from, for a per-source breakdown. Roboflow tags are NOT
# in the COCO export, so map by clip name instead. Keys are matched as an exact clip name OR as a
# name prefix (first match wins). Example:
#   SOURCE_BY_CLIP = {"testfoot": "TestA (DJI)", "s": "TestB (project drone)"}
# Clips that match nothing are reported as "unmapped".
SOURCE_BY_CLIP = {"testfoot_uni": "TestB (project drone)",
                  "testfoot": "TestA (DJI)",}

SCENARIO_NAMES = {
    1: "Empty field + bags",
    2: "People in field + bags",
    3: "Bushland, no people + bags",
    4: "People in bushland (walking, partial cover)",
    5: "Bushland, people lying / sitting",
}
# =====================================================================

SCRIPT_VERSION = "1.0"

MODEL_REGISTRY = {
    "YOLOv11n": {"type": "ultralytics",
        "weights": os.path.join(MODELS_BASE, "1_Untrained_Models/YOLOv11n_UNTRAINED/yolo11n.pt"),
        "config": None,},
    "YOLOv11n_SARD": {"type": "ultralytics",
        "weights": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/YOLOv11n_SARD/1_Optimal/YOLOv11n_Trained_SARD.pt"),
        "config": None,},
    "YOLOv11n_HERIDAL": {"type": "ultralytics",
        "weights": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/YOLOv11n_HERIDAL/1_Optimal/YOLOv11n_Trained_HERIDAL.pt"),
        "config": None,},
    "YOLOv11n_VisDrone": {"type": "ultralytics",
        "weights": os.path.join(MODELS_BASE, "3_VisDrone_Trained_Models/YOLOv11n_VisDrone/1_Optimal/YOLOv11n_Trained_VisDrone.pt"),
        "config": None,},
    "YOLOX-tiny": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "1_Untrained_Models/YOLOx-tiny_UNTRAINED/yolox_tiny.pth"),
        "config": os.path.join(MODELS_BASE, "1_Untrained_Models/YOLOx-tiny_UNTRAINED/yolox_tiny_8xb8-300e_coco.py"),},
    "YOLOX-tiny_SARD": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/YOLOx_tiny_Trained_SARD.pth"),
        "config": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/config.py"),},
    "YOLOX-tiny_HERIDAL": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/YOLOx-tiny_Trained_Heridal.pth"),
        "config": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/config.py"),},
    "YOLOX-tiny_VisDrone": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/YOLOx-tiny_Trained_VisDrone.pth"),
        "config": os.path.join(MODELS_BASE, "3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/config.py"),},
    "RTMDet-tiny": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "1_Untrained_Models/RTMDet-tiny_UNTRAINED/rtmdet_tiny.pth"),
        "config": os.path.join(MODELS_BASE, "1_Untrained_Models/RTMDet-tiny_UNTRAINED/rtmdet_tiny_8xb32-300e_coco.py"),},
    "RTMDet-tiny_SARD": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/RTMDet_tiny_Trained_SARD.pth"),
        "config": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/config.py"),},
    "RTMDet-tiny_HERIDAL": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/RTMDet-tiny_Trained_HERIDAL.pth"),
        "config": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/config.py"),},
    "RTMDet-tiny_VisDrone": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/RTMDet_tiny_Trained_VisDrone.pth"),
        "config": os.path.join(MODELS_BASE, "3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/config.py"),},
    "Faster-RCNN-R50": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "1_Untrained_Models/FasterRCNN-50_UNTRAINED/faster_rcnn_r50.pth"),
        "config": os.path.join(MODELS_BASE, "1_Untrained_Models/FasterRCNN-50_UNTRAINED/faster-rcnn_r50_fpn_1x_coco.py"),},
    "Faster-RCNN-R50_SARD": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/FasterRCNN_Trained_SARD.pth"),
        "config": os.path.join(MODELS_BASE, "2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/config.py"),},
    "Faster-RCNN-R50_HERIDAL": {"type": "mmdet",
        "weights": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/FasterRCNN_Trained_HERIDAL.pth"),
        "config": os.path.join(MODELS_BASE, "4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/config.py"),},
}


# =====================================================================
# DATASET (Roboflow COCO export)
# =====================================================================
RF_SUFFIX = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+\.(?:jpg|jpeg|png)$", re.I)
FRAME_NAME = re.compile(r"^(?P<clip>.+)_f(?P<idx>\d{6})_t(?P<t>\d+(?:[_.\-]\d+)?)s$")
CLIP_NAME = re.compile(r"^s(?P<s>\d+)_v(?P<v>\d+)$", re.I)


def image_key(file_name):
    """Roboflow renames 'clip_f000030_t001.000s.jpg' to
    'clip_f000030_t001_000s_jpg.rf.<hash>.jpg'. Undo that."""
    base = os.path.basename(file_name)
    if RF_SUFFIX.search(base):
        return RF_SUFFIX.sub("", base)
    return os.path.splitext(base)[0].replace(".", "_")


def parse_key(key):
    m = FRAME_NAME.match(key)
    if not m:
        return None, None, None, None, None
    clip = m.group("clip")
    idx = int(m.group("idx"))
    t = float(re.sub(r"[_\-]", ".", m.group("t")))
    cm = CLIP_NAME.match(clip)
    scen = int(cm.group("s")) if cm else None
    ver = int(cm.group("v")) if cm else None
    return clip, idx, t, scen, ver


def source_of(clip):
    for k, v in SOURCE_BY_CLIP.items():
        if clip == k or clip.startswith(k):
            return v
    return "unmapped" if SOURCE_BY_CLIP else None


def _resolve_dataset_dir():
    p = DATASET_DIR
    if os.path.isfile(p) and p.lower().endswith(".zip"):
        out = os.path.join(RESULTS_DIR, "_dataset_unzipped")
        if not os.path.isdir(out):
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(out)
        return out
    return p


def load_manifests(clips):
    """clip -> {source_frame_idx: dict(altitude, time_s)} from extract_frames manifests."""
    meta, fps_samples = {}, []
    for clip in clips:
        mp = os.path.join(FRAMES_DIR, clip, "manifest.csv")
        if not os.path.isfile(mp):
            continue
        rows = list(csv.DictReader(open(mp)))
        d = {}
        for r in rows:
            try:
                alt = float(r["altitude_m"]) if r.get("altitude_m") not in (None, "", "None") else None
            except ValueError:
                alt = None
            d[int(r["source_frame_idx"])] = {"altitude": alt, "time_s": float(r["time_s"])}
        meta[clip] = d
        ts = [float(r["time_s"]) for r in rows]
        if len(ts) > 2:
            fps_samples.append(1.0 / max(np.median(np.diff(ts)), 1e-6))
    return meta, fps_samples


def load_dataset():
    root = _resolve_dataset_dir()
    ann_files = sorted(glob.glob(os.path.join(root, "**", "_annotations.coco.json"), recursive=True))
    if not ann_files:
        sys.exit(f"No _annotations.coco.json found under {root}.\n"
                 f"Download the Roboflow export in COCO format and unzip it there.")
    images, notes = {}, []
    person_names, ignore_names = set(), set()
    n_gt = n_ign = 0
    for af in ann_files:
        split_dir = os.path.dirname(af)
        data = json.load(open(af))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        ids_used = {a["category_id"] for a in data.get("annotations", [])}
        for cid in ids_used:
            (ignore_names if "ignore" in cats.get(cid, "").lower() else person_names).add(cats.get(cid, str(cid)))
        by_img = defaultdict(list)
        for a in data.get("annotations", []):
            by_img[a["image_id"]].append(a)
        for im in data["images"]:
            key = image_key(im["file_name"])
            if key in images:
                notes.append(f"duplicate image {key} (kept the first)")
                continue
            gt, ign = [], []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                box = [x, y, x + w, y + h]
                if "ignore" in cats.get(a["category_id"], "").lower():
                    ign.append(box)
                else:
                    gt.append(box)
            clip, idx, t, scen, ver = parse_key(key)
            images[key] = {
                "key": key, "path": os.path.join(split_dir, im["file_name"]),
                "w": im.get("width"), "h": im.get("height"),
                "clip": clip or "unknown", "idx": idx, "t": t, "scenario": scen, "version": ver,
                "gt": np.array(gt, float).reshape(-1, 4), "ign": np.array(ign, float).reshape(-1, 4),
                "altitude": None, "source": source_of(clip or "unknown"),
            }
            n_gt += len(gt)
            n_ign += len(ign)

    # frames the export dropped (no Mark Null) for clips the user says are fully labelled
    for clip in ASSUME_UNLISTED_FRAMES_EMPTY_FOR_CLIPS:
        mp = os.path.join(FRAMES_DIR, clip, "manifest.csv")
        if not os.path.isfile(mp):
            notes.append(f"{clip}: no manifest, cannot add unlisted frames")
            continue
        added = 0
        for r in csv.DictReader(open(mp)):
            key = image_key(r["image"])
            if key not in images:
                c, idx, t, scen, ver = parse_key(key)
                images[key] = {"key": key, "path": os.path.join(FRAMES_DIR, clip, r["image"]),
                               "w": int(r["width"]), "h": int(r["height"]), "clip": c or clip, "idx": idx,
                               "t": t, "scenario": scen, "version": ver,
                               "gt": np.zeros((0, 4)), "ign": np.zeros((0, 4)), "altitude": None,
                               "source": source_of(c or clip)}
                added += 1
        notes.append(f"{clip}: {added} unlisted frame(s) treated as empty")

    # metadata join (altitude) + sample fps
    clips = sorted({im["clip"] for im in images.values()})
    meta, fps_samples = load_manifests(clips)
    n_alt = 0
    for im in images.values():
        m = meta.get(im["clip"], {}).get(im["idx"])
        if m and m["altitude"] is not None:
            im["altitude"] = m["altitude"]
            n_alt += 1
    sample_fps = SAMPLE_FPS or (float(np.median(fps_samples)) if fps_samples else 2.0)
    info = {"person_names": sorted(person_names), "ignore_names": sorted(ignore_names),
            "n_gt": n_gt, "n_ignore": n_ign, "n_with_altitude": n_alt,
            "sample_fps": round(sample_fps, 3), "notes": notes,
            "annotation_files": [os.path.relpath(a, root) for a in ann_files]}
    return images, info


# =====================================================================
# DETECTORS
# =====================================================================
class Detector:
    """Wraps one model. detect(frame, mode) -> array (N, 6): x1 y1 x2 y2 score cls
    (all classes above CONF_FLOOR, in full-frame pixel coordinates)."""

    def __init__(self, name):
        self.name = name
        self.entry = MODEL_REGISTRY[name]
        self.type = self.entry["type"]
        w = self.entry["weights"]
        if not os.path.isfile(w):
            raise FileNotFoundError(f"weights not found: {w}")
        if self.entry["config"] and not os.path.isfile(self.entry["config"]):
            raise FileNotFoundError(f"config not found: {self.entry['config']}")
        if self.type == "ultralytics":
            from ultralytics import YOLO
            self.model = YOLO(w)
            names = getattr(self.model, "names", {})
            self.class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        elif self.type == "mmdet":
            from mmdet.apis import init_detector
            self.model = init_detector(self.entry["config"], w, device=MMDET_DEVICE)
            meta = getattr(self.model, "dataset_meta", None) or {}
            self.class_names = list(meta.get("classes", []))
        else:
            raise ValueError(f"unknown model type {self.type}")
        self.full_fallback = False

    def modes(self):
        return list(MMDET_INPUT_MODES) if self.type == "mmdet" else ["full"]

    def detect(self, frame, mode):
        if self.type == "ultralytics":
            kw = {"verbose": False, "conf": CONF_FLOOR}
            if YOLO_IMGSZ:
                kw["imgsz"] = YOLO_IMGSZ
            r = self.model(frame, **kw)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return np.zeros((0, 6))
            b = r.boxes
            return np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                    b.cls.cpu().numpy()]).astype(float)
        from mmdet.apis import inference_detector
        h, w = frame.shape[:2]
        if mode == "squash":
            inp, sx, sy = cv2.resize(frame, (640, 640)), w / 640.0, h / 640.0
        else:
            inp, sx, sy = frame, 1.0, 1.0
        if mode == "full" and self.full_fallback:
            inp, sx, sy = self._letterbox(frame)
        try:
            pred = inference_detector(self.model, inp).pred_instances
        except RuntimeError as e:
            if mode == "full" and not self.full_fallback and "Sizes of tensors must match" in str(e):
                self.full_fallback = True
                print(f"        NOTE: {self.name} cannot take a 1280x720 frame as-is (its config has no padding step). "
                      f"Using aspect-preserving resize (long side 640) + pad to a multiple of 32 for 'full' mode.")
                return self.detect(frame, mode)
            raise
        boxes = pred.bboxes.cpu().numpy().astype(float)
        scores = pred.scores.cpu().numpy().astype(float)
        labels = pred.labels.cpu().numpy().astype(float)
        keep = scores >= CONF_FLOOR
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(boxes) == 0:
            return np.zeros((0, 6))
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        return np.column_stack([boxes[:, :4], scores, labels])


def _letterbox_impl(frame):
    h, w = frame.shape[:2]
    sc = 640.0 / max(h, w)
    img = cv2.resize(frame, (int(round(w * sc)), int(round(h * sc))))
    H, W = img.shape[:2]
    ph, pw = (-H) % 32, (-W) % 32
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, 1.0 / sc, 1.0 / sc


Detector._letterbox = staticmethod(_letterbox_impl)


def person_classes_for(name):
    return list(PERSON_CLASSES.get(name, [0]))


def weights_signature(name):
    w = MODEL_REGISTRY[name]["weights"]
    try:
        st = os.stat(w)
        return {"weights": w, "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"weights": w}


def cache_path(name, mode):
    return os.path.join(RESULTS_DIR, "predictions", f"{name}__{mode}.json")


def load_cache(name, mode):
    p = cache_path(name, mode)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def save_cache(name, mode, meta, dets, ms):
    os.makedirs(os.path.dirname(cache_path(name, mode)), exist_ok=True)
    tmp = cache_path(name, mode) + ".tmp"
    json.dump({"meta": meta, "dets": dets, "ms": ms}, open(tmp, "w"))
    os.replace(tmp, cache_path(name, mode))


def run_inference(names, images, rerun):
    """Fill the prediction cache for each requested model (skips ones already complete)."""
    for name in names:
        if name not in MODEL_REGISTRY:
            print(f"[skip ] unknown model '{name}'")
            continue
        entry = MODEL_REGISTRY[name]
        modes = list(MMDET_INPUT_MODES) if entry["type"] == "mmdet" else ["full"]
        sig = weights_signature(name)
        todo = {}
        for mode in modes:
            c = None if rerun else load_cache(name, mode)
            valid = (c is not None and c["meta"].get("sig") == sig
                     and c["meta"].get("conf_floor") == CONF_FLOOR
                     and c["meta"].get("yolo_imgsz") == YOLO_IMGSZ)
            have = set(c["dets"]) if valid else set()
            missing = [k for k in images if k not in have]
            if missing:
                todo[mode] = (c if valid else None, missing)
        if not todo:
            print(f"[cache] {name}: all {len(images)} frames already processed")
            continue
        try:
            det = Detector(name)
        except Exception as e:
            print(f"[skip ] {name}: {e!r}")
            continue
        print(f"[load ] {name}  classes={det.class_names}")
        pc = person_classes_for(name)
        if len(det.class_names) > 1 and pc == [0]:
            print(f"        NOTE: model has {len(det.class_names)} classes; only class 0 "
                  f"('{det.class_names[0]}') counts as person. Edit PERSON_CLASSES if more should.")
        for mode, (old, missing) in todo.items():
            dets = dict(old["dets"]) if old else {}
            ms = dict(old["ms"]) if old else {}
            first = cv2.imread(images[missing[0]]["path"])
            if first is None:
                print(f"[skip ] cannot read {images[missing[0]]['path']}")
                continue
            for _ in range(2):                      # warm-up, not timed
                det.detect(first, mode)
            t_run = time.perf_counter()
            for i, k in enumerate(missing, 1):
                frame = cv2.imread(images[k]["path"])
                if frame is None:
                    print(f"[warn ] unreadable image {images[k]['path']}")
                    continue
                t0 = time.perf_counter()
                out = det.detect(frame, mode)
                ms[k] = (time.perf_counter() - t0) * 1000.0
                dets[k] = [[round(float(v), 3) for v in row] for row in out]
                if i % 25 == 0 or i == len(missing):
                    print(f"        {name} [{mode}] {i}/{len(missing)}  "
                          f"({(time.perf_counter() - t_run) / i * 1000:.0f} ms/img incl. file read)", end="\r")
            print()
            meta = {"sig": sig, "conf_floor": CONF_FLOOR, "yolo_imgsz": YOLO_IMGSZ, "mode": mode, "full_fallback_letterbox": bool(getattr(det, "full_fallback", False)),
                    "class_names": det.class_names, "host": socket.gethostname(),
                    "device": MMDET_DEVICE if entry["type"] == "mmdet" else "ultralytics-auto",
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "versions": backend_versions()}
            save_cache(name, mode, meta, dets, ms)
        del det


def backend_versions():
    v = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}
    for mod in ("torch", "ultralytics", "mmdet", "mmcv"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


# =====================================================================
# SCORING
# =====================================================================
def _thresholds():
    base = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    thrs = sorted(set(base + [round(IOU_MATCH, 2)]))
    return thrs, thrs.index(round(IOU_MATCH, 2)), thrs.index(0.5), [thrs.index(b) for b in base]


THRS, OP_I, AP50_I, AP_IDX = _thresholds()


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def ign_fraction(a, ign):
    """fraction of each detection's area lying inside its best-overlapping ignore box"""
    if len(a) == 0 or len(ign) == 0:
        return np.zeros(len(a))
    ix1 = np.maximum(a[:, None, 0], ign[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], ign[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], ign[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], ign[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)
    return (inter / area[:, None]).max(1)


def match_image(d, gt, ign):
    """d: (n,>=5) sorted by score desc. COCO-style greedy matching at every IoU threshold.
    Returns tp (T,n) and ignored (T,n) boolean arrays, plus gt_matched at the operating IoU."""
    n, T = len(d), len(THRS)
    tp = np.zeros((T, n), bool)
    ig = np.zeros((T, n), bool)
    gt_op = np.zeros(len(gt), bool)
    if n == 0:
        return tp, ig, gt_op
    ious = iou_matrix(d[:, :4], gt)
    frac = ign_fraction(d[:, :4], ign)
    for ti, thr in enumerate(THRS):
        gm = np.zeros(len(gt), bool)
        for i in range(n):
            if len(gt):
                cand = np.where((~gm) & (ious[i] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    gm[j] = True
                    tp[ti, i] = True
                    continue
            if frac[i] >= IGNORE_OVERLAP:
                ig[ti, i] = True
        if ti == OP_I:
            gt_op = gm
    return tp, ig, gt_op


def build_records(name, cache, images, pcs=None):
    pcs = person_classes_for(name) if pcs is None else pcs
    recs = []
    for key, im in images.items():
        if key not in cache["dets"]:
            continue
        arr = np.array(cache["dets"][key], float).reshape(-1, 6)
        arr = arr[(arr[:, 4] >= CONF_FLOOR) & np.isin(arr[:, 5], pcs)]
        arr = arr[np.argsort(-arr[:, 4], kind="stable")[:100]]
        tp, ig, gm = match_image(arr, im["gt"], im["ign"])
        recs.append({"key": key, "clip": im["clip"], "scenario": im["scenario"],
                     "altitude": im["altitude"], "source": im.get("source"), "n_gt": len(im["gt"]),
                     "scores": arr[:, 4], "boxes": arr[:, :4], "tp": tp, "ign": ig, "gt_matched": gm})
    return recs


def ap_from(scores, tp, ign, n_gt):
    """COCO-style AP (101-point interpolation); identical to pycocotools when there are no ignore regions."""
    if n_gt == 0:
        return float("nan")
    keep = ~ign
    s, t = scores[keep], tp[keep]
    if len(s) == 0:
        return 0.0
    t = t[np.argsort(-s, kind="stable")]
    tpc = np.cumsum(t)
    fpc = np.cumsum(~t)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    inds = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(prec):
            q[ri] = prec[pi]
    return float(q.mean())


def agg_arrays(recs, ti):
    if not recs:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), 0
    return (np.concatenate([r["scores"] for r in recs]),
            np.concatenate([r["tp"][ti] for r in recs]),
            np.concatenate([r["ign"][ti] for r in recs]),
            int(sum(r["n_gt"] for r in recs)))


def counts_at(recs, thr):
    TP = FP = 0
    n_gt = sum(r["n_gt"] for r in recs)
    neg = neg_alarm = pos = pos_hit = 0
    for r in recs:
        sel = (r["scores"] >= thr) & ~r["ign"][OP_I]
        tp_i = int((r["tp"][OP_I] & sel).sum())
        fp_i = int((~r["tp"][OP_I] & sel).sum())
        TP += tp_i
        FP += fp_i
        if r["n_gt"] == 0:
            neg += 1
            neg_alarm += fp_i > 0
        else:
            pos += 1
            pos_hit += tp_i > 0
    return {"tp": TP, "fp": FP, "fn": n_gt - TP, "n_gt": n_gt, "neg": neg, "neg_alarm": neg_alarm,
            "pos": pos, "pos_hit": pos_hit}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else float("nan")
    return p, r, f


def bootstrap_ci(recs, sample_fps):
    """95% CIs by resampling whole clips. Returns dict or None if fewer than 3 clips."""
    clips = sorted({r["clip"] for r in recs})
    nc = len(clips)
    if nc < 3 or N_BOOT <= 0:
        return None
    per = [[r for r in recs if r["clip"] == c] for c in clips]
    nthr = len(OPERATING_THRESHOLDS)
    C = np.zeros((nc, nthr, 3))          # tp, fp, fn
    nimg = np.array([len(p) for p in per], float)
    for ci, rs in enumerate(per):
        for ti, thr in enumerate(OPERATING_THRESHOLDS):
            c = counts_at(rs, thr)
            C[ci, ti] = (c["tp"], c["fp"], c["fn"])
    ap_parts = [agg_arrays(rs, AP50_I) for rs in per]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, nc, size=(N_BOOT, nc))
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for ti, thr in enumerate(OPERATING_THRESHOLDS):
            tp, fp, fn = (C[idx, ti, k].sum(1) for k in range(3))
            P = tp / (tp + fp)
            R = tp / (tp + fn)
            F = 2 * P * R / (P + R)
            fpi = fp / nimg[idx].sum(1)
            for nm, arr in (("precision", P), ("recall", R), ("f1", F), ("fp_per_img", fpi),
                            ("fp_per_min", fpi * sample_fps * 60)):
                v = arr[np.isfinite(arr)]
                out[(nm, thr)] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (float("nan"),) * 2
    aps = []
    for b in range(min(N_BOOT, 300)):
        ch = idx[b]
        s = np.concatenate([ap_parts[c][0] for c in ch])
        t = np.concatenate([ap_parts[c][1] for c in ch])
        g = np.concatenate([ap_parts[c][2] for c in ch])
        aps.append(ap_from(s, t, g, sum(ap_parts[c][3] for c in ch)))
    aps = np.array([a for a in aps if a == a])
    out[("ap50", None)] = (float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))) if len(aps) > 20 else (float("nan"),) * 2
    return out


def subset_defs(recs):
    d = {"ALL frames": recs}
    scen = sorted({r["scenario"] for r in recs if r["scenario"] is not None})
    for s in scen:
        d[f"Scenario {s}: {SCENARIO_NAMES.get(s, '')}".strip().rstrip(":")] = [r for r in recs if r["scenario"] == s]
    srcs = sorted({r["source"] for r in recs if r.get("source")})
    if len(srcs) > 1 or (srcs and srcs != ["unmapped"]):
        for src in srcs:
            d[f"Source: {src}"] = [r for r in recs if r.get("source") == src]
    d["Frames with NO person (false-positive test)"] = [r for r in recs if r["n_gt"] == 0]
    d["Frames WITH a person"] = [r for r in recs if r["n_gt"] > 0]
    have_alt = [r for r in recs if r["altitude"] is not None]
    if len(have_alt) >= 0.5 * max(len(recs), 1) and have_alt:
        for lo, hi in zip(ALTITUDE_BINS[:-1], ALTITUDE_BINS[1:]):
            sel = [r for r in have_alt if lo <= r["altitude"] < hi]
            if sel:
                d[f"Altitude {lo}-{hi if hi < 999 else '+'} m"] = sel
    return {k: v for k, v in d.items() if v}


def subset_row(model, mode, subset, recs, sample_fps):
    row = {"model": model, "mode": mode, "subset": subset, "n_images": len(recs),
           "n_clips": len({r["clip"] for r in recs}), "n_gt": int(sum(r["n_gt"] for r in recs))}
    s, t, g, n = agg_arrays(recs, AP50_I)
    row["ap50"] = ap_from(s, t, g, n)
    aps = []
    for ti in AP_IDX:
        s, t, g, n = agg_arrays(recs, ti)
        aps.append(ap_from(s, t, g, n))
    row["ap50_95"] = float(np.mean(aps)) if n > 0 else float("nan")
    ci = bootstrap_ci(recs, sample_fps)
    row["ci_available"] = ci is not None
    if ci:
        row["ap50_lo"], row["ap50_hi"] = ci[("ap50", None)]
    for thr in OPERATING_THRESHOLDS:
        c = counts_at(recs, thr)
        p, r_, f = prf(c["tp"], c["fp"], c["fn"])
        fpi = c["fp"] / len(recs)
        k = f"@{thr}"
        row.update({f"tp{k}": c["tp"], f"fp{k}": c["fp"], f"fn{k}": c["fn"],
                    f"precision{k}": p, f"recall{k}": r_, f"f1{k}": f,
                    f"fp_per_img{k}": fpi, f"fp_per_min{k}": fpi * sample_fps * 60,
                    f"neg_frame_false_alarm_rate{k}": (c["neg_alarm"] / c["neg"]) if c["neg"] else float("nan"),
                    f"frame_hit_rate{k}": (c["pos_hit"] / c["pos"]) if c["pos"] else float("nan")})
        if ci:
            for nm in ("precision", "recall", "f1", "fp_per_img", "fp_per_min"):
                row[f"{nm}_lo{k}"], row[f"{nm}_hi{k}"] = ci[(nm, thr)]
    return row


def threshold_sweep(recs):
    rows = []
    for thr in np.round(np.arange(0.05, 0.951, 0.05), 2):
        c = counts_at(recs, float(thr))
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        rows.append((float(thr), p, r, f, c["fp"] / max(len(recs), 1)))
    return rows


# =====================================================================
# OUTPUTS
# =====================================================================
def label_of(name, mode):
    return name if MODEL_REGISTRY.get(name, {}).get("type") != "mmdet" else f"{name} [{mode}]"


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.{nd}f}"


def fmt_ci(v, lo, hi, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if lo is None or hi is None or lo != lo or hi != hi:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})"


def write_csv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and v != v) else v) for k, v in r.items()})


def write_markdown(path, rows, info, n_images, n_clips):
    P = PRIMARY_THRESHOLD
    k = f"@{P}"
    L = []
    L.append("# Offline evaluation summary\n")
    L.append(f"- Labelled frames scored: **{n_images}** from **{n_clips}** clip(s); "
             f"{info['n_gt']} person boxes, {info['n_ignore']} ignore regions.")
    L.append(f"- Matching: IoU ≥ {IOU_MATCH}; detections inside ignore regions (≥ {IGNORE_OVERLAP} of their area) are not counted.")
    L.append(f"- Operating threshold for P/R/F1/FP: **{P}** (flight scripts). AP is threshold-free.")
    L.append(f"- FP/min assumes the labelled frames are sampled at {info['sample_fps']} frames per second of footage.")
    L.append("- 95% intervals come from resampling whole clips; “–” intervals mean fewer than 3 clips.\n")
    modes = list(dict.fromkeys((r["model"], r["mode"]) for r in rows))
    L.append("## Overall (all labelled frames)\n")
    L.append(f"| Model | AP50 (95% CI) | AP50:95 | Precision{k} | Recall{k} | F1{k} | FP/min{k} | False-alarm rate on empty frames{k} |")
    L.append("|---|---|---|---|---|---|---|---|")
    allr = [r for r in rows if r["subset"] == "ALL frames"]
    for r in sorted(allr, key=lambda r: -(r["ap50"] if r["ap50"] == r["ap50"] else -1)):
        L.append(f"| {label_of(r['model'], r['mode'])} | {fmt_ci(r['ap50'], r.get('ap50_lo'), r.get('ap50_hi'))} | "
                 f"{fmt(r['ap50_95'])} | {fmt(r['precision' + k])} | {fmt(r['recall' + k])} | {fmt(r['f1' + k])} | "
                 f"{fmt(r['fp_per_min' + k], 1)} | {fmt(r['neg_frame_false_alarm_rate' + k])} |")
    subsets = list(dict.fromkeys(r["subset"] for r in rows if r["subset"] != "ALL frames"))
    for s in subsets:
        L.append(f"\n## {s}\n")
        sr = [r for r in rows if r["subset"] == s]
        L.append(f"_{sr[0]['n_images']} frames, {sr[0]['n_clips']} clip(s), {sr[0]['n_gt']} person boxes_\n")
        L.append(f"| Model | AP50 | Precision{k} | Recall{k} | F1{k} | FP{k} | FP/min{k} |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(sr, key=lambda r: label_of(r["model"], r["mode"])):
            L.append(f"| {label_of(r['model'], r['mode'])} | {fmt(r['ap50'])} | {fmt(r['precision' + k])} | "
                     f"{fmt(r['recall' + k])} | {fmt(r['f1' + k])} | {r['fp' + k]} | {fmt(r['fp_per_min' + k], 1)} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def draw_visual(im, rec, thr, path):
    img = cv2.imread(im["path"])
    if img is None:
        return
    def box(b, col, th=2):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
    for b in im["ign"]:
        box(b, (150, 150, 150), 1)
    for j, b in enumerate(im["gt"]):
        box(b, (0, 200, 0) if rec["gt_matched"][j] else (0, 220, 255), 2)   # green hit, yellow missed
    for i in range(len(rec["scores"])):
        if rec["scores"][i] < thr or rec["ign"][OP_I][i]:
            continue
        ok = rec["tp"][OP_I][i]
        box(rec["boxes"][i], (255, 120, 0) if ok else (0, 0, 255), 2)         # blue TP, red FP
        cv2.putText(img, f"{rec['scores'][i]:.2f}", (int(rec["boxes"][i][0]), max(int(rec["boxes"][i][1]) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if not ok else (255, 120, 0), 1)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_visuals(label, recs, images, thr):
    out = os.path.join(RESULTS_DIR, "visuals", re.sub(r"[^\w\-.\[\]]+", "_", label))
    os.makedirs(out, exist_ok=True)
    scored = []
    for r in recs:
        sel = (r["scores"] >= thr) & ~r["ign"][OP_I]
        fp = int((~r["tp"][OP_I] & sel).sum())
        fn = int((~r["gt_matched"]).sum())
        if fp + fn:
            scored.append((fp + fn, r))
    scored.sort(key=lambda x: -x[0])
    for _, r in scored[:MAX_VISUALS_PER_MODEL]:
        draw_visual(images[r["key"]], r, thr, os.path.join(out, r["key"] + ".jpg"))


def plot_pr(curves, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[info ] matplotlib not installed: skipping pr_curves.png")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (rec, prec) in curves.items():
        ax.plot(rec, prec, label=label, lw=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"Precision-recall at IoU {IOU_MATCH} (all labelled frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pr_curve(recs):
    s, t, g, n = agg_arrays(recs, AP50_I)
    keep = ~g
    s, t = s[keep], t[keep]
    if n == 0 or len(s) == 0:
        return np.zeros(0), np.zeros(0)
    t = t[np.argsort(-s, kind="stable")]
    tpc, fpc = np.cumsum(t), np.cumsum(~t)
    return tpc / n, tpc / np.maximum(tpc + fpc, 1e-12)


# =====================================================================
# MAIN
# =====================================================================
def do_check(images, info, names):
    clips = sorted({im["clip"] for im in images.values()})
    print(f"\nDataset: {len(images)} labelled frames, {len(clips)} clip(s)")
    print(f"  person classes: {info['person_names']}   ignore classes: {info['ignore_names']}")
    print(f"  person boxes: {info['n_gt']}   ignore boxes: {info['n_ignore']}   frames with altitude: {info['n_with_altitude']}")
    print(f"  sample rate assumed: {info['sample_fps']} fps")
    n_empty = sum(1 for im in images.values() if len(im["gt"]) == 0)
    print(f"  frames with no person (negatives): {n_empty}")
    for c in clips:
        ims = [im for im in images.values() if im["clip"] == c]
        print(f"    {c:12s} {len(ims):4d} frames, {sum(len(i['gt']) for i in ims):4d} persons")
    if SOURCE_BY_CLIP:
        for src in sorted({im["source"] for im in images.values()}):
            ims = [im for im in images.values() if im["source"] == src]
            print(f"  source {src}: {len(ims)} frames in {len({i['clip'] for i in ims})} clip(s)")
    unparsed = [k for k, im in images.items() if im["idx"] is None]
    if unparsed:
        print(f"  WARNING: {len(unparsed)} image name(s) did not match the clip_fNNNNNN_tNNN.NNNs pattern, e.g. {unparsed[0]}")
    for n in info["notes"][:10]:
        print("  note:", n)
    print("\nModels:")
    for name in names:
        e = MODEL_REGISTRY[name]
        ok = os.path.isfile(e["weights"]) and (not e["config"] or os.path.isfile(e["config"]))
        print(f"  [{'ok' if ok else 'MISSING':7s}] {name:28s} {e['weights']}")


def main():
    ap = argparse.ArgumentParser(description="Score models on Roboflow-labelled drone frames.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models", default=None, help="comma-separated names")
    ap.add_argument("--modes", default=None, help="comma-separated: full,squash")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--only-selected", action="store_true",
                    help="score only the models named by --models (default: score EVERY model that has saved detections)")
    ap.add_argument("--dataset", default=None, help="COCO export folder or .zip (overrides DATASET_DIR)")
    ap.add_argument("--results", default=None, help="results folder (overrides RESULTS_DIR)")
    a = ap.parse_args()
    global DATASET_DIR, RESULTS_DIR
    if a.dataset:
        DATASET_DIR = os.path.expanduser(a.dataset)
    if a.results:
        RESULTS_DIR = os.path.expanduser(a.results)

    images, info = load_dataset()
    names = [n.strip() for n in a.models.split(",")] if a.models else (MODELS_TO_RUN or list(MODEL_REGISTRY))
    bad = [n for n in names if n not in MODEL_REGISTRY]
    if bad:
        sys.exit(f"Unknown model(s): {bad}\nKnown: {list(MODEL_REGISTRY)}")
    if a.check:
        do_check(images, info, names)
        return
    if not images:
        sys.exit("Dataset is empty.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if a.modes:
        global MMDET_INPUT_MODES
        MMDET_INPUT_MODES = [m.strip() for m in a.modes.split(",")]
    if not a.score_only:
        run_inference(names, images, a.rerun)

    # ---- scoring everything that is cached ----
    # --models limits which models get INFERENCE; the tables always include every model with saved detections
    names = list(names) if a.only_selected else list(MODEL_REGISTRY)
    sample_fps = info["sample_fps"]
    allrows, sweeps, perimg, timing, curves, infos = [], [], [], [], {}, {}
    for name in names:
        for mode in (["full", "squash"] if MODEL_REGISTRY[name]["type"] == "mmdet" else ["full"]):
            if a.modes and mode not in MMDET_INPUT_MODES and MODEL_REGISTRY[name]["type"] == "mmdet":
                continue
            cache = load_cache(name, mode)
            if not cache:
                continue
            label = label_of(name, mode)
            covered = [k for k in images if k in cache["dets"]]
            if len(covered) < len(images):
                print(f"[warn ] {label}: only {len(covered)}/{len(images)} frames have detections (scored on those)")
            recs = build_records(name, cache, images)
            if not recs:
                continue
            for sname, srecs in subset_defs(recs).items():
                allrows.append(subset_row(name, mode, sname, srecs, sample_fps))
            sw = threshold_sweep(recs)
            best = max(sw, key=lambda x: (x[3] if x[3] == x[3] else -1))
            for thr, p, r_, f, fpi in sw:
                sweeps.append({"model": name, "mode": mode, "threshold": thr, "precision": p, "recall": r_,
                               "f1": f, "fp_per_img": fpi, "fp_per_min": fpi * sample_fps * 60,
                               "is_best_f1": thr == best[0]})
            for r in recs:
                sel = (r["scores"] >= PRIMARY_THRESHOLD) & ~r["ign"][OP_I]
                perimg.append({"model": name, "mode": mode, "image": r["key"], "clip": r["clip"],
                               "scenario": r["scenario"], "altitude_m": r["altitude"], "n_person": r["n_gt"],
                               "n_det": int(sel.sum()), "tp": int((r["tp"][OP_I] & sel).sum()),
                               "fp": int((~r["tp"][OP_I] & sel).sum()), "fn": int((~r["gt_matched"]).sum()),
                               "best_score": float(r["scores"][~r["ign"][OP_I]].max()) if (~r["ign"][OP_I]).any() else 0.0})
            ms = np.array([v for k, v in cache["ms"].items() if k in images])
            if len(ms):
                timing.append({"model": name, "mode": mode, "n": len(ms), "median_ms": float(np.median(ms)),
                               "mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95)),
                               "fps_from_median": 1000.0 / float(np.median(ms)),
                               "host": cache["meta"].get("host"), "device": cache["meta"].get("device"),
                               "note": "pure model time on that host; PC and Jetson numbers are not comparable"})
            curves[label] = pr_curve(recs)
            infos[label] = {"weights": cache["meta"].get("sig"), "class_names": cache["meta"].get("class_names"),
                            "person_classes": person_classes_for(name), "versions": cache["meta"].get("versions"),
                            "host": cache["meta"].get("host"), "created": cache["meta"].get("created")}
            if SAVE_VISUALS:
                save_visuals(label, recs, images, PRIMARY_THRESHOLD)
    if not allrows:
        sys.exit("No cached detections to score. Run without --score-only in the right environment first.")

    write_csv(os.path.join(RESULTS_DIR, "summary.csv"), allrows)
    write_csv(os.path.join(RESULTS_DIR, "per_image.csv"), perimg)
    write_csv(os.path.join(RESULTS_DIR, "threshold_sweep.csv"), sweeps)
    write_csv(os.path.join(RESULTS_DIR, "timing.csv"), timing)
    n_clips = len({im["clip"] for im in images.values()})
    write_markdown(os.path.join(RESULTS_DIR, "summary.md"), allrows, info, len(images), n_clips)
    plot_pr(curves, os.path.join(RESULTS_DIR, "pr_curves.png"))
    json.dump({"script_version": SCRIPT_VERSION, "run": datetime.now().isoformat(timespec="seconds"),
               "host": socket.gethostname(), "config": {
                   "CONF_FLOOR": CONF_FLOOR, "OPERATING_THRESHOLDS": OPERATING_THRESHOLDS,
                   "PRIMARY_THRESHOLD": PRIMARY_THRESHOLD, "IOU_MATCH": IOU_MATCH, "IGNORE_OVERLAP": IGNORE_OVERLAP,
                   "SAMPLE_FPS": sample_fps, "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED},
               "dataset": info, "n_images": len(images), "n_clips": n_clips, "models": infos},
              open(os.path.join(RESULTS_DIR, "run_info.json"), "w"), indent=2, default=str)
    print(f"\nWrote results to {RESULTS_DIR}\n  summary.md  summary.csv  per_image.csv  threshold_sweep.csv  "
          f"timing.csv  pr_curves.png  visuals/  run_info.json")


if __name__ == "__main__":
    main()