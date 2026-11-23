# ============================================================
# SAR Drone - Model Benchmark Script (PC + Jetson, unified)
#
# Runs inference on a test split for ONE model at a time and
# appends performance metrics (mAP family, speed, precision/
# recall at threshold) as a row in a shared CSV file.
#
# Designed to run UNCHANGED on both PC and Jetson: put the
# datasets, trained models, and results CSV all on the same USB
# drive, then the only thing that differs between devices is the
# USB_ROOT mount path block below. Model and dataset blocks are
# identical either way -- uncomment the same ones on both machines
# to get directly comparable rows in the same CSV.
#
# HOW TO RUN (PC):
#   source ~/mmdet_train_env/bin/activate
#   python3 benchmark.py
#
# HOW TO RUN (Jetson):
#   source ~/mmdet_env/bin/activate
#   python3 ~/SAR_Drone/benchmark.py
#
# WORKFLOW:
#   1. Set PLATFORM_LABEL and uncomment the matching USB_ROOT line
#   2. Uncomment ONE dataset block (defines test set + class mapping)
#   3. Uncomment ONE model block
#   4. (Optional) set RUN_LABEL, e.g. "baseline" or "visdrone_e50"
#   5. Run the script -- repeat for each model/checkpoint/device
#   6. Open benchmark_results.csv (on the USB) to compare all runs
# ============================================================

import os
import csv
import time
import json
import datetime

import numpy as np
import torch

# ---- PLATFORM (uncomment ONE) --------------------------------
PLATFORM_LABEL = "PC"
# PLATFORM_LABEL = "Jetson"

# USB_ROOT = "/mnt/e"                 # PC (WSL) -- adjust drive letter to match your setup
# USB_ROOT = "/media/sam/ELLAB"       # Jetson
USB_ROOT = "/mnt/usb"  # <-- ACTIVE: change this line (or the two above) to match your mount point

# ---- DATASET (uncomment ONE) ----------------------------------
# Each preset defines: where the test images/annotations live, and
# CLASS_ID_TO_COCO_CAT -- the mapping from the MODEL's own output
# class index (0, 1, 2...) to the COCO category_id used in the
# ground-truth JSON. This mapping must match how the dataset's
# classes were ordered at training/conversion time (sorted by COCO
# category id ascending -> local index 0, 1, 2...), same convention
# used by coco_to_yolo.py.

# --- SARD (1-class: person) ---
# DATASET_NAME = "SARD-test"
# DATASET_ROOT = USB_ROOT + "/datasets/SARD"
# TEST_ANN_FILE = DATASET_ROOT + "/test/_annotations.coco.json"
# TEST_IMG_DIR = DATASET_ROOT + "/test"
# VALID_CLASS_IDS = [0]
# CLASS_ID_TO_COCO_CAT = {0: 1}

# --- VisDrone (3-class: person, vehicle, cyclist) ---
DATASET_NAME = "VisDrone-test-dev"
DATASET_ROOT = USB_ROOT + "/datasets/VisDrone"
TEST_ANN_FILE = DATASET_ROOT + "/test/_annotations_merged.coco.json"
TEST_IMG_DIR = DATASET_ROOT + "/test"
VALID_CLASS_IDS = [0, 1, 2]
CLASS_ID_TO_COCO_CAT = {0: 1, 1: 2, 2: 3}  # local idx -> COCO category_id


# --- HERIDAL (1-class: person) ---
# DATASET_NAME = "HERIDAL-test"
# DATASET_ROOT = USB_ROOT + "/datasets/HERIDAL"
# TEST_ANN_FILE = DATASET_ROOT + "/test/_annotations.coco.json"
# TEST_IMG_DIR = DATASET_ROOT + "/test"
# VALID_CLASS_IDS = [0]
# CLASS_ID_TO_COCO_CAT = {0: 1}


# ---- MODEL (uncomment ONE) -------------------------------------

# MODEL_NAME = "Faster-RCNN-R50"
# MODEL_TYPE = "mmdet"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/FasterRCNN_Trained_HERIDAL.pth"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/FasterRCNN_Trained_SARD.pth"
# CONFIG_PATH = USB_ROOT + "/models/Faster-RCNN-R50_HERIDAL/config.py"
# WEIGHTS_PATH = USB_ROOT + "/models/Faster-RCNN-R50_HERIDAL/FasterRCNN_Trained_HERIDAL.pth"

# MODEL_NAME = "RTMDet-tiny"
# MODEL_TYPE = "mmdet"  # "mmdet" or "yolo"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/RTMDet_tiny_Trained_VisDrone.pth"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/RTMDet-tiny_Trained_HERIDAL.pth"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/RTMDet_tiny_Trained_SARD.pth"
# CONFIG_PATH = USB_ROOT + "/models/RTMDet-tiny_VisDrone/config.py"
# WEIGHTS_PATH = USB_ROOT + "/models/RTMDet-tiny_VisDrone/RTMDet_tiny_Trained_VisDrone.pth"

# MODEL_NAME = "YOLOX-tiny"
# MODEL_TYPE = "mmdet"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/YOLOx-tiny_Trained_VisDrone.pth"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/YOLOx-tiny_Trained_Heridal.pth"
# CONFIG_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/config.py"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/YOLOx_tiny_Trained_SARD.pth"
# CONFIG_PATH = USB_ROOT + "/models/YOLOX-tiny_VisDrone/config.py"
# WEIGHTS_PATH = USB_ROOT + "/models/YOLOX-tiny_VisDrone/YOLOX_tiny_Trained_VisDrone.pth"

MODEL_NAME = "YOLOv11n"
MODEL_TYPE = "yolo"
CONFIG_PATH = None
WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOv11n_VisDrone/1_Optimal/YOLOv11n_Trained_VisDrone.pt"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOv11n_HERIDAL/1_Optimal/YOLOv11n_Trained_HERIDAL.pt"
# WEIGHTS_PATH = "/home/sam_leahy/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOv11n_SARD/1_Optimal/YOLOv11n_Trained_SARD.pt"
# WEIGHTS_PATH = USB_ROOT + "/models/YOLOv11n_VisDrone/YOLOv11n_Trained_VisDrone.pt"

# ---- RUN METADATA --------------------------------------------
RUN_LABEL = "YOLOv11n_VisDrone_final"  # e.g. "visdrone_e50", "heridal_only_e30"

# Set to an integer (e.g. 20) for a quick debug run.
# Set to None to run the FULL test split (use this for real results).
MAX_IMAGES = None

# ---- EVAL CONFIG ----------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
IOU_MATCH_THRESH = 0.5  # for precision/recall/F1-at-threshold (separate from COCO mAP's own IoU sweep)
DEVICE = "cuda:0"

RESULTS_CSV = USB_ROOT + "/benchmark_results.csv"

# ---- SAMPLE IMAGE OUTPUT ----------------------------------------
SAVE_SAMPLE_IMAGES = True
NUM_SAMPLE_IMAGES = 8
SAMPLE_OUTPUT_ROOT = USB_ROOT + "/sample_outputs"
SAMPLE_SEED = 42  # fixed seed -- do not change between runs you want to compare

local_config = str(CONFIG_PATH)

# ---- END CONFIG -------------------------------------------------


def count_params(model):
    """Count trainable parameters in a torch model, in millions."""
    try:
        total = sum(p.numel() for p in model.parameters())
        return round(total / 1e6, 2)
    except Exception:
        return None


def load_coco_gt(ann_file):
    """
    Load ground truth annotations grouped by image_id.
    Returns (images_meta, gt_by_image) where gt_by_image maps
    image_id -> list of (bbox_xywh, category_id) tuples.
    category_id is the RAW COCO category id from the JSON -- callers
    must compare against CLASS_ID_TO_COCO_CAT-mapped prediction
    categories, not raw model class indices.
    """
    with open(ann_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    gt_by_image = {img_id: [] for img_id in images}
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append((ann["bbox"], ann["category_id"]))

    return images, gt_by_image


def iou_xywh(box_a, box_b):
    """IoU between two boxes in [x, y, w, h] format."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections_to_gt(preds, gts, iou_thresh=0.5):
    """
    Class-AWARE greedy matching of predicted boxes to ground truth.
    preds: list of (box_xywh, score, coco_category_id)
    gts:   list of (box_xywh, coco_category_id)
    A prediction only counts as a true positive if it matches a GT
    box of THE SAME category at IoU >= iou_thresh. This is the fix
    for the old script's bug, where IoU-only matching let a
    correctly-localised but wrong-class box (e.g. "vehicle" box
    overlapping a "person" GT box) count as a hit.
    Returns (tp, fp, fn) counts for one image.
    """
    matched_gt = set()
    tp = 0

    # Sort predictions by score descending so higher-confidence boxes
    # get first claim on a GT box during greedy matching.
    preds_sorted = sorted(preds, key=lambda p: p[1], reverse=True)

    for pred_box, _score, pred_cat in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, (gt_box, gt_cat) in enumerate(gts):
            if idx in matched_gt:
                continue
            if gt_cat != pred_cat:
                continue  # class-aware: skip GT of a different category entirely
            iou = iou_xywh(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)

    return tp, fp, fn


def main():
    print(f"\n{'='*60}")
    print(f"Benchmarking: {MODEL_NAME}  (run label: {RUN_LABEL})")
    print(f"Platform: {PLATFORM_LABEL}  |  Dataset: {DATASET_NAME}")
    print(f"{'='*60}")

    if not os.path.exists(USB_ROOT):
        print(f"ERROR: USB_ROOT not found at {USB_ROOT}")
        print("Plug in the USB drive and make sure it's mounted, then try again.")
        return
    if not os.path.exists(TEST_ANN_FILE):
        print(f"ERROR: Test annotation file not found at {TEST_ANN_FILE}")
        return

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda:0":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load ground truth ----
    images_meta, gt_by_image = load_coco_gt(TEST_ANN_FILE)
    image_ids = list(images_meta.keys())
    if MAX_IMAGES is not None:
        image_ids = image_ids[:MAX_IMAGES]
    print(f"Evaluating on {len(image_ids)} images "
          f"({'FULL test split' if MAX_IMAGES is None else 'SUBSET - debug mode'})")

    # ---- Deterministic sample selection for qualitative output ----
    sample_image_ids = set()
    if SAVE_SAMPLE_IMAGES and len(image_ids) > 0:
        rng = np.random.RandomState(SAMPLE_SEED)
        n_samples = min(NUM_SAMPLE_IMAGES, len(image_ids))
        candidate_ids = [iid for iid in image_ids if len(gt_by_image.get(iid, [])) > 0]
        if len(candidate_ids) == 0:
            candidate_ids = image_ids
        chosen = rng.choice(candidate_ids, size=min(n_samples, len(candidate_ids)), replace=False)
        sample_image_ids = set(chosen.tolist())

    # ---- Load model ----
    print(f"Loading model: {MODEL_NAME}")
    param_count_m = None

    if MODEL_TYPE == "mmdet":
        from mmdet.apis import DetInferencer
        inferencer = DetInferencer(
            model=local_config,
            weights=WEIGHTS_PATH,
            device=device,
            show_progress=False
        )
        try:
            param_count_m = count_params(inferencer.model)
        except Exception:
            pass

    elif MODEL_TYPE == "yolo":
        from ultralytics import YOLO
        inferencer = YOLO(WEIGHTS_PATH)
        try:
            param_count_m = count_params(inferencer.model)
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # ---- Warmup ----
    import cv2
    first_img_path = os.path.join(TEST_IMG_DIR, images_meta[image_ids[0]]["file_name"])
    warmup_frame = cv2.imread(first_img_path)
    for _ in range(2):
        if MODEL_TYPE == "mmdet":
            inferencer(warmup_frame, show=False)
        else:
            inferencer(warmup_frame, verbose=False)

    # ---- Sample output dir for this run ----
    sample_dir = None
    if SAVE_SAMPLE_IMAGES:
        run_tag = f"{PLATFORM_LABEL}_{MODEL_NAME}_{RUN_LABEL}".replace(" ", "_")
        sample_dir = os.path.join(SAMPLE_OUTPUT_ROOT, run_tag)
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Sample images will be saved to: {sample_dir}")

    # ---- Run inference over test set ----
    latencies_ms = []
    total_tp, total_fp, total_fn = 0, 0, 0

    # coco_format_preds accumulated directly (category_id now correct,
    # taken from the same CLASS_ID_TO_COCO_CAT mapping used for matching)
    coco_format_preds = []

    for i, img_id in enumerate(image_ids):
        meta = images_meta[img_id]
        img_path = os.path.join(TEST_IMG_DIR, meta["file_name"])
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        gt_entries = gt_by_image.get(img_id, [])  # list of (box, coco_category_id)

        start = time.perf_counter()

        # preds_this_image: list of (box_xywh, score, coco_category_id)
        preds_this_image = []
        preds_for_map = []

        if MODEL_TYPE == "mmdet":
            result = inferencer(frame, show=False)
            elapsed_ms = (time.perf_counter() - start) * 1000
            preds = result["predictions"][0]
            for bbox, score, label in zip(preds["bboxes"], preds["scores"], preds["labels"]):
                label = int(label)
                if label in VALID_CLASS_IDS:
                    x1, y1, x2, y2 = bbox
                    box_xywh = [x1, y1, x2 - x1, y2 - y1]
                    coco_cat = CLASS_ID_TO_COCO_CAT[label]
                    preds_for_map.append((box_xywh, float(score), coco_cat))
                    if score >= CONFIDENCE_THRESHOLD:
                        preds_this_image.append((box_xywh, float(score), coco_cat))

        elif MODEL_TYPE == "yolo":
            results = inferencer(frame, verbose=False, conf=0.001)
            elapsed_ms = (time.perf_counter() - start) * 1000
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])
                    if class_id in VALID_CLASS_IDS:
                        x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
                        box_xywh = [x1, y1, x2 - x1, y2 - y1]
                        coco_cat = CLASS_ID_TO_COCO_CAT[class_id]
                        preds_for_map.append((box_xywh, float(score), coco_cat))
                        if score >= CONFIDENCE_THRESHOLD:
                            preds_this_image.append((box_xywh, score, coco_cat))

        latencies_ms.append(elapsed_ms)

        tp, fp, fn = match_detections_to_gt(preds_this_image, gt_entries, iou_thresh=IOU_MATCH_THRESH)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for box_xywh, score, coco_cat in preds_for_map:
            coco_format_preds.append({
                "image_id": img_id,
                "category_id": coco_cat,
                "bbox": box_xywh,
                "score": score
            })

        # ---- Save annotated sample image if this one was selected ----
        if SAVE_SAMPLE_IMAGES and img_id in sample_image_ids:
            annotated = frame.copy()

            for gx_gy_gw_gh, _gt_cat in gt_entries:
                gx, gy, gw, gh = gx_gy_gw_gh
                x1, y1, x2, y2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(annotated, "GT", (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

            for box_xywh, score, coco_cat in preds_this_image:
                px, py, pw, ph = box_xywh
                x1, y1, x2, y2 = int(px), int(py), int(px + pw), int(py + ph)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{score:.2f}", (x1, y1 - 6 if y1 > 12 else y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(annotated, f"{PLATFORM_LABEL} | {MODEL_NAME} | {RUN_LABEL}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            out_name = f"img{img_id}_{meta['file_name']}"
            cv2.imwrite(os.path.join(sample_dir, out_name), annotated)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_ids):
            print(f"  Processed {i+1}/{len(image_ids)} images...")

    # ---- Speed metrics ----
    avg_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # ---- Precision / Recall / F1 at fixed threshold (now class-aware) ----
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ---- COCO-style mAP using pycocotools ----
    coco_map, coco_map50, coco_map75 = None, None, None
    coco_map_s, coco_map_m, coco_map_l = None, None, None

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(TEST_ANN_FILE)
        coco_dt = coco_gt.loadRes(coco_format_preds) if coco_format_preds else None

        if coco_dt is not None:
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            coco_map = round(float(coco_eval.stats[0]), 4)
            coco_map50 = round(float(coco_eval.stats[1]), 4)
            coco_map75 = round(float(coco_eval.stats[2]), 4)
            coco_map_s = round(float(coco_eval.stats[3]), 4)
            coco_map_m = round(float(coco_eval.stats[4]), 4)
            coco_map_l = round(float(coco_eval.stats[5]), 4)

            # Per-class AP breakdown, printed only (keeps CSV schema
            # stable across single-class and multi-class datasets)
            print(f"\nPer-class AP breakdown:")
            cat_ids = coco_gt.getCatIds()
            for cat_id in cat_ids:
                cat_name = coco_gt.loadCats([cat_id])[0]["name"]
                eval_copy = COCOeval(coco_gt, coco_dt, iouType="bbox")
                eval_copy.params.imgIds = image_ids
                eval_copy.params.catIds = [cat_id]
                eval_copy.evaluate()
                eval_copy.accumulate()
                ap = eval_copy.stats[0] if len(eval_copy.stats) else float("nan")
                ap50 = eval_copy.stats[1] if len(eval_copy.stats) else float("nan")
                print(f"  {cat_name:12s}  AP: {ap:.3f}   AP50: {ap50:.3f}")
        else:
            print("No predictions above threshold - skipping COCO mAP calculation")

    except Exception as e:
        print(f"COCO mAP calculation skipped due to error: {e}")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS: {MODEL_NAME} ({RUN_LABEL}) on {PLATFORM_LABEL} / {DATASET_NAME}")
    print(f"{'='*60}")
    print(f"Images evaluated:      {len(image_ids)}")
    print(f"Avg latency:           {avg_ms:.1f} ms  (std {std_ms:.1f} ms)")
    print(f"Avg FPS:                {avg_fps:.1f}")
    print(f"Precision @ {CONFIDENCE_THRESHOLD} / IoU {IOU_MATCH_THRESH} (class-aware): {precision:.3f}")
    print(f"Recall    @ {CONFIDENCE_THRESHOLD} / IoU {IOU_MATCH_THRESH} (class-aware): {recall:.3f}")
    print(f"F1 score:               {f1:.3f}")
    if coco_map is not None:
        print(f"COCO mAP [.5:.95]:      {coco_map}")
        print(f"COCO mAP@50:            {coco_map50}")
        print(f"COCO mAP@75:            {coco_map75}")
        print(f"COCO mAP (small):       {coco_map_s}")
        print(f"COCO mAP (medium):      {coco_map_m}")
        print(f"COCO mAP (large):       {coco_map_l}")
    print(f"Param count (M):        {param_count_m}")
    if SAVE_SAMPLE_IMAGES:
        print(f"Sample images saved:    {len(sample_image_ids)} -> {sample_dir}")
    print(f"{'='*60}")

    # ---- Append to CSV ----
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": PLATFORM_LABEL,
        "model_name": MODEL_NAME,
        "run_label": RUN_LABEL,
        "dataset_name": DATASET_NAME,
        "weights_path": WEIGHTS_PATH,
        "num_images": len(image_ids),
        "avg_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "avg_fps": round(avg_fps, 2),
        "precision_at_thresh": round(precision, 4),
        "recall_at_thresh": round(recall, 4),
        "f1_at_thresh": round(f1, 4),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "coco_mAP": coco_map,
        "coco_mAP_50": coco_map50,
        "coco_mAP_75": coco_map75,
        "coco_mAP_small": coco_map_s,
        "coco_mAP_medium": coco_map_m,
        "coco_mAP_large": coco_map_l,
        "param_count_millions": param_count_m,
        "device": device,
    }

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResult appended to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
