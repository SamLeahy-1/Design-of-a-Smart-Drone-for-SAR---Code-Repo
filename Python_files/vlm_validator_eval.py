#!/usr/bin/env python3
"""
vlm_validator_eval_v3.py  --  VLM validator scored on your labelled field frames.

Builds on evaluate_models2.py (keep it in the SAME folder). It reuses:
  * the Roboflow dataset loader (all splits merged, ignore regions, Test A/B source mapping)
  * the CACHED detections in results/predictions/  (so no detector, mmdet or GPU is needed here)
  * the SAME TP / FP / ignore matching, so numbers line up exactly with Chapter 4's field results

Only detections at >= THRESH that are not in ignore regions are sent to the VLM.
For each one the evidence image is SAVED BEFORE the VLM call, the answer is parsed
(first word yes/no; anything else = format failure = NOT confirmed, the fail-safe),
and everything is logged. VLM answers are cached on disk, so a crashed or repeated run resumes.

Prerequisite (once per detector, in the right venv):
    python3 evaluate_models2.py --models RTMDet-tiny_VisDrone

Usage:
    ollama pull moondream
    python3 vlm_validator_eval_v3.py --check
    python3 vlm_validator_eval_v3.py --limit 30                      # quick debug
    python3 vlm_validator_eval_v3.py --model RTMDet-tiny_VisDrone --modes full,squash
    python3 vlm_validator_eval_v3.py --model YOLOv11n_VisDrone --vlm qwen2.5vl:3b
"""
import os, re, csv, json, time, base64, hashlib, argparse, socket, statistics
from datetime import datetime

import cv2
import numpy as np
import requests

import evaluate_models2 as em

# ============================== SETTINGS ==============================
MODEL = "RTMDet-tiny_VisDrone"          # any key of em.MODEL_REGISTRY
MODES = ["full", "squash"]              # ultralytics models only have "full"
THRESH = 0.5                            # flight threshold

OLLAMA_URL = "http://localhost:11434/api/chat"
VLM_MODEL = "moondream"
VARIANTS = ["crop", "boxed"]
CTX_PROMPT = "Is there a real person at the centre of this image? Answer with exactly one word: yes or no."
PROMPTS = {
    "crop": "Does this image show a real human being? Answer with exactly one word: yes or no.",
    "boxed": "Is there a real person inside the red rectangle? Answer with exactly one word: yes or no.",
    "ctx3": CTX_PROMPT,      # crop with 3x box padding each side (more scene, no heavy upscaling)
    "ctx5": CTX_PROMPT,      # crop with 5x box padding each side
}
VARIANT_PAD = {"crop": 0.5, "ctx3": 3.0, "ctx5": 5.0}   # padding as a multiple of box size, each side
PAD_FRAC = 0.5          # crop padding as a fraction of box size, each side
MIN_CROP_SIDE = 224     # upscale small crops so the VLM has pixels to work with
VLM_TIMEOUT_S = 90
# =====================================================================


def safe(s):
    return re.sub(r"[^\w\-.]+", "_", s)


def make_crop(frame, b, pad=PAD_FRAC):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = b
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    a, c = int(max(0, x1 - px)), int(max(0, y1 - py))
    d, e = int(min(w, x2 + px)), int(min(h, y2 + py))
    crop = frame[c:e, a:d]
    short = min(crop.shape[:2])
    if short < MIN_CROP_SIDE:
        s = MIN_CROP_SIDE / max(short, 1)
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def make_boxed(frame, b):
    img = frame.copy()
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
    return img


NUM_PREDICT = 8
DESCRIBE = False    # describe-then-keyword mode (same rule as jetson_vlm_timing.py --describe)
DESCRIBE_PROMPT = "Describe this image in one short sentence."
PARSE = "keyword"   # "keyword": person word anywhere in the text | "last": last yes/no word in the text
TAG = ""            # suffix for the output folder, so a different prompt never overwrites earlier results
PERSON_WORDS = ("person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "human", "pedestrian",
                "hiker", "someone", "figure", "lady", "guy", "adult", "individual")
NUM_CTX = 8192      # larger context fixed a degenerate 'GGGGGGGG' output seen at the default (4096)
RETRIES = 2         # re-ask this many extra times when the answer is garbage (not yes/no)


def ask_vlm(img_path, prompt, vlm):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"model": vlm, "stream": False, "keep_alive": "10m",
               "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
               "messages": [{"role": "user", "content": prompt, "images": [b64]}]}
    t0 = time.perf_counter()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=VLM_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["message"]["content"], (time.perf_counter() - t0) * 1000, ""
    except Exception as e:
        return "", (time.perf_counter() - t0) * 1000, str(e)[:120]


def parse_answer(raw, err):
    if DESCRIBE:
        if err or not raw.strip():
            return "format_fail"
        words = re.findall(r"[a-z]+", raw.lower())
        if PARSE == "count":
            nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            for w in re.findall(r"[a-z]+|\d+", raw.lower()):
                if w in ("zero", "none", "no", "nobody", "0"):
                    return "no"
                if w.isdigit():
                    return "yes" if int(w) > 0 else "no"
                if w in nums or w in PERSON_WORDS:
                    return "yes"
            return "format_fail"
        if PARSE == "last":
            yn = [w for w in words if w in ("yes", "no")]
            return yn[-1] if yn else "format_fail"
        return "yes" if set(words) & set(PERSON_WORDS) else "no"
    if err:
        return "format_fail"
    words = re.findall(r"[a-z]+", raw.lower())
    return words[0] if words and words[0] in ("yes", "no") else "format_fail"


def run_one(model, mode, vlm, variants, images, info, limit, redo, source=None):
    cache = em.load_cache(model, mode)
    if not cache:
        print(f"[skip ] no cached detections for {model} [{mode}]. Run evaluate_models2.py for it first.")
        return
    recs = em.build_records(model, cache, images)
    if source:
        recs = [r for r in recs if source.lower() in (r['source'] or '').lower()]
    if limit:
        recs = recs[:limit]
    out_dir = os.path.join(em.RESULTS_DIR, "vlm", f"{safe(model)}__{mode}__{safe(vlm)}" + (f"__{safe(TAG)}" if TAG else ""))
    ev_dir = os.path.join(out_dir, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    cpath = os.path.join(out_dir, "vlm_cache.json")
    vcache = {} if redo or not os.path.isfile(cpath) else json.load(open(cpath))
    phash = {v: hashlib.md5(PROMPTS[v].encode()).hexdigest()[:6] for v in variants}
    sample_fps = info["sample_fps"]

    print(f"\n=== {model} [{mode}] | VLM {vlm} | conf>={THRESH} | {len(recs)} frames ===")
    ask_vlm_warm = os.path.join(out_dir, "_warm.jpg")
    cv2.imwrite(ask_vlm_warm, np.full((224, 224, 3), 127, np.uint8))
    ask_vlm(ask_vlm_warm, "Say yes.", vlm)          # warm-up, so no timing includes a model load

    rows, calls = [], 0
    for ri, r in enumerate(recs):
        im = images[r["key"]]
        frame = None
        for i in range(len(r["scores"])):
            if r["scores"][i] < THRESH:
                continue
            base = dict(source=im.get("source") or "?", clip=r["clip"], image_key=r["key"], det=i,
                        score=round(float(r["scores"][i]), 3))
            if r["ign"][em.OP_I][i]:
                rows.append({**base, "gt_label": "IGNORED", "variant": "-", "raw": "", "answer": "-",
                             "latency_ms": "", "err": "", "evidence": ""})
                continue
            label = "TP" if r["tp"][em.OP_I][i] else "FP"
            for v in variants:
                ev = os.path.join(ev_dir, f"{r['key']}_d{i}_{v}_{label}.jpg")
                if not os.path.isfile(ev):                       # evidence saved BEFORE the VLM call
                    if frame is None:
                        frame = cv2.imread(im["path"])
                    b = r["boxes"][i]
                    img = make_crop(frame, b, VARIANT_PAD[v]) if v in VARIANT_PAD else make_boxed(frame, b)
                    cv2.imwrite(ev, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                ck = f"{r['key']}|{i}|{v}|{phash[v]}"
                old = vcache.get(ck)
                if old is None or parse_answer(old["raw"], old["err"]) == "format_fail":
                    # new call, or a cached garbage answer from an earlier run: ask again
                    for attempt in range(RETRIES + 1):
                        raw, ms, err = ask_vlm(ev, PROMPTS[v], vlm)
                        calls += 1
                        if parse_answer(raw, err) != "format_fail":
                            break
                    vcache[ck] = {"raw": raw.strip().replace("\n", " ")[:200], "ms": round(ms, 1), "err": err,
                                  "tries": attempt + 1, "num_ctx": NUM_CTX}
                    if calls % 20 == 0:
                        json.dump(vcache, open(cpath, "w"))
                        print(f"   {ri + 1}/{len(recs)} frames, {calls} VLM calls", end="\r")
                c = vcache[ck]
                rows.append({**base, "gt_label": label, "variant": v, "raw": c["raw"],
                             "answer": parse_answer(c["raw"], c["err"]), "latency_ms": c["ms"],
                             "err": c["err"], "evidence": os.path.relpath(ev, out_dir)})
    json.dump(vcache, open(cpath, "w"))
    print()
    if not rows:
        print("no detections above threshold"); return

    with open(os.path.join(out_dir, "vlm_decisions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- summary ----------------
    scopes = ["ALL"] + sorted({r["source"] or "?" for r in recs})
    summ = []
    print(f"(frames are sampled at {sample_fps} fps, so FP/min = FP per frame x {sample_fps} x 60)")
    for scope in scopes:
        sc = [r for r in recs if scope == "ALL" or (r["source"] or "?") == scope]
        keys = {r["key"] for r in sc}
        n_gt = sum(r["n_gt"] for r in sc)
        n_fr = len(sc)
        n_ign = sum(1 for x in rows if x["variant"] == "-" and x["image_key"] in keys)
        print(f"\n--- {scope}: {n_fr} frames, {n_gt} person boxes, {n_ign} detections in ignore regions (excluded) ---")
        for v in variants:
            vr = [x for x in rows if x["variant"] == v and x["image_key"] in keys]
            tp = [x for x in vr if x["gt_label"] == "TP"]
            fp = [x for x in vr if x["gt_label"] == "FP"]
            ktp = sum(x["answer"] == "yes" for x in tp)
            kfp = sum(x["answer"] == "yes" for x in fp)
            fmt_fail = sum(x["answer"] == "format_fail" for x in vr)
            lat = [x["latency_ms"] for x in vr if x["latency_ms"] != ""]
            med = statistics.median(lat) if lat else float("nan")
            p0 = len(tp) / (len(tp) + len(fp)) if (tp or fp) else float("nan")
            p1 = ktp / (ktp + kfp) if (ktp + kfp) else float("nan")
            r0 = len(tp) / n_gt if n_gt else float("nan")
            r1 = ktp / n_gt if n_gt else float("nan")
            fm0 = len(fp) / max(n_fr, 1) * sample_fps * 60
            fm1 = kfp / max(n_fr, 1) * sample_fps * 60
            print(f"[{v}] TP {len(tp)} FP {len(fp)} | false alarms removed {len(fp) - kfp}/{len(fp)} | "
                  f"true detections lost {len(tp) - ktp}/{len(tp)} | format failures {fmt_fail}/{len(vr)} | "
                  f"median {med:.0f} ms")
            print(f"      precision {p0:.3f} -> {p1:.3f} | recall {r0:.3f} -> {r1:.3f} | FP/min {fm0:.1f} -> {fm1:.1f}")
            summ.append(dict(model=model, mode=mode, vlm=vlm, scope=scope, variant=v, frames=n_fr, n_gt=n_gt,
                             tp=len(tp), fp=len(fp), fp_removed=len(fp) - kfp, tp_lost=len(tp) - ktp,
                             format_fail=fmt_fail, n_calls=len(vr), median_latency_ms=round(med, 1),
                             precision_before=p0, precision_after=p1, recall_before=r0, recall_after=r1,
                             fp_per_min_before=fm0, fp_per_min_after=fm1))
    with open(os.path.join(out_dir, "vlm_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    json.dump({"run": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
               "model": model, "mode": mode, "vlm": vlm, "thresh": THRESH, "prompts": PROMPTS,
               "pad_frac": PAD_FRAC, "variant_pad": VARIANT_PAD, "min_crop_side": MIN_CROP_SIDE, "sample_fps": sample_fps,
               "detector_cache": cache["meta"]}, open(os.path.join(out_dir, "run_info.json"), "w"),
              indent=2, default=str)
    print(f"\nOutputs in {out_dir}  (vlm_decisions.csv, vlm_summary.csv, evidence/)")


def main():
    global NUM_CTX, NUM_PREDICT, DESCRIBE, DESCRIBE_PROMPT, TAG, PARSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--vlm", default=VLM_MODEL)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (debug)")
    ap.add_argument("--redo", action="store_true", help="ignore the saved VLM answers")
    ap.add_argument("--source", default=None, help="only frames whose source contains this text, e.g. TestA or TestB")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX, help="Ollama context size (default 8192)")
    ap.add_argument("--describe", action="store_true", help="describe-then-keyword mode (for moondream and other models that will not answer yes/no)")
    ap.add_argument("--prompt", default=None, help="with --describe: replace the describe question (output goes to a separate folder)")
    ap.add_argument("--parse", choices=["keyword", "last", "count"], default="keyword", help="how to turn the reply into yes/no (default keyword)")
    ap.add_argument("--num-predict", type=int, default=32, help="max reply tokens in --describe mode")
    ap.add_argument("--tag", default=None, help="name suffix for the output folder")
    ap.add_argument("--check", action="store_true", help="dataset + cache status, no VLM calls")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    if a.dataset:
        em.DATASET_DIR = os.path.expanduser(a.dataset)
    if a.results:
        em.RESULTS_DIR = os.path.expanduser(a.results)

    NUM_CTX = a.num_ctx
    if a.describe:
        DESCRIBE, NUM_PREDICT, PARSE = True, a.num_predict, a.parse
        if a.prompt:
            DESCRIBE_PROMPT = a.prompt
            TAG = a.tag or "p" + hashlib.md5(a.prompt.encode()).hexdigest()[:4]
        elif a.tag:
            TAG = a.tag
        for k in PROMPTS:
            PROMPTS[k] = DESCRIBE_PROMPT
    images, info = em.load_dataset()
    if a.model not in em.MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {a.model}. Known: {list(em.MODEL_REGISTRY)}")
    modes = [m.strip() for m in a.modes.split(",")]
    variants = [v.strip() for v in a.variants.split(",")]

    if a.check:
        em.do_check(images, info, [a.model])
        for m in modes:
            c = em.load_cache(a.model, m)
            print(f"  detections cached for {a.model} [{m}]: {'yes, ' + str(len(c['dets'])) + ' frames' if c else 'NO'}")
        try:
            r = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5).json()
            print("  Ollama models:", [m["name"] for m in r.get("models", [])])
        except Exception as e:
            print("  Ollama not reachable:", e)
        return

    for m in modes:
        run_one(a.model, m, a.vlm, variants, images, info, a.limit, a.redo, a.source)


if __name__ == "__main__":
    main()
