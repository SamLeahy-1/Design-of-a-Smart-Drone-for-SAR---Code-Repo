#!/usr/bin/env python3
"""
SAR Drone - V6 GPS Control + Active Vision (Detection-Triggered Hover)
==============================================================================
V4_Vision_Passive has been verified. This script is cleared for flight testing.

REVISION r2 (field-readiness fixes -- V6 logic otherwise unchanged):
  1. Detection backlog fixed. The vision thread queues a detection ~10x/s. The
     queue is now drained after every hover and throughout the cooldown, and a
     queued detection older than DETECTION_MAX_AGE_S is discarded, so a hover can
     only be triggered by a person seen in the last ~1 s.
  2. Legs fly along the nose heading captured at engage (ALIGN_LEGS_TO_HEADING)
     instead of geographic north, so the camera looks along the search path.
     Set ALIGN_LEGS_TO_HEADING = False to restore the old north-only behaviour.
  3. GPS glitch guard recovers: after GPS_REBASE_AFTER_N_BAD_TICKS consecutive
     rejected ticks the reference position is re-based (the jump is NOT counted
     as distance) instead of rejecting every later tick forever.
  4. Camera opened with V4L2 + MJPG 1280x720@30 + buffer size 1; the negotiated
     mode is printed, with a warning if it differs from the request.
  5. Vision start-up is checked: any exception loading the model/camera is
     caught, a warm-up inference is run, and the script refuses to enter standby
     (or start a run) unless the vision pipeline is actually running.
  6. mmdet boxes are scaled from 640x640 back to camera-frame pixels so saved
     detection images / logged bbox line up with the person.
  MODEL_NAME / MODEL_CONFIGS / distances are untouched -- keep your Jetson values.

Changes from V5:
  - Battery % display fixed: MAVSDK returns 0.0-1.0 float; multiplied correctly.
  - Per-flight terminal log (.txt) saved alongside run JSON -- mirrors exactly
    what prints to SSH, timestamped, one file per trigger cycle.
  - Run JSON enriched: flight_duration_s, total_distance_m, battery min/max
    voltage during run, gps_satellites_min, config snapshot, post_engage_status.
  - STATUS_LOG_PATH now per-session (timestamped filename) not a shared append.
  - AXIS_SPEED_MPS raised to 1.39 m/s (5 km/h) for open-field testing.
    Adjust back to 0.15 for confined/indoor testing.
  - YOLOX-tiny config path flagged with a TODO -- was pointing at RTMDet config.

On a qualifying detection (confidence >= CONFIDENCE_THRESHOLD, class == person):
  - The drone immediately hovers in place (zero velocity, hold yaw)
  - Plays a detection beep
  - Prints detection info to terminal (SSH) and to per-flight .txt log
  - Saves the annotated detection frame
  - Logs the detection to detections_log.jsonl
  - Hovers for DETECTION_HOVER_S seconds
  - Resumes the GPS leg from the current position toward the remaining distance
  - A DETECTION_COOLDOWN_S window after each hover prevents re-triggering

FLIGHT SEQUENCE:
  Forward  : GPS-confirmed FORWARD_DISTANCE_M, pausing on detections
  Pause    : PAUSE_BETWEEN_S hover (detections still logged, not actioned)
  Backward : GPS-confirmed BACKWARD_DISTANCE_M, pausing on detections
  Hold     : hover in GUIDED until pilot flips SWA back

SAFETY (identical to V4/V5):
  - Mode-abort check every ~50ms tick, unconditionally highest priority.
  - Abort check runs inside detection hovers too.
  - Vision thread never touches the abort path.
  - Pre-engage EKF health warning if position estimate not OK.

MODEL SELECTION: set MODEL_NAME to any key in MODEL_CONFIGS.

ENVIRONMENT:
  MMDet models  : source ~/mmdet_env/bin/activate   (+ pip install mavsdk)
  YOLO models   : source ~/drone_env/bin/activate    (+ pip install mavsdk)
"""

import asyncio
import io
import json
import math
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.telemetry import FlightMode
from mavsdk.tune import TuneDescription, SongElement

# ============================================================
# ---- CONFIGURATION ------------------------------------------
# ============================================================

IS_SIMULATION = False

SIM_ADDRESS          = "udpin://0.0.0.0:14540"
HARDWARE_ADDRESS     = "serial:///dev/ttyTHS1:115200"
CONNECTION_ADDRESS   = SIM_ADDRESS if IS_SIMULATION else HARDWARE_ADDRESS
CONNECTION_TIMEOUT_S = 30

TARGET_TRIGGER_MODE = FlightMode.OFFBOARD
TRIGGER_DEBOUNCE_S  = 0.5

# --- GPS flight path ---
# For open-field testing at ~5 km/h use 1.39 m/s.
# For confined/backyard testing use 0.15 m/s.
FORWARD_DISTANCE_M     = 10.0   # increase for open-field runs
BACKWARD_DISTANCE_M    = 10.0
AXIS_SPEED_MPS         = 1.39   # ~5 km/h -- suitable for open field
PAUSE_BETWEEN_S        = 2.0
POSITION_LOOP_PERIOD_S = 0.05
MAX_PLAUSIBLE_TICK_M   = 2.0

# --- r2 additions -------------------------------------------------------
ALIGN_LEGS_TO_HEADING        = True   # fly along nose heading at engage (False = old: geographic north)
GPS_REBASE_AFTER_N_BAD_TICKS = 3      # consecutive rejected GPS ticks before the reference is re-based

# --- Detection-triggered hover ---
DETECTION_HOVER_S    = 5.0    # how long to hover on a detection
DETECTION_COOLDOWN_S = 3.0    # ignore new detections for this long after a hover ends
DETECTION_MAX_AGE_S  = 1.0    # r2: a queued detection older than this is stale and ignored

# --- Status logging ---
STATUS_LOG_PERIOD_S = 1.0

# --- Vision ---
MODEL_NAME           = "YOLOv11n_HERIDAL"
CAMERA_SOURCE        = 0
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID      = 0
INFERENCE_TARGET_HZ  = 10
SAMPLE_FRAME_EVERY_N = 30

# --- r2: camera mode (J.Burrows webcam needs MJPG for 720p30 over USB 2.0) ---
CAMERA_USE_MJPG        = True
CAMERA_WIDTH           = 1280
CAMERA_HEIGHT          = 720
CAMERA_FPS             = 30
VISION_READY_TIMEOUT_S = 180    # max wait for model + camera at start-up

# Copied verbatim from jetson_vision_test.py -- keep in sync.
MODEL_CONFIGS = {
    "YOLOv11n": {
        "type": "ultralytics",
        "weights": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/YOLOv11n_UNTRAINED/yolo11n.pt",
        "config": "none",
    },
    "YOLOv11n_SARD": {
        "type": "ultralytics",
        "weights": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOv11n_SARD/1_Optimal/YOLOv11n_Trained_SARD.pt",
        "config": "none",
    },
    "YOLOv11n_HERIDAL": {
        "type": "ultralytics",
        "weights": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOv11n_HERIDAL/1_Optimal/YOLOv11n_Trained_HERIDAL.pt",
        "config": "none",
    },
    "YOLOv11n_VisDrone": {
        "type": "ultralytics",
        "weights": "/home/sam/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOv11n_VisDrone/1_Optimal/YOLOv11n_Trained_VisDrone.pt",
        "config": "none",
    },
    "YOLOX-tiny": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/YOLOx-tiny_UNTRAINED/yolox_tiny.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/YOLOx-tiny_UNTRAINED/yolox_tiny_8xb8-300e_coco.py",
    },
    "YOLOX-tiny_SARD": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/YOLOx_tiny_Trained_SARD.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/YOLOx-tiny_SARD/1_Optimal/config.py",
    },
    "YOLOX-tiny_HERIDAL": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/YOLOx-tiny_Trained_Heridal.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/YOLOx-tiny_HERIDAL/1_Optimal/config.py",
    },
    "YOLOX-tiny_VisDrone": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/YOLOx-tiny_Trained_VisDrone.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/YOLOx-tiny_VisDrone/1_Optimal/config.py",
    },
    "RTMDet-tiny": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/RTMDet-tiny_UNTRAINED/rtmdet_tiny.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/RTMDet-tiny_UNTRAINED/rtmdet_tiny_8xb32-300e_coco.py",
    },
    "RTMDet-tiny_SARD": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/RTMDet_tiny_Trained_SARD.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/RTMDet-tiny_SARD/1_Optimal/config.py",
    },
    "RTMDet-tiny_HERIDAL": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/RTMDet-tiny_Trained_HERIDAL.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/RTMDet-tiny_HERIDAL/1_Optimal/config.py",
    },
    "RTMDet-tiny_VisDrone": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/RTMDet_tiny_Trained_VisDrone.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/3_VisDrone_Trained_Models/RTMDet-tiny_VisDrone/1_Optimal/config.py",
    },
    "Faster-RCNN-R50": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/FasterRCNN-50_UNTRAINED/faster_rcnn_r50.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/1_Untrained_Models/FasterRCNN-50_UNTRAINED/faster-rcnn_r50_fpn_1x_coco.py",
    },
    "Faster-RCNN-R50_SARD": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/FasterRCNN_Trained_SARD.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/2_SARD_Trained_Models/FasterRCNN-50_SARD/1_Optimal/config.py",
    },
    "Faster-RCNN-R50_HERIDAL": {
        "type": "mmdet",
        "weights": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/FasterRCNN_Trained_HERIDAL.pth",
        "config": "/home/sam/1_AI_Training_Benchmarking/4_Heridal_Trained_Models/FasterRCNN-50_HERIDAL/1_Optimal/config.py",
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v6_vision_active_logs")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

# Set at runtime in run() -- unique per session.
STATUS_LOG_PATH = None

TUNE_AUTO_ENGAGED  = TuneDescription([SongElement.NOTE_C, SongElement.NOTE_E, SongElement.NOTE_G], 200)
TUNE_MANUAL_RETURN = TuneDescription([SongElement.NOTE_G, SongElement.NOTE_E, SongElement.NOTE_C], 200)
TUNE_DETECTION     = TuneDescription([SongElement.NOTE_A, SongElement.NOTE_A, SongElement.NOTE_A], 150)

# ============================================================
# ---- SHARED STATE -------------------------------------------
# ============================================================

class TelemetryCache:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.data = {}

telemetry_cache = TelemetryCache()

class ModeState:
    def __init__(self):
        self.current_mode = None
        self.is_armed     = False

mode_state = ModeState()

class StatusState:
    def __init__(self):
        self.battery_voltage_v     = None
        self.battery_remaining_pct = None   # raw 0.0-1.0 float from MAVSDK
        self.gps_fix_type          = None
        self.gps_num_satellites    = None
        self.is_global_position_ok = None
        self.is_local_position_ok  = None
        self.is_home_position_ok   = None

status_state = StatusState()

detection_queue   = queue.Queue()
vision_stop_event = threading.Event()
vision_ready      = threading.Event()   # r2: set once camera is streaming + model warm
vision_failed     = threading.Event()   # r2: set if camera/model failed to start


# ============================================================
# ---- FLIGHT LOGGER (per-run .txt mirror of SSH output) ------
# ============================================================

class FlightLogger:
    """
    Wraps print() so every message is also written to a per-flight .txt file.
    Use flog.print() instead of print() for anything that should appear in
    both the terminal and the flight log.  The file is closed and finalised
    when close() is called at the end of each trigger cycle.
    """
    def __init__(self, path: str):
        self._path = path
        self._fh   = open(path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        header = (
            f"SAR Drone V6 r2 -- Flight Log\n"
            f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model   : {MODEL_NAME}\n"
            f"Speed   : {AXIS_SPEED_MPS} m/s  ({AXIS_SPEED_MPS * 3.6:.1f} km/h)\n"
            f"Path    : {FORWARD_DISTANCE_M}m fwd / {BACKWARD_DISTANCE_M}m back\n"
            f"{'=' * 70}\n"
        )
        self._fh.write(header)
        self._fh.flush()

    def print(self, *args, **kwargs):
        """Write to stdout and to the log file."""
        msg = " ".join(str(a) for a in args)
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        print(line)
        with self._lock:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self):
        try:
            self._fh.write(f"\n{'=' * 70}\n"
                           f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._fh.close()
        except Exception:
            pass

# Module-level logger placeholder; replaced per trigger cycle in run_takeover_sequence.
_flog: FlightLogger | None = None

def flog_print(*args, **kwargs):
    """Safe wrapper: uses per-flight logger if active, otherwise plain print."""
    if _flog is not None:
        _flog.print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# ============================================================
# ---- MODEL INFERENCER (identical to jetson_vision_test.py) --
# ============================================================

class ModelInferencer:
    def __init__(self, model_name: str):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Options: {list(MODEL_CONFIGS.keys())}")
        cfg = MODEL_CONFIGS[model_name]
        self.model_type = cfg["type"]
        self.model_name = model_name
        print(f"[VISION] Loading {model_name} ({self.model_type})...")

        if self.model_type == "ultralytics":
            from ultralytics import YOLO
            self._model = YOLO(cfg["weights"])

        elif self.model_type == "mmdet":
            from mmdet.apis import DetInferencer
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.6)
            self._model = DetInferencer(
                model=cfg["config"],
                weights=cfg["weights"],
                device="cuda:0",
                show_progress=False,
            )
        print(f"[VISION] {model_name} loaded.")

    def warmup(self, n: int = 2):
        """r2: a few dummy inferences so CUDA/cuDNN initialisation happens on the
        ground, not on the first flight frame."""
        dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        t0 = time.monotonic()
        for _ in range(n):
            self.infer(dummy)
        print(f"[VISION] Warm-up done ({n} inferences, "
              f"{(time.monotonic() - t0) / n * 1000:.0f} ms avg incl. first-call overhead).")

    def infer(self, frame):
        best_score = 0.0
        best_box   = None

        if self.model_type == "ultralytics":
            results = self._model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == PERSON_CLASS_ID:
                        score = float(box.conf[0])
                        if score > CONFIDENCE_THRESHOLD and score > best_score:
                            best_score = score
                            best_box   = list(map(int, box.xyxy[0]))

        elif self.model_type == "mmdet":
            fh, fw  = frame.shape[:2]
            sx, sy  = fw / 640.0, fh / 640.0   # r2: map 640x640 boxes back to frame pixels
            resized = cv2.resize(frame, (640, 640))
            results = self._model(resized, show=False)
            preds   = results["predictions"][0]
            for bbox, score, label in zip(
                preds["bboxes"], preds["scores"], preds["labels"]
            ):
                if label == PERSON_CLASS_ID and score > CONFIDENCE_THRESHOLD:
                    if score > best_score:
                        best_score = score
                        best_box   = [int(bbox[0] * sx), int(bbox[1] * sy),
                                      int(bbox[2] * sx), int(bbox[3] * sy)]

        return best_score, best_box


class FrameSource:
    def __init__(self, source):
        self.source     = source
        self.is_file    = isinstance(source, str)
        self.negotiated = None   # (width, height, fps, fourcc) the camera actually delivers
        if self.is_file:
            self.capture = cv2.VideoCapture(source)
        else:
            # r2: V4L2 backend so MJPG 720p30 can be requested.
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                print("[VISION WARN] V4L2 open failed -- falling back to default backend.")
                self.capture.release()
                self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera: {source}")
        if not self.is_file:
            self._configure_camera()

    def _configure_camera(self):
        cap = self.capture
        if CAMERA_USE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # avoid reading stale queued frames
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        self.negotiated = (w, h, fps, fourcc)
        print(f"[VISION] Camera mode: {w}x{h} @ {fps:.0f}fps  fourcc={fourcc or '?'}")
        if (w, h) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            print(f"[VISION WARN] Requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} but camera reports "
                  f"{w}x{h}. Check USB port/cable and MJPG support "
                  f"(v4l2-ctl --list-formats-ext).")

    def read(self):
        ret, frame = self.capture.read()
        if not ret and self.is_file:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        self.capture.release()


# ============================================================
# ---- VISION WORKER THREAD -----------------------------------
# ============================================================

def vision_worker():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    det_log_path = os.path.join(OUTPUT_DIR, "detections_log.jsonl")

    try:
        source = FrameSource(CAMERA_SOURCE)
    except Exception as e:                          # r2: was RuntimeError only
        print(f"[VISION ERROR] Camera: {e!r}")
        vision_failed.set()
        return

    try:
        inferencer = ModelInferencer(MODEL_NAME)
        inferencer.warmup()
    except Exception as e:                          # r2: was (ValueError, ImportError); a bad
        print(f"[VISION ERROR] Model load: {e!r}")  # path raises FileNotFoundError/OSError/...
        vision_failed.set()
        source.release()
        return

    min_period   = 1.0 / INFERENCE_TARGET_HZ
    frame_count  = 0
    detect_count = 0
    last_fps_t   = time.monotonic()

    print(f"[VISION] Worker running. {MODEL_NAME} @ up to {INFERENCE_TARGET_HZ}Hz.")

    try:
        while not vision_stop_event.is_set():
            t0 = time.monotonic()

            ok, frame = source.read()
            if not ok:
                print("[VISION WARN] Frame grab failed.")
                time.sleep(0.1)
                continue

            frame_count += 1

            if not vision_ready.is_set():
                vision_ready.set()
                print("[VISION] READY -- camera streaming, model warm.")

            try:
                best_score, best_box = inferencer.infer(frame)
            except Exception as e:
                print(f"[VISION ERROR] Inference: {e}")
                best_score, best_box = 0.0, None

            now   = time.time()
            ts_ms = int(now * 1000)

            # Sample frame -- saves periodically regardless of detection.
            if frame_count % SAMPLE_FRAME_EVERY_N == 0:
                cv2.imwrite(os.path.join(FRAMES_DIR, f"sample_{ts_ms}.jpg"), frame)

            if best_score >= CONFIDENCE_THRESHOLD and best_box is not None:
                detect_count += 1

                annotated = frame.copy()
                x1, y1, x2, y2 = best_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Person {best_score:.2f}",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                det_path = os.path.join(FRAMES_DIR,
                                        f"detection_{ts_ms}_{best_score:.2f}.jpg")
                cv2.imwrite(det_path, annotated)

                record = {
                    "timestamp":  now,
                    "model":      MODEL_NAME,
                    "confidence": round(best_score, 4),
                    "bbox":       best_box,
                    "frame_path": det_path,
                    "frame_num":  frame_count,
                }
                with open(det_log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                detection_queue.put(record)

            elapsed_fps = time.monotonic() - last_fps_t
            if elapsed_fps >= 5.0:
                fps = frame_count / elapsed_fps
                print(f"[VISION] {fps:.1f} fps | {detect_count} detections so far")
                frame_count  = 0
                detect_count = 0
                last_fps_t   = time.monotonic()

            time.sleep(max(0.0, min_period - (time.monotonic() - t0)))

    finally:
        source.release()
        print("[VISION] Worker thread stopped.")


# ============================================================
# ---- HAVERSINE ----------------------------------------------
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R       = 6_371_000.0
    phi1    = math.radians(lat1)
    phi2    = math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# ---- BACKGROUND ASYNC TASKS ---------------------------------
# ============================================================

async def play_audio_feedback(drone: System, tune: TuneDescription):
    try:
        await drone.tune.play_tune(tune)
    except Exception as e:
        print(f"[AUDIO WARN] {e}")

async def telemetry_cache_task(drone: System):
    async def watch_attitude():
        async for att in drone.telemetry.attitude_euler():
            async with telemetry_cache.lock:
                telemetry_cache.data["yaw_deg"] = att.yaw_deg
    async def watch_position():
        async for pos in drone.telemetry.position():
            async with telemetry_cache.lock:
                telemetry_cache.data["lat_deg"] = pos.latitude_deg
                telemetry_cache.data["lon_deg"] = pos.longitude_deg
    await asyncio.gather(watch_attitude(), watch_position())

async def mode_watchdog_task(drone: System):
    async for mode in drone.telemetry.flight_mode():
        if mode_state.current_mode != mode:
            print(f"\n[CONTROL] Flight mode: {mode}")
        mode_state.current_mode = mode

async def armed_watchdog_task(drone: System):
    async for armed in drone.telemetry.armed():
        mode_state.is_armed = armed

async def battery_watchdog_task(drone: System):
    async for battery in drone.telemetry.battery():
        prev_v = status_state.battery_voltage_v
        status_state.battery_voltage_v     = battery.voltage_v
        status_state.battery_remaining_pct = battery.remaining_percent  # 0.0-1.0
        if prev_v is None or abs(prev_v - battery.voltage_v) > 0.3:
            # remaining_percent is 0.0-1.0 from MAVSDK -- multiply by 100 to display.
            pct_display = battery.remaining_percent * 100
            print(f"\n[BATTERY] {battery.voltage_v:.2f}V "
                  f"({pct_display:.0f}% remaining)")

async def gps_watchdog_task(drone: System):
    async for gps in drone.telemetry.gps_info():
        prev = status_state.gps_fix_type
        status_state.gps_fix_type       = str(gps.fix_type)
        status_state.gps_num_satellites = gps.num_satellites
        if prev != status_state.gps_fix_type:
            print(f"\n[GPS] fix={gps.fix_type} satellites={gps.num_satellites}")

async def health_watchdog_task(drone: System):
    async for health in drone.telemetry.health():
        changed = (
            status_state.is_global_position_ok != health.is_global_position_ok
            or status_state.is_local_position_ok != health.is_local_position_ok
            or status_state.is_home_position_ok  != health.is_home_position_ok
        )
        status_state.is_global_position_ok = health.is_global_position_ok
        status_state.is_local_position_ok  = health.is_local_position_ok
        status_state.is_home_position_ok   = health.is_home_position_ok
        if changed:
            print(f"\n[HEALTH] global_pos_ok={health.is_global_position_ok} "
                  f"local_pos_ok={health.is_local_position_ok} "
                  f"home_pos_ok={health.is_home_position_ok}")
            if not health.is_global_position_ok or not health.is_local_position_ok:
                print("[HEALTH WARNING] Position estimate not OK -- "
                      "RTL/LOITER may be rejected by the FC.")

async def status_log_task():
    while True:
        async with telemetry_cache.lock:
            lat = telemetry_cache.data.get("lat_deg")
            lon = telemetry_cache.data.get("lon_deg")
        snapshot = {
            "t":                     time.time(),
            "flight_mode":           str(mode_state.current_mode),
            "is_armed":              mode_state.is_armed,
            "battery_voltage_v":     status_state.battery_voltage_v,
            # Store as 0-100 in the log for readability.
            "battery_remaining_pct": (
                round(status_state.battery_remaining_pct * 100, 1)
                if status_state.battery_remaining_pct is not None else None
            ),
            "gps_fix_type":          status_state.gps_fix_type,
            "gps_num_satellites":    status_state.gps_num_satellites,
            "is_global_position_ok": status_state.is_global_position_ok,
            "is_local_position_ok":  status_state.is_local_position_ok,
            "is_home_position_ok":   status_state.is_home_position_ok,
            "lat_deg":               lat,
            "lon_deg":               lon,
        }
        try:
            with open(STATUS_LOG_PATH, "a") as f:
                f.write(json.dumps(snapshot) + "\n")
        except Exception as e:
            print(f"[STATUS LOG WARN] {e}")
        await asyncio.sleep(STATUS_LOG_PERIOD_S)

async def wait_for_sustained_mode(target_mode: FlightMode, duration: float) -> bool:
    start = time.monotonic()
    while (time.monotonic() - start) < duration:
        if mode_state.current_mode != target_mode:
            return False
        await asyncio.sleep(0.05)
    return True

async def wait_for_connection(drone: System):
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[CONTROL] Flight controller link confirmed.")
            return

async def start_offboard_with_retry(drone: System, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        try:
            await drone.offboard.start()
            return True
        except OffboardError as e:
            print(f"[CONTROL] Offboard start retry: {e}")
            await asyncio.sleep(0.1)
    return False

async def _stop_offboard(drone: System):
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass

async def _get_current_position():
    async with telemetry_cache.lock:
        lat = telemetry_cache.data.get("lat_deg")
        lon = telemetry_cache.data.get("lon_deg")
    return lat, lon


# ============================================================
# ---- DETECTION QUEUE HELPERS (r2) ---------------------------
# ============================================================

def _drain_detection_queue() -> int:
    """Discard everything currently queued. Returns the number discarded."""
    n = 0
    while True:
        try:
            detection_queue.get_nowait()
            n += 1
        except queue.Empty:
            return n

def _get_fresh_detection():
    """
    Pop everything queued and return the NEWEST detection if it is no older than
    DETECTION_MAX_AGE_S, else None. Anything older is backlog from while the drone
    was hovering / cooling down and must never trigger a hover.
    """
    latest = None
    while True:
        try:
            latest = detection_queue.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return None
    if (time.time() - latest["timestamp"]) > DETECTION_MAX_AGE_S:
        return None
    return latest


# ============================================================
# ---- DETECTION HOVER ----------------------------------------
# ============================================================

async def run_detection_hover(drone: System, det: dict, yaw_deg: float,
                               distance_so_far: float,
                               run_detections: list) -> str:
    flog_print(f"\n[VISION] *** PERSON DETECTED ***  "
               f"conf={det['confidence']:.2f}  bbox={det['bbox']}")
    flog_print(f"[VISION] Hovering {DETECTION_HOVER_S}s at {distance_so_far:.2f}m into leg.")
    flog_print(f"[VISION] Frame saved: {os.path.basename(det['frame_path'])}")

    asyncio.create_task(play_audio_feedback(drone, TUNE_DETECTION))

    run_detections.append({
        **det,
        "distance_at_detect_m": distance_so_far,
        "hover_duration_s":     DETECTION_HOVER_S,
    })

    hover_start = time.monotonic()
    while (time.monotonic() - hover_start) < DETECTION_HOVER_S:
        if mode_state.current_mode != TARGET_TRIGGER_MODE:
            flog_print("[SAFETY] Mode changed during detection hover. Aborting.")
            return "aborted"
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw_deg))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)

    flog_print("[VISION] Hover complete. Resuming leg.")
    return "completed"


# ============================================================
# ---- GPS LEG WITH ACTIVE DETECTION HOVER --------------------
# ============================================================

async def run_gps_leg(drone: System, label: str, north_vel: float, east_vel: float,
                      yaw_deg: float, target_distance_m: float,
                      run_detections: list, log: list,
                      volt_samples: list) -> str:
    start_lat, start_lon = await _get_current_position()
    if start_lat is None or start_lon is None:
        flog_print(f"[CONTROL ERROR] No GPS at start of '{label}'. Aborting.")
        log.append({"step": label, "status": "aborted_no_gps", "t": time.time()})
        return "aborted"

    flog_print(f"\n[SEQUENCE] {label} -- target={target_distance_m}m "
               f"vN={north_vel:+.2f} vE={east_vel:+.2f} m/s  yaw={yaw_deg:.1f}°")
    flog_print(f"[SEQUENCE] Start GPS: lat={start_lat:.7f} lon={start_lon:.7f}")

    distance_traveled = 0.0
    last_lat, last_lon = start_lat, start_lon
    last_print_t       = time.monotonic()
    cooldown_until     = 0.0
    hover_count        = 0
    bad_ticks          = 0     # r2: consecutive rejected GPS ticks
    gps_rebases        = 0     # r2: how many times the GPS reference was re-based

    while distance_traveled < target_distance_m:

        # --- ABORT: unconditional, always first ---
        if mode_state.current_mode != TARGET_TRIGGER_MODE:
            flog_print(f"[SAFETY] Mode changed during '{label}'. "
                       f"Traveled {distance_traveled:.2f}m. Aborting.")
            log.append({
                "step": label, "status": "aborted", "t": time.time(),
                "distance_traveled_m": distance_traveled,
                "battery_voltage_v": status_state.battery_voltage_v,
                "gps_fix_type": status_state.gps_fix_type,
                "is_global_position_ok": status_state.is_global_position_ok,
                "gps_rebases": gps_rebases,
            })
            return "aborted"

        # --- Battery voltage sample (for min/max summary) ---
        if status_state.battery_voltage_v is not None:
            volt_samples.append(status_state.battery_voltage_v)

        # --- GPS accumulation ---
        cur_lat, cur_lon = await _get_current_position()
        if cur_lat is not None and cur_lon is not None:
            tick_dist = haversine_distance(last_lat, last_lon, cur_lat, cur_lon)
            if tick_dist < MAX_PLAUSIBLE_TICK_M:
                distance_traveled += tick_dist
                last_lat, last_lon = cur_lat, cur_lon
                bad_ticks = 0
            else:
                bad_ticks += 1
                flog_print(f"[GPS WARN] Implausible tick {tick_dist:.2f}m ignored "
                           f"({bad_ticks}/{GPS_REBASE_AFTER_N_BAD_TICKS}).")
                if bad_ticks >= GPS_REBASE_AFTER_N_BAD_TICKS:
                    # r2: persistent jump (GPS stall / re-acquire). Re-base the
                    # reference so later ticks are measured again. The jump itself
                    # is NOT counted as distance.
                    flog_print(f"[GPS WARN] Re-basing GPS reference after {bad_ticks} "
                               f"consecutive bad ticks (jump of {tick_dist:.2f}m not counted).")
                    last_lat, last_lon = cur_lat, cur_lon
                    bad_ticks    = 0
                    gps_rebases += 1

        # --- DETECTION: check queue, respect cooldown ---
        now = time.monotonic()
        if now >= cooldown_until:
            # r2: newest detection only, and only if fresh (<= DETECTION_MAX_AGE_S old).
            det = _get_fresh_detection()
            if det is not None:
                hover_status = await run_detection_hover(
                    drone, det, yaw_deg, distance_traveled, run_detections)

                if hover_status == "aborted":
                    log.append({
                        "step": label, "status": "aborted", "t": time.time(),
                        "distance_traveled_m": distance_traveled,
                        "abort_during": "detection_hover",
                    })
                    return "aborted"

                hover_count    += 1
                # r2: discard everything the vision thread queued while hovering.
                _drain_detection_queue()
                cooldown_until  = time.monotonic() + DETECTION_COOLDOWN_S

                new_lat, new_lon = await _get_current_position()
                if new_lat is not None:
                    last_lat, last_lon = new_lat, new_lon
                bad_ticks = 0

                continue
        else:
            # r2: keep the queue empty during cooldown so a backlog cannot build up.
            _drain_detection_queue()

        # --- Progress print ~every 1s ---
        if time.monotonic() - last_print_t >= 1.0:
            flog_print(f"[SEQUENCE] {label}: {distance_traveled:.2f}m / "
                       f"{target_distance_m:.2f}m  (hovers: {hover_count})")
            last_print_t = time.monotonic()

        # --- Velocity command ---
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(north_vel, east_vel, 0.0, yaw_deg))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)

    end_lat, end_lon = await _get_current_position()
    straight = haversine_distance(start_lat, start_lon,
                                  end_lat or last_lat, end_lon or last_lon)
    flog_print(f"[SEQUENCE] {label} complete. "
               f"Accumulated={distance_traveled:.2f}m  "
               f"straight-line={straight:.2f}m  "
               f"detection hovers={hover_count}")
    log.append({
        "step": label, "status": "completed", "t": time.time(),
        "distance_traveled_m": distance_traveled,
        "straight_line_m": straight,
        "detection_hovers": hover_count,
        "gps_rebases": gps_rebases,
        "start_lat": start_lat, "start_lon": start_lon,
        "end_lat": end_lat, "end_lon": end_lon,
    })
    return "completed"


async def run_hover_pause(drone: System, label: str, duration: float,
                          yaw_deg: float, run_detections: list, log: list) -> str:
    flog_print(f"\n[SEQUENCE] {label} -- {duration}s pause between legs")
    start = time.monotonic()
    while (time.monotonic() - start) < duration:
        if mode_state.current_mode != TARGET_TRIGGER_MODE:
            flog_print(f"[SAFETY] Mode changed during '{label}'. Aborting.")
            log.append({"step": label, "status": "aborted", "t": time.time()})
            return "aborted"
        try:
            det = detection_queue.get_nowait()
            flog_print(f"[VISION] Detection during pause (logged, not actioned): "
                       f"conf={det['confidence']:.2f}")
            run_detections.append({**det, "leg": label, "actioned": False})
        except queue.Empty:
            pass
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw_deg))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)
    log.append({"step": label, "status": "completed", "t": time.time()})
    return "completed"


# ============================================================
# ---- MAIN SEQUENCE ------------------------------------------
# ============================================================

async def run_takeover_sequence(drone: System, run_index: int) -> dict:
    global _flog

    # --- Open per-flight log file ---
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt  = os.path.join(OUTPUT_DIR, f"flight_{ts_str}_run{run_index:02d}.txt")
    _flog    = FlightLogger(log_txt)

    run_start_t = time.time()
    volt_samples: list[float] = []

    record = {
        "timestamp":     run_start_t,
        "run_index":     run_index,
        "is_simulation": IS_SIMULATION,
        "test_type":     "v6_gps_vision_active",
        "script_revision": "r2",
        "model":         MODEL_NAME,
        "flight_log":    log_txt,
        "config": {
            "forward_distance_m":  FORWARD_DISTANCE_M,
            "backward_distance_m": BACKWARD_DISTANCE_M,
            "axis_speed_mps":      AXIS_SPEED_MPS,
            "axis_speed_kmh":      round(AXIS_SPEED_MPS * 3.6, 2),
            "detection_hover_s":   DETECTION_HOVER_S,
            "detection_cooldown_s":DETECTION_COOLDOWN_S,
            "confidence_threshold":CONFIDENCE_THRESHOLD,
            "detection_max_age_s": DETECTION_MAX_AGE_S,
            "align_legs_to_heading": ALIGN_LEGS_TO_HEADING,
        },
        "pre_engage_status": {
            "battery_voltage_v":     status_state.battery_voltage_v,
            "battery_remaining_pct": (
                round(status_state.battery_remaining_pct * 100, 1)
                if status_state.battery_remaining_pct is not None else None
            ),
            "gps_fix_type":          status_state.gps_fix_type,
            "gps_num_satellites":    status_state.gps_num_satellites,
            "is_global_position_ok": status_state.is_global_position_ok,
            "is_local_position_ok":  status_state.is_local_position_ok,
        },
        "aborted":       False,
        "abort_step":    None,
        "detections":    [],
        "steps":         [],
        # Filled at the end:
        "post_engage_status":  None,
        "flight_duration_s":   None,
        "total_distance_m":    None,
        "battery_volt_min":    None,
        "battery_volt_max":    None,
        "gps_satellites_min":  None,
    }
    log        = record["steps"]
    detections = record["detections"]

    if not status_state.is_global_position_ok or not status_state.is_local_position_ok:
        flog_print("[SAFETY WARNING] Position health not OK. "
                   "GPS distance control may be unreliable.")

    # Drain stale detections accumulated during standby.
    drained = 0
    while not detection_queue.empty():
        try:
            detection_queue.get_nowait()
            drained += 1
        except queue.Empty:
            break
    if drained:
        flog_print(f"[CONTROL] Drained {drained} stale detection(s) from standby queue.")

    started = await start_offboard_with_retry(drone, timeout=3.0)
    if not started:
        flog_print("[CONTROL ERROR] Could not start offboard mode.")
        record["aborted"] = True; record["abort_step"] = "offboard_start"
        _flog.close(); _flog = None
        return record

    confirm_deadline = time.monotonic() + 2.0
    while time.monotonic() < confirm_deadline:
        if mode_state.current_mode == TARGET_TRIGGER_MODE:
            break
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await asyncio.sleep(0.05)
    else:
        flog_print("[CONTROL ERROR] GUIDED mode not confirmed within grace period.")
        record["aborted"] = True; record["abort_step"] = "offboard_confirm"
        await _stop_offboard(drone)
        _flog.close(); _flog = None
        return record

    flog_print("[CONTROL] Offboard active. Detections will trigger hover.")
    asyncio.create_task(play_audio_feedback(drone, TUNE_AUTO_ENGAGED))

    async with telemetry_cache.lock:
        base_yaw = telemetry_cache.data.get("yaw_deg", 0.0)

    # r2: unit vector along the nose heading (yaw 0 = north, +90 = east).
    if ALIGN_LEGS_TO_HEADING:
        fwd_n = math.cos(math.radians(base_yaw))
        fwd_e = math.sin(math.radians(base_yaw))
    else:
        fwd_n, fwd_e = 1.0, 0.0
    flog_print(f"[CONTROL] Leg heading: yaw={base_yaw:.1f}° -> unit vector "
               f"N={fwd_n:+.3f} E={fwd_e:+.3f} (align_to_heading={ALIGN_LEGS_TO_HEADING})")

    # Forward leg
    status = await run_gps_leg(
        drone, "forward",
        north_vel=AXIS_SPEED_MPS * fwd_n, east_vel=AXIS_SPEED_MPS * fwd_e,
        yaw_deg=base_yaw,
        target_distance_m=FORWARD_DISTANCE_M,
        run_detections=detections, log=log,
        volt_samples=volt_samples,
    )
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "forward"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples)
        _flog.close(); _flog = None
        return record

    # Pause between legs
    status = await run_hover_pause(
        drone, "pause", PAUSE_BETWEEN_S, base_yaw, detections, log)
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "pause"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples)
        _flog.close(); _flog = None
        return record

    # Backward leg
    status = await run_gps_leg(
        drone, "backward",
        north_vel=-AXIS_SPEED_MPS * fwd_n, east_vel=-AXIS_SPEED_MPS * fwd_e,
        yaw_deg=base_yaw,
        target_distance_m=BACKWARD_DISTANCE_M,
        run_detections=detections, log=log,
        volt_samples=volt_samples,
    )
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "backward"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples)
        _flog.close(); _flog = None
        return record

    # Hold until pilot flips SWA back
    total_det = len(detections)
    flog_print(f"\n[CONTROL] Sequence complete. {total_det} detection hovers this run.")
    flog_print("[CONTROL] Holding in GUIDED until manual override (flip SWA back)...")
    while mode_state.current_mode == TARGET_TRIGGER_MODE:
        try:
            det = detection_queue.get_nowait()
            flog_print(f"[VISION] Detection during hold (logged only): "
                       f"conf={det['confidence']:.2f}")
            detections.append({**det, "leg": "hold", "actioned": False})
        except queue.Empty:
            pass
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, base_yaw))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)

    flog_print("[CONTROL] Manual override detected. Handing back to pilot.")
    await _stop_offboard(drone)
    asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))

    _finalise_record(record, run_start_t, log, volt_samples)
    _flog.close(); _flog = None
    return record


def _finalise_record(record: dict, run_start_t: float,
                     log: list, volt_samples: list):
    """Fill in the summary fields at the end of every run (aborted or complete)."""
    now = time.time()
    record["flight_duration_s"] = round(now - run_start_t, 1)
    record["post_engage_status"] = {
        "battery_voltage_v":     status_state.battery_voltage_v,
        "battery_remaining_pct": (
            round(status_state.battery_remaining_pct * 100, 1)
            if status_state.battery_remaining_pct is not None else None
        ),
        "gps_fix_type":          status_state.gps_fix_type,
        "gps_num_satellites":    status_state.gps_num_satellites,
        "is_global_position_ok": status_state.is_global_position_ok,
        "is_local_position_ok":  status_state.is_local_position_ok,
    }
    if volt_samples:
        record["battery_volt_min"] = round(min(volt_samples), 3)
        record["battery_volt_max"] = round(max(volt_samples), 3)

    # Total GPS distance from completed legs.
    total_dist = sum(
        s.get("distance_traveled_m", 0.0)
        for s in log if s.get("status") in ("completed", "aborted")
    )
    record["total_distance_m"] = round(total_dist, 2)

    # Min satellite count seen during the run (proxy for GPS quality).
    if status_state.gps_num_satellites is not None:
        record["gps_satellites_min"] = status_state.gps_num_satellites


# ============================================================
# ---- MAIN ---------------------------------------------------
# ============================================================

async def run():
    global STATUS_LOG_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Per-session status log (timestamped) so multiple sessions don't append to one file.
    session_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATUS_LOG_PATH = os.path.join(OUTPUT_DIR, f"status_log_{session_ts}.jsonl")

    print(f"[CONTROL] Mode: {'SIMULATION' if IS_SIMULATION else 'HARDWARE'}")
    print(f"[CONTROL] Connecting to {CONNECTION_ADDRESS}...")
    print(f"[VISION]  Starting inference thread ({MODEL_NAME})...")

    vision_thread = threading.Thread(target=vision_worker, daemon=True)
    vision_thread.start()

    drone = System()
    await drone.connect(system_address=CONNECTION_ADDRESS)

    try:
        await asyncio.wait_for(wait_for_connection(drone), timeout=CONNECTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        print("[CONTROL ERROR] No connection within timeout.")
        vision_stop_event.set()
        return

    background_tasks = [
        asyncio.create_task(telemetry_cache_task(drone)),
        asyncio.create_task(mode_watchdog_task(drone)),
        asyncio.create_task(armed_watchdog_task(drone)),
        asyncio.create_task(battery_watchdog_task(drone)),
        asyncio.create_task(gps_watchdog_task(drone)),
        asyncio.create_task(health_watchdog_task(drone)),
        asyncio.create_task(status_log_task()),
    ]

    if IS_SIMULATION:
        print("[CONTROL] SIMULATION MODE -- arming and taking off...")
        await drone.action.arm()
        await drone.action.takeoff()
        await asyncio.sleep(5)
    else:
        print("\n" + "=" * 70)
        print(">>> HARDWARE MODE -- V6 GPS CONTROL + ACTIVE VISION.")
        print(f">>> {FORWARD_DISTANCE_M}m forward, pause, {BACKWARD_DISTANCE_M}m back "
              f"at {AXIS_SPEED_MPS} m/s ({AXIS_SPEED_MPS * 3.6:.1f} km/h).")
        print(f">>> Model: {MODEL_NAME}  |  Hover on detection: {DETECTION_HOVER_S}s")
        print(f">>> Cooldown between hovers: {DETECTION_COOLDOWN_S}s")
        print(">>> WAIT for [HEALTH] global_pos_ok=True before triggering.")
        print(">>> Flip SWA (Ch6) to trigger GUIDED. Flip back at ANY time to abort.")
        print(f">>> Logs: {OUTPUT_DIR}")
        print("=" * 70 + "\n")

    run_index = 1
    try:
        # r2: do not enter standby until the vision pipeline is really running.
        print("[VISION] Waiting for model + camera to come up "
              "(model load can take 10-30 s)...")
        t_wait = time.monotonic()
        while not vision_ready.is_set():
            if vision_failed.is_set() or not vision_thread.is_alive():
                print("[VISION ERROR] Vision pipeline failed to start (see error above). "
                      "NOT entering standby -- fix and restart the script.")
                return
            if time.monotonic() - t_wait > VISION_READY_TIMEOUT_S:
                print(f"[VISION ERROR] Vision not ready after {VISION_READY_TIMEOUT_S}s. "
                      f"NOT entering standby.")
                return
            await asyncio.sleep(0.2)
        print(">>> VISION READY -- trigger only after [HEALTH] global_pos_ok=True.\n")

        while True:
            print("[CONTROL] Standing by (armed + GPS healthy + SWA to GUIDED)...")
            while True:
                if mode_state.is_armed and mode_state.current_mode == TARGET_TRIGGER_MODE:
                    if await wait_for_sustained_mode(TARGET_TRIGGER_MODE, TRIGGER_DEBOUNCE_S):
                        break
                await asyncio.sleep(0.1)

            if not vision_thread.is_alive():
                print("[VISION ERROR] Vision thread is not running -- refusing to start "
                      "the sequence. Flip SWA back and restart the script.")
                while mode_state.current_mode == TARGET_TRIGGER_MODE:
                    await asyncio.sleep(0.1)
                continue

            record = await run_takeover_sequence(drone, run_index)
            run_index += 1

            ts   = int(record["timestamp"])
            path = os.path.join(OUTPUT_DIR, f"run_{ts}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=4)

            n_det = len(record.get("detections", []))
            dur   = record.get("flight_duration_s", "?")
            dist  = record.get("total_distance_m", "?")
            print(f"[DATA] Run log saved : {path}")
            print(f"[DATA] Flight log    : {record.get('flight_log', '?')}")
            print(f"[DATA] Duration      : {dur}s  |  Distance: {dist}m  |  Detections: {n_det}")
            print(f"[DATA] Frames        : {FRAMES_DIR}")
            print(f"[DATA] Detection log : {os.path.join(OUTPUT_DIR, 'detections_log.jsonl')}")

            print("[CONTROL] 3-second cooldown...")
            await asyncio.sleep(3.0)
            print("[CONTROL] Cooldown complete. Returning to standby.\n")

    except KeyboardInterrupt:
        print("\n[CONTROL] Terminated by user.")
    finally:
        vision_stop_event.set()
        for t in background_tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        vision_thread.join(timeout=2.0)

if __name__ == "__main__":
    asyncio.run(run())
