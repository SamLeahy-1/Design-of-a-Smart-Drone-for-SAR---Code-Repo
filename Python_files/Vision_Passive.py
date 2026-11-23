#!/usr/bin/env python3
"""
SAR Drone - V7 GPS Control + Passive Vision (Logging Only)
==============================================================================
Purpose: Autonomous pipeline validation + video collection for offline
         model comparison.

This script is the passive counterpart to V6_Vision_Active.  It performs
identical GPS-controlled flight and real-time AI inference, but does NOT
alter the flight path on detection.  Every detection is logged and saved,
but the drone continues its search pattern without interruption.

REVISION r2 (field-readiness fixes -- V7 logic otherwise unchanged):
  1. Model is loaded ONCE at script start (before connecting to the FC), warmed
     up, and kept resident. Previously it was loaded inside the vision thread
     after the SWA flip, which delayed/starved the control loop and video.
  2. Each run waits for "vision ready" (camera streaming + video file opened)
     before moving. If the camera or recorder fails, the sequence is NOT flown;
     the script holds until SWA is flipped back.
  3. Legs fly along the nose heading captured at engage (ALIGN_LEGS_TO_HEADING)
     instead of geographic north. Set it False to restore north-only legs.
  4. GPS glitch guard recovers: after GPS_REBASE_AFTER_N_BAD_TICKS consecutive
     rejected ticks the reference is re-based (jump NOT counted as distance).
  5. Camera opened with V4L2 + MJPG 1280x720@30 + buffer size 1; negotiated
     mode printed, warning if it differs. Probed once at start-up.
  6. Per-frame sidecar CSV next to each video (frame_idx, t_unix, altitude,
     lat, lon, best_score) so frames can be aligned to flight data offline,
     and video_stats (frames, real average fps, resolution) in the run JSON.
     detection "frame_num" is now the frame index within the video.
  7. mmdet boxes scaled from 640x640 back to camera-frame pixels.
  MODEL_NAME / MODEL_CONFIGS / distances are untouched -- keep your Jetson values.

Key additions over V4_Vision_Passive:
  - Per-flight terminal log (.txt) -- mirrors exactly what prints to SSH.
  - V6-style enriched run JSON:
      flight_duration_s, total_distance_m, battery volt min/max,
      gps_satellites_min, altitude_m (min/max/at-engage), config snapshot,
      pre- and post-engage status blocks.
  - Battery % display fixed: MAVSDK remaining_percent is 0.0-1.0.
  - Per-session status log (timestamped filename, not a shared append).
  - Altitude telemetry: relative_altitude_m logged in every step record
    and summarised in the run JSON.
  - VideoWriter: onboard camera footage saved to OUTPUT_DIR for offline
    model comparison.  Writer opens on first frame so resolution is
    auto-detected from the live camera.

FLIGHT SEQUENCE:
  Forward  : GPS-confirmed FORWARD_DISTANCE_M
  Pause    : PAUSE_BETWEEN_S hover (detections still logged)
  Backward : GPS-confirmed BACKWARD_DISTANCE_M
  Hold     : hover in GUIDED until pilot flips SWA back

SAFETY (identical to V4/V6):
  - Mode-abort check every ~50ms tick, unconditionally highest priority.
  - Vision thread never touches the abort path.
  - Pre-engage EKF health warning if position estimate not OK.

MODEL SELECTION: set MODEL_NAME to any key in MODEL_CONFIGS.

ENVIRONMENT:
  MMDet models  : source ~/mmdet_env/bin/activate   (+ pip install mavsdk)
  YOLO models   : source ~/drone_env/bin/activate    (+ pip install mavsdk)
"""

import asyncio
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
FORWARD_DISTANCE_M     = 40.0
BACKWARD_DISTANCE_M    = 40.0
AXIS_SPEED_MPS         = 2   # ~5 km/h -- suitable for open field
PAUSE_BETWEEN_S        = 2.0
POSITION_LOOP_PERIOD_S = 0.05
MAX_PLAUSIBLE_TICK_M   = 2.0

# --- r2 additions -------------------------------------------------------
ALIGN_LEGS_TO_HEADING        = True   # fly along nose heading at engage (False = old: geographic north)
GPS_REBASE_AFTER_N_BAD_TICKS = 3      # consecutive rejected GPS ticks before the reference is re-based

# --- Status logging ---
STATUS_LOG_PERIOD_S = 1.0

# --- Vision ---
MODEL_NAME           = "YOLOv11n_VisDrone"
CAMERA_SOURCE        = 0
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID      = 0
INFERENCE_TARGET_HZ  = 10
SAMPLE_FRAME_EVERY_N = 30   # save a sample frame every N frames regardless of detection

# --- r2: camera mode (J.Burrows webcam needs MJPG for 720p30 over USB 2.0) ---
CAMERA_USE_MJPG        = True
CAMERA_WIDTH           = 1280
CAMERA_HEIGHT          = 720
CAMERA_FPS             = 30
VISION_READY_TIMEOUT_S = 15    # per-run wait for camera + video file (model is pre-loaded)

# --- Video recording ---
SAVE_VIDEO       = True          # set False to disable recording
VIDEO_CODEC      = "XVID"        # XVID/.avi is more reliable on Jetson than mp4v
VIDEO_FPS        = 10.0          # should match INFERENCE_TARGET_HZ for 1:1 playback
# Resolution is auto-detected from the first frame; override here if needed.
VIDEO_RESOLUTION = None          # e.g. (1280, 720) -- None = auto

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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v7_vision_passive_logs")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_DIR  = os.path.join(OUTPUT_DIR, "video")

# Set at runtime in run() -- unique per session.
STATUS_LOG_PATH = None

TUNE_AUTO_ENGAGED  = TuneDescription([SongElement.NOTE_C, SongElement.NOTE_E, SongElement.NOTE_G], 200)
TUNE_MANUAL_RETURN = TuneDescription([SongElement.NOTE_G, SongElement.NOTE_E, SongElement.NOTE_C], 200)

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
        self.altitude_m            = None   # relative altitude AGL (m)

status_state = StatusState()

detection_queue   = queue.Queue()
vision_stop_event = threading.Event()
vision_ready      = threading.Event()   # r2: camera streaming + video file open
vision_failed     = threading.Event()   # r2: camera/recorder failed to start
video_stats       = {}                  # r2: filled by the vision thread at the end of each run


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
            f"SAR Drone V7 r2 -- Flight Log (Passive)\n"
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
        msg  = " ".join(str(a) for a in args)
        ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
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
            self._fh.write(
                f"\n{'=' * 70}\n"
                f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
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

def vision_worker(video_path: str | None, inferencer):
    """
    Captures frames, runs inference, logs detections, saves sample frames.
    If video_path is not None, also records the raw camera feed to that file.

    The VideoWriter is opened on the first successfully captured frame so
    resolution is auto-detected from the live camera.  If VIDEO_RESOLUTION
    is set in config, that takes precedence.
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    det_log_path = os.path.join(OUTPUT_DIR, "detections_log.jsonl")

    # r2: the model is pre-loaded in run() and passed in; only the camera opens here.
    try:
        source = FrameSource(CAMERA_SOURCE)
    except Exception as e:
        print(f"[VISION ERROR] Camera: {e!r}")
        vision_failed.set()
        return

    min_period   = 1.0 / INFERENCE_TARGET_HZ
    frame_count  = 0      # resets every 5 s (fps print / sample cadence)
    total_frames = 0      # r2: frame index within this run's video
    detect_count = 0
    last_fps_t   = time.monotonic()
    t_start      = time.monotonic()
    writer       = None   # cv2.VideoWriter, opened on first frame
    csv_fh       = None   # r2: per-frame sidecar CSV
    vid_wh       = None

    print(f"[VISION] Worker running. {MODEL_NAME} @ up to {INFERENCE_TARGET_HZ}Hz.")
    if video_path:
        print(f"[VISION] Video will be saved to: {video_path}")

    try:
        while not vision_stop_event.is_set():
            t0 = time.monotonic()

            ok, frame = source.read()
            if not ok:
                print("[VISION WARN] Frame grab failed.")
                time.sleep(0.1)
                continue

            frame_count  += 1
            total_frames += 1
            frame_t = time.time()

            # --- Open VideoWriter on first frame (auto-detect resolution) ---
            if video_path and writer is None:
                if VIDEO_RESOLUTION is not None:
                    w, h = VIDEO_RESOLUTION
                else:
                    h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
                if not writer.isOpened():
                    # r2: video is the point of V7 -- do not fly a run that can't record.
                    print(f"[VISION ERROR] VideoWriter failed to open: {video_path}.")
                    vision_failed.set()
                    return
                print(f"[VISION] Recording started: {w}x{h} @ {VIDEO_FPS}fps")
                vid_wh = (w, h)
                csv_path = os.path.splitext(video_path)[0] + "_frames.csv"
                try:
                    csv_fh = open(csv_path, "w", encoding="utf-8")
                    csv_fh.write("frame_idx,t_unix,altitude_m,lat_deg,lon_deg,best_score\n")
                except Exception as e:
                    print(f"[VISION WARN] Frame CSV not written: {e}")
                    csv_fh = None

            # --- Write raw frame to video ---
            if writer is not None:
                writer.write(frame)

            if not vision_ready.is_set():
                vision_ready.set()
                print("[VISION] READY -- camera streaming, video file open.")

            try:
                best_score, best_box = inferencer.infer(frame)
            except Exception as e:
                print(f"[VISION ERROR] Inference: {e}")
                best_score, best_box = 0.0, None

            # r2: per-frame sidecar row (flushed every frame; ~10 rows/s).
            if csv_fh is not None:
                try:
                    csv_fh.write(f"{total_frames},{frame_t:.3f},{status_state.altitude_m},"
                                 f"{telemetry_cache.data.get('lat_deg')},"
                                 f"{telemetry_cache.data.get('lon_deg')},{best_score:.4f}\n")
                    csv_fh.flush()
                except Exception:
                    csv_fh = None

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
                det_path = os.path.join(
                    FRAMES_DIR, f"detection_{ts_ms}_{best_score:.2f}.jpg")
                cv2.imwrite(det_path, annotated)

                record = {
                    "timestamp":  now,
                    "model":      MODEL_NAME,
                    "confidence": round(best_score, 4),
                    "bbox":       best_box,
                    "frame_path": det_path,
                    "frame_num":  total_frames,
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
        dur_s = time.monotonic() - t_start
        if writer is not None:
            writer.release()
            print("[VISION] VideoWriter closed.")
        if csv_fh is not None:
            try:
                csv_fh.close()
            except Exception:
                pass
        avg_fps = (total_frames / dur_s) if dur_s > 0 else 0.0
        video_stats.clear()
        video_stats.update({
            "frames":            total_frames,
            "duration_s":        round(dur_s, 2),
            "avg_capture_fps":   round(avg_fps, 2),
            "video_header_fps":  VIDEO_FPS,
            "frame_size":        list(vid_wh) if vid_wh else None,
            "camera_negotiated": list(source.negotiated) if source.negotiated else None,
        })
        print(f"[VISION] Run video: {total_frames} frames in {dur_s:.1f}s "
              f"= {avg_fps:.1f} fps average (video header says {VIDEO_FPS}). "
              f"Use the _frames.csv timestamps if these differ.")
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
            # altitude updated in shared state directly (no lock needed for float write)
            status_state.altitude_m = pos.relative_altitude_m
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
            "altitude_m":            status_state.altitude_m,
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


async def _wait_for_vision_ready(timeout_s: float) -> str:
    """r2: returns 'ready', 'failed', 'timeout' or 'aborted' (SWA flipped back)."""
    t0 = time.monotonic()
    while not vision_ready.is_set():
        if mode_state.current_mode != TARGET_TRIGGER_MODE:
            return "aborted"
        if vision_failed.is_set():
            return "failed"
        if time.monotonic() - t0 > timeout_s:
            return "timeout"
        await asyncio.sleep(0.05)
    return "ready"


# ============================================================
# ---- GPS LEG (PASSIVE -- no detection hover) ----------------
# ============================================================

async def run_gps_leg(drone: System, label: str, north_vel: float, east_vel: float,
                      yaw_deg: float, target_distance_m: float,
                      run_detections: list, log: list,
                      volt_samples: list, alt_samples: list) -> str:
    start_lat, start_lon = await _get_current_position()
    if start_lat is None or start_lon is None:
        flog_print(f"[CONTROL ERROR] No GPS at start of '{label}'. Aborting.")
        log.append({"step": label, "status": "aborted_no_gps", "t": time.time()})
        return "aborted"

    flog_print(f"\n[SEQUENCE] {label} -- target={target_distance_m}m "
               f"vN={north_vel:+.2f} vE={east_vel:+.2f} m/s  yaw={yaw_deg:.1f}°")
    flog_print(f"[SEQUENCE] Start GPS: lat={start_lat:.7f} lon={start_lon:.7f}  "
               f"alt={status_state.altitude_m:.1f}m" if status_state.altitude_m is not None
               else f"[SEQUENCE] Start GPS: lat={start_lat:.7f} lon={start_lon:.7f}")

    distance_traveled = 0.0
    last_lat, last_lon = start_lat, start_lon
    last_print_t       = time.monotonic()
    bad_ticks          = 0     # r2: consecutive rejected GPS ticks
    gps_rebases        = 0     # r2: how many times the GPS reference was re-based

    while distance_traveled < target_distance_m:

        # --- ABORT: unconditional, always first ---
        if mode_state.current_mode != TARGET_TRIGGER_MODE:
            flog_print(f"[SAFETY] Mode changed during '{label}'. "
                       f"Traveled {distance_traveled:.2f}m. Aborting.")
            log.append({
                "step":                label,
                "status":              "aborted",
                "t":                   time.time(),
                "distance_traveled_m": distance_traveled,
                "altitude_m":          status_state.altitude_m,
                "battery_voltage_v":   status_state.battery_voltage_v,
                "gps_fix_type":        status_state.gps_fix_type,
                "is_global_position_ok": status_state.is_global_position_ok,
                "gps_rebases":         gps_rebases,
            })
            return "aborted"

        # --- Battery and altitude samples (for min/max summary) ---
        if status_state.battery_voltage_v is not None:
            volt_samples.append(status_state.battery_voltage_v)
        if status_state.altitude_m is not None:
            alt_samples.append(status_state.altitude_m)

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

        # --- Log detection events (passive -- no flight path change) ---
        try:
            det = detection_queue.get_nowait()
            flog_print(f"[VISION] DETECTION: conf={det['confidence']:.2f}  "
                       f"bbox={det['bbox']}  at {distance_traveled:.2f}m into leg  "
                       f"alt={status_state.altitude_m:.1f}m" if status_state.altitude_m is not None
                       else f"[VISION] DETECTION: conf={det['confidence']:.2f}  "
                            f"bbox={det['bbox']}  at {distance_traveled:.2f}m into leg")
            run_detections.append({
                **det,
                "leg":                  label,
                "distance_at_detect_m": distance_traveled,
                "altitude_at_detect_m": status_state.altitude_m,
            })
        except queue.Empty:
            pass

        # --- Progress print ~every 1s ---
        if time.monotonic() - last_print_t >= 1.0:
            alt_str = (f"  alt={status_state.altitude_m:.1f}m"
                       if status_state.altitude_m is not None else "")
            flog_print(f"[SEQUENCE] {label}: {distance_traveled:.2f}m / "
                       f"{target_distance_m:.2f}m{alt_str}")
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
               f"straight-line={straight:.2f}m")
    log.append({
        "step":                label,
        "status":              "completed",
        "t":                   time.time(),
        "distance_traveled_m": distance_traveled,
        "straight_line_m":     straight,
        "gps_rebases":         gps_rebases,
        "altitude_m":          status_state.altitude_m,
        "start_lat":           start_lat,
        "start_lon":           start_lon,
        "end_lat":             end_lat,
        "end_lon":             end_lon,
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
            flog_print(f"[VISION] Detection during pause (logged): "
                       f"conf={det['confidence']:.2f}")
            run_detections.append({**det, "leg": label})
        except queue.Empty:
            pass
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw_deg))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)
    log.append({"step": label, "status": "completed", "t": time.time()})
    return "completed"


# ============================================================
# ---- RECORD HELPERS -----------------------------------------
# ============================================================

def _finalise_record(record: dict, run_start_t: float,
                     log: list, volt_samples: list, alt_samples: list):
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
        "altitude_m":            status_state.altitude_m,
    }
    if volt_samples:
        record["battery_volt_min"] = round(min(volt_samples), 3)
        record["battery_volt_max"] = round(max(volt_samples), 3)
    if alt_samples:
        record["altitude_m_min"] = round(min(alt_samples), 2)
        record["altitude_m_max"] = round(max(alt_samples), 2)
        record["altitude_m_mean"] = round(sum(alt_samples) / len(alt_samples), 2)

    # Total GPS distance from completed/aborted legs.
    total_dist = sum(
        s.get("distance_traveled_m", 0.0)
        for s in log if s.get("status") in ("completed", "aborted")
    )
    record["total_distance_m"] = round(total_dist, 2)

    # Min satellite count seen during the run (proxy for GPS quality).
    if status_state.gps_num_satellites is not None:
        record["gps_satellites_min"] = status_state.gps_num_satellites


# ============================================================
# ---- MAIN SEQUENCE ------------------------------------------
# ============================================================

async def run_takeover_sequence(drone: System, run_index: int,
                                video_path: str | None) -> dict:
    global _flog

    # --- Open per-flight log file ---
    ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt = os.path.join(OUTPUT_DIR, f"flight_{ts_str}_run{run_index:02d}.txt")
    _flog   = FlightLogger(log_txt)

    run_start_t = time.time()
    volt_samples: list[float] = []
    alt_samples:  list[float] = []

    record = {
        "timestamp":     run_start_t,
        "run_index":     run_index,
        "is_simulation": IS_SIMULATION,
        "test_type":     "v7_gps_vision_passive",
        "script_revision": "r2",
        "model":         MODEL_NAME,
        "flight_log":    log_txt,
        "video_path":    video_path,
        "config": {
            "forward_distance_m":   FORWARD_DISTANCE_M,
            "backward_distance_m":  BACKWARD_DISTANCE_M,
            "axis_speed_mps":       AXIS_SPEED_MPS,
            "axis_speed_kmh":       round(AXIS_SPEED_MPS * 3.6, 2),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "inference_target_hz":  INFERENCE_TARGET_HZ,
            "save_video":           SAVE_VIDEO,
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
            "altitude_m":            status_state.altitude_m,
        },
        "aborted":       False,
        "abort_step":    None,
        "detections":    [],
        "steps":         [],
        # Filled by _finalise_record():
        "post_engage_status":  None,
        "flight_duration_s":   None,
        "total_distance_m":    None,
        "battery_volt_min":    None,
        "battery_volt_max":    None,
        "altitude_m_min":      None,
        "altitude_m_max":      None,
        "altitude_m_mean":     None,
        "gps_satellites_min":  None,
    }
    log        = record["steps"]
    detections = record["detections"]

    if not status_state.is_global_position_ok or not status_state.is_local_position_ok:
        flog_print("[SAFETY WARNING] Position health not OK. "
                   "GPS distance control may be unreliable.")

    # r2: do not start moving until camera + video file are confirmed streaming.
    vstate = await _wait_for_vision_ready(VISION_READY_TIMEOUT_S)
    if vstate != "ready":
        flog_print(f"[VISION ERROR] Vision not ready ({vstate}). Sequence NOT started; "
                   f"holding until SWA is flipped back.")
        record["aborted"] = True; record["abort_step"] = f"vision_{vstate}"
        while mode_state.current_mode == TARGET_TRIGGER_MODE:
            await asyncio.sleep(0.1)
        _finalise_record(record, run_start_t, log, volt_samples, alt_samples)
        _flog.close(); _flog = None
        return record

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

    flog_print("[CONTROL] Offboard active. Passive inference running (no flight path changes).")
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
        volt_samples=volt_samples, alt_samples=alt_samples,
    )
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "forward"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples, alt_samples)
        _flog.close(); _flog = None
        return record

    # Pause between legs
    status = await run_hover_pause(
        drone, "pause", PAUSE_BETWEEN_S, base_yaw, detections, log)
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "pause"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples, alt_samples)
        _flog.close(); _flog = None
        return record

    # Backward leg
    status = await run_gps_leg(
        drone, "backward",
        north_vel=-AXIS_SPEED_MPS * fwd_n, east_vel=-AXIS_SPEED_MPS * fwd_e,
        yaw_deg=base_yaw,
        target_distance_m=BACKWARD_DISTANCE_M,
        run_detections=detections, log=log,
        volt_samples=volt_samples, alt_samples=alt_samples,
    )
    if status == "aborted":
        record["aborted"] = True; record["abort_step"] = "backward"
        await _stop_offboard(drone)
        asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))
        _finalise_record(record, run_start_t, log, volt_samples, alt_samples)
        _flog.close(); _flog = None
        return record

    # Hold until pilot flips SWA back
    total_det = len(detections)
    flog_print(f"\n[CONTROL] Sequence complete. {total_det} detections logged this run.")
    flog_print("[CONTROL] Holding in GUIDED until manual override (flip SWA back)...")
    while mode_state.current_mode == TARGET_TRIGGER_MODE:
        try:
            det = detection_queue.get_nowait()
            flog_print(f"[VISION] Detection during hold (logged): "
                       f"conf={det['confidence']:.2f}")
            detections.append({**det, "leg": "hold"})
        except queue.Empty:
            pass
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, base_yaw))
        await asyncio.sleep(POSITION_LOOP_PERIOD_S)

    flog_print("[CONTROL] Manual override detected. Handing back to pilot.")
    await _stop_offboard(drone)
    asyncio.create_task(play_audio_feedback(drone, TUNE_MANUAL_RETURN))

    _finalise_record(record, run_start_t, log, volt_samples, alt_samples)
    _flog.close(); _flog = None
    return record


# ============================================================
# ---- MAIN ---------------------------------------------------
# ============================================================

async def run():
    global STATUS_LOG_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR,  exist_ok=True)

    # Per-session status log (timestamped) so multiple sessions don't append to one file.
    session_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATUS_LOG_PATH = os.path.join(OUTPUT_DIR, f"status_log_{session_ts}.jsonl")

    print(f"[CONTROL] Mode: {'SIMULATION' if IS_SIMULATION else 'HARDWARE'}")
    print(f"[CONTROL] Connecting to {CONNECTION_ADDRESS}...")
    print(f"[VISION]  Starting inference thread ({MODEL_NAME})...")
    print(f"[VISION]  Video recording: {'ENABLED' if SAVE_VIDEO else 'DISABLED'}")

    # r2: load the model ONCE here, before connecting to the FC and long before any
    # flight, and keep it resident. A fresh camera/video thread is still started per
    # trigger cycle (below) so each run gets its own clean video file.
    print(f"[VISION]  Pre-loading {MODEL_NAME} (10-30 s on the Jetson)...")
    try:
        inferencer = ModelInferencer(MODEL_NAME)
        inferencer.warmup()
    except Exception as e:
        print(f"[VISION ERROR] Could not load {MODEL_NAME}: {e!r}")
        print("[VISION ERROR] Check MODEL_NAME / MODEL_CONFIGS paths (Linux is "
              "case-sensitive). NOT starting.")
        return
    try:
        probe = FrameSource(CAMERA_SOURCE)
        ok, _ = probe.read()
        probe.release()
        if not ok:
            raise RuntimeError("camera opened but returned no frame")
    except Exception as e:
        print(f"[VISION ERROR] Camera check failed: {e!r}. NOT starting.")
        return
    print(">>> VISION READY: model resident + warm, camera OK.\n")

    drone = System()
    await drone.connect(system_address=CONNECTION_ADDRESS)

    try:
        await asyncio.wait_for(wait_for_connection(drone), timeout=CONNECTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        print("[CONTROL ERROR] No connection within timeout.")
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
        print(">>> HARDWARE MODE -- V7 GPS CONTROL + PASSIVE VISION.")
        print(f">>> {FORWARD_DISTANCE_M}m forward, pause, {BACKWARD_DISTANCE_M}m back "
              f"at {AXIS_SPEED_MPS} m/s ({AXIS_SPEED_MPS * 3.6:.1f} km/h).")
        print(f">>> Model: {MODEL_NAME}  |  No flight-path changes on detection.")
        print(f">>> Video recording: {'ON -- saved per run' if SAVE_VIDEO else 'OFF'}")
        print(">>> WAIT for [HEALTH] global_pos_ok=True before triggering.")
        print(">>> Flip SWA (Ch6) to trigger GUIDED. Flip back at ANY time to abort.")
        print(f">>> Logs: {OUTPUT_DIR}")
        print("=" * 70 + "\n")

    run_index = 1
    try:
        while True:
            print("[CONTROL] Standing by (armed + GPS healthy + SWA to GUIDED)...")
            while True:
                if mode_state.is_armed and mode_state.current_mode == TARGET_TRIGGER_MODE:
                    if await wait_for_sustained_mode(TARGET_TRIGGER_MODE, TRIGGER_DEBOUNCE_S):
                        break
                await asyncio.sleep(0.1)

            # --- Start a fresh vision thread + per-run video file for this cycle ---
            ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = (
                os.path.join(VIDEO_DIR, f"video_{ts_str}_run{run_index:02d}.avi")
                if SAVE_VIDEO else None
            )

            vision_stop_event.clear()
            vision_ready.clear()
            vision_failed.clear()
            video_stats.clear()
            vision_thread = threading.Thread(
                target=vision_worker,
                args=(video_path, inferencer),
                daemon=True,
            )
            vision_thread.start()

            record = await run_takeover_sequence(drone, run_index, video_path)
            run_index += 1

            # Stop vision thread cleanly after this run completes.
            vision_stop_event.set()
            vision_thread.join(timeout=10.0)
            if vision_thread.is_alive():
                print("[VISION WARN] Vision thread still closing the video file.")
            record["video_stats"] = dict(video_stats)

            ts   = int(record["timestamp"])
            path = os.path.join(OUTPUT_DIR, f"run_{ts}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=4)

            n_det  = len(record.get("detections", []))
            dur    = record.get("flight_duration_s", "?")
            dist   = record.get("total_distance_m", "?")
            alt_mn = record.get("altitude_m_min", "?")
            alt_mx = record.get("altitude_m_max", "?")
            print(f"[DATA] Run log saved : {path}")
            print(f"[DATA] Flight log    : {record.get('flight_log', '?')}")
            if video_path:
                print(f"[DATA] Video         : {video_path}")
            print(f"[DATA] Duration      : {dur}s  |  Distance: {dist}m  |  Detections: {n_det}")
            print(f"[DATA] Altitude      : min={alt_mn}m  max={alt_mx}m")
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
        if "vision_thread" in dir() and vision_thread.is_alive():
            vision_thread.join(timeout=2.0)

if __name__ == "__main__":
    asyncio.run(run())
