# ============================================================
# SAR Drone - MMDetection Training Launcher
#
# Launches a full training run for ONE mmdet model at a time on
# the SARD dataset, using transfer learning from COCO pretrained
# weights. Uses mmdet's own tools/train.py under the hood.
#
# HOW TO RUN:
#   source ~/mmdet_train_env/bin/activate
#   export CUDA_HOME=/usr/local/cuda-12.6
#   export PATH=$CUDA_HOME/bin:$PATH
#   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#   python3 train_launcher.py
#
# WORKFLOW:
#   1. Uncomment ONE model block in the CONFIG section below
#   2. Run this script - it will call mmdet's training tool for you
#   3. Repeat for each model you want to fine-tune
#   4. Best checkpoint for each run is saved under work_dirs/<run>/
#
# NOTE: This script does NOT use mmdet's own GUI/CLI directly,
# it just builds and runs the correct command so you don't have
# to remember paths/flags for each model. You can still run the
# underlying mmdet command manually if you prefer - it's printed
# before running either way.
# ============================================================

import subprocess
import sys
import os

# ---- CONFIG (uncomment ONE model block at a time) -----------

# MODEL_NAME = "RTMDet-tiny-HERIDAL"
# CONFIG_PATH = os.path.expanduser("~/sar_training/configs/rtmdet_tiny_heridal_960_ext.py")

MODEL_NAME = "YOLOX-tiny"
CONFIG_PATH = os.path.expanduser("~/sar_training/configs/yolox_tiny_visdrone_960.py")

# MODEL_NAME = "Faster-RCNN-R50"
# CONFIG_PATH = os.path.expanduser("~/sar_training/configs/faster_rcnn_r50_sard.py")

# ---- OPTIONAL OVERRIDES --------------------------------------
# Leave as None to use whatever is already set in the config file.
# Set a value here to override just for this run, without editing
# the config (useful for a quick smoke test before a full run).
OVERRIDE_MAX_EPOCHS = 50   # e.g. 1 for a fast dry run, None to use config default
OVERRIDE_BATCH_SIZE = 2     # e.g. 4 if you hit GPU memory issues

# ---- END CONFIG -------------------------------------------------

MMDETECTION_ROOT = os.path.expanduser("~/mmdetection")
TRAIN_SCRIPT = os.path.join(MMDETECTION_ROOT, "tools", "train.py")


def main():
    if not os.path.isfile(CONFIG_PATH):
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        print("Check the path, or that you've uncommented a valid model block.")
        sys.exit(1)

    if not os.path.isfile(TRAIN_SCRIPT):
        print(f"ERROR: mmdet train.py not found at: {TRAIN_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, TRAIN_SCRIPT, CONFIG_PATH]

    cfg_options = []
    if OVERRIDE_MAX_EPOCHS is not None:
        cfg_options.append(f"train_cfg.max_epochs={OVERRIDE_MAX_EPOCHS}")
    if OVERRIDE_BATCH_SIZE is not None:
        cfg_options.append(f"train_dataloader.batch_size={OVERRIDE_BATCH_SIZE}")

    if cfg_options:
        cmd.append("--cfg-options")
        cmd.extend(cfg_options)

    print(f"\n{'='*60}")
    print(f"Launching training: {MODEL_NAME}")
    print(f"Config: {CONFIG_PATH}")
    if cfg_options:
        print(f"Overrides: {cfg_options}")
    print(f"{'='*60}")
    print(f"\nEquivalent manual command:\n{' '.join(cmd)}\n")

    # Run from the mmdetection root so relative imports/paths inside
    # mmdet's own tooling resolve correctly
    subprocess.run(cmd, cwd=MMDETECTION_ROOT)


if __name__ == "__main__":
    main()
