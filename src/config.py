"""Project configuration and paths."""

import os

# Project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Checkpoints directory (all model weights go here)
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Model paths
MODEL_CANOPY = os.path.join(CHECKPOINTS_DIR, "canopy11s640.pt")

# Video paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets/canopy")
DEFAULT_VIDEO = os.path.join(DATASET_DIR, "DJI_0009.MOV")

# Image dimensions
IMG_WIDTH = 640
IMG_HEIGHT = 384

# Tracking: minimum IoU with previous bbox to accept a detection as the same segment (else LOST)
TRACK_IOU_THRESHOLD = 0.6

# Training
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
TRAIN_RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "segment")
