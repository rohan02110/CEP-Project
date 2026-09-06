"""
Centralized configuration for the AI-Assisted Vehicle Damage Assessment Pipeline.
Ensures reproducibility across all training, validation, and evaluation steps.
"""

from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

EXPERIMENTS_DIR = WORKSPACE_ROOT / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"
MODELS_DIR = EXPERIMENTS_DIR / "models"
METRICS_DIR = EXPERIMENTS_DIR / "metrics"

# Ensure all critical directories exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
         EXPERIMENTS_DIR, PLOTS_DIR, MODELS_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Global Reproducibility Seed
RANDOM_SEED = 42

# CarDD Damage Detection Configuration
CARDD_CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat"
]
CARDD_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CARDD_CLASSES)}
CARDD_IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CARDD_CLASSES)}

# Severity Benchmark Configuration
SEVERITY_CLASSES = ["normal", "moderate_breakage", "severe_crushed"]
SEVERITY_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SEVERITY_CLASSES)}
SEVERITY_IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(SEVERITY_CLASSES)}

# Image Preprocessing Constants
IMG_SIZE_DETECTION = 640
IMG_SIZE_CLASSIFICATION = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Split Ratios (Leakage-free)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
