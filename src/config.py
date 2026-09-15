"""
Centralized configuration for the Classical Computer Vision & Tabular ML Vehicle Damage Assessment Pipeline.
Ensures zero-leakage reproducibility across all feature extraction, validation, and evaluation steps.
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
LEGACY_DIR = WORKSPACE_ROOT / "legacy" / "deep_learning_pipeline"

# Processed Feature Table Paths
FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "features.csv"
FEATURES_XLSX_PATH = PROCESSED_DATA_DIR / "features.xlsx"
FEATURES_SEVERITY_CSV = PROCESSED_DATA_DIR / "features_severity.csv"
FEATURES_CARDD_CSV = PROCESSED_DATA_DIR / "features_cardd.csv"

# Ensure all critical directories exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
         EXPERIMENTS_DIR, PLOTS_DIR, MODELS_DIR, METRICS_DIR, LEGACY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Global Reproducibility Seed
RANDOM_SEED = 42

# CarDD Damage Category Classes
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

# Severity Benchmark Classes
SEVERITY_CLASSES = ["normal", "moderate_breakage", "severe_crushed"]
SEVERITY_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SEVERITY_CLASSES)}
SEVERITY_IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(SEVERITY_CLASSES)}

# Classical Computer Vision Preprocessing Constants
IMG_STANDARDIZED_SIZE = (512, 512)
TEXTURE_EVAL_SIZE = (256, 256)
FFT_EVAL_SIZE = (128, 128)

# Zero-Leakage Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Central Actuarial Currency Conversion Anchor (USD -> INR)
# Fixed conversion rate as of project submission (1 USD = 95.5 INR)
USD_TO_INR = 95.5
CURRENCY_SYMBOL = "₹"

