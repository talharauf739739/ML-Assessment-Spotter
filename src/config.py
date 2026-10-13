"""
Configuration parameters, directory paths, and column constants
for the Spotter Freight Rate ML pipeline.
"""
from pathlib import Path
import sys

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "scorer_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths (Single Source of Truth in data/)
TRAIN_DATA_PATH = DATA_DIR / "train_test.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation.csv"
DECEMBER_CHART_INPUTS_PATH = DATA_DIR / "december_chart_inputs.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"

# Temporal Split Dates
# Development data spans Jan 1, 2025 to Oct 31, 2025.
# Validation holdout set spans Sep 1, 2025 to Oct 31, 2025 (2 months to mirror the 2-month test horizon Nov-Dec).
OOT_SPLIT_DATE = "2025-09-01"

# Reproducibility
RANDOM_SEED = 42

# Column Definitions
ID_COL = "load_id"
DATE_COL = "date"
TARGET_COL = "posted_rate"

RAW_NUMERIC_COLS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

RAW_CATEGORICAL_COLS = [
    "pickup",
    "delivery",
    "equipment",
]

EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed"]
