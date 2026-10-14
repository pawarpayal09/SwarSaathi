"""Central settings for SwarSaathi. Every other module imports paths and column names from here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Aging_prediction_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.json"

TARGET = "aging_outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3          # repeated stratified CV: more stable than a single 5-fold run

NUMERIC_FEATURES = ["age", "education_years", "chronic_conditions",
                    "memory_score", "bmi", "blood_pressure"]
CATEGORICAL_FEATURES = ["gender", "income_level", "physical_activity",
                        "smoking_status", "diet_quality",
                        "alcohol_consumption", "social_engagement"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Documented "physically impossible" limits. A value outside these is treated
# as an invalid entry and becomes missing (NaN); the pipeline then imputes it.
# These are deliberately WIDE: they only catch impossible values, they are not
# clinical cut-offs.
VALID_RANGES = {
    "age": (0, 120),
    "education_years": (0, 30),
    "chronic_conditions": (0, 20),
    "memory_score": (0, 100),
    "bmi": (10, 70),
    "blood_pressure": (50, 300),
}

# ---- extra artefacts produced by training (used by the Streamlit app) ----------------
CLASS_ORDER = ["Healthy", "At-risk", "Frail"]          # least -> most severe
CLASS_COLORS = {"Healthy": "#16a34a", "At-risk": "#f59e0b", "Frail": "#dc2626"}
RISK_WEIGHTS = {"Healthy": 0.0, "At-risk": 0.5, "Frail": 1.0}   # used for the 0-100 risk index
OOF_REPORT_PATH = REPORTS_DIR / "oof_per_class_metrics.csv"
OOF_CM_PATH = REPORTS_DIR / "oof_confusion_matrix.png"
SAMPLE_BATCH_PATH = PROJECT_ROOT / "data" / "sample_batch_input.csv"
