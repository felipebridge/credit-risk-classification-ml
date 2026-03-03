from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CASE_STUDY_1_PATH = DATA_RAW_DIR / "case_study1.xlsx"
CASE_STUDY_2_PATH = DATA_RAW_DIR / "case_study2.xlsx"

ID_COL = "PROSPECTID"
TARGET_COL = "Approved_Flag"

MISSING_VALUE_SENTINEL = -99999

RANDOM_SEED = 42
TEST_SIZE = 0.2

DROP_CREDIT_SCORE = False
CREDIT_SCORE_COL = "Credit_Score"

MODEL_PATH = MODELS_DIR / "credit_approval_model.joblib"

METRICS_WITH_SCORE_PATH = REPORTS_DIR / "metrics_with_credit_score.txt"
METRICS_NO_SCORE_PATH = REPORTS_DIR / "metrics_without_credit_score.txt"

FEATURE_IMPORTANCE_WITH_SCORE_PATH = REPORTS_DIR / "feature_importance_with_score.csv"
FEATURE_IMPORTANCE_NO_SCORE_PATH = REPORTS_DIR / "feature_importance_without_score.csv"