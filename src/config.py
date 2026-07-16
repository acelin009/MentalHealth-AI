from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# CONSISTENT: Using MODELS_DIR (plural)
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# ==========================================
# Dataset Paths
# ==========================================

MODELING_DATASET = PROCESSED_DATA_DIR / "ml_ready_dataset.csv"
ENGINEERED_DATASET = PROCESSED_DATA_DIR / "engineered_osmi_data.csv"

# ==========================================
# Saved Models
# ==========================================

BEST_MODEL = MODELS_DIR / "best_model.pkl"
PREPROCESSOR = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, DASHBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)