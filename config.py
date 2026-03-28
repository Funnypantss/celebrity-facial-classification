"""
config.py
Central configuration for the Celebrity Image Classifier.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed")
DATASET_CSV_PATH = os.getenv("DATASET_CSV_PATH", "data/dataset.csv")
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR", "models/saved")

# Haar Cascade XML paths (bundled with OpenCV; copy to project dir for portability)
HAARCASCADE_FACE = os.getenv(
    "HAARCASCADE_FACE",
    "haarcascades/haarcascade_frontalface_default.xml",
)
HAARCASCADE_EYE = os.getenv(
    "HAARCASCADE_EYE",
    "haarcascades/haarcascade_eye.xml",
)

# ── Image Processing ──────────────────────────────────────────────────────────
IMAGE_SIZE = (32, 32)           # Target size after face crop
MIN_EYES_REQUIRED = 2           # Minimum eyes detected to accept a face crop
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Feature Engineering ───────────────────────────────────────────────────────
WAVELET = "db1"                  # Daubechies wavelet family
WAVELET_LEVEL = 2                # Decomposition level for PyWavelets

# ── Model ─────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5                     # Cross-validation folds

# Hyperparameter grids for GridSearchCV
SVM_PARAM_GRID = {
    "C": [1, 10, 100],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
}

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
