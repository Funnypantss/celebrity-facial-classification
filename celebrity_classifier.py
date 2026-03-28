"""
============================================================
  Celebrity Image Classifier — ML Pipeline
============================================================
  Stack : Python · NumPy · Pandas · Scikit-learn · OpenCV/PIL
  Split : 80 / 20  train / test
  Model : Support Vector Machine (SVM) + PCA + Wavelet features
============================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import pywt                                   # PyWavelets  (pip install PyWavelets)
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import joblib                                  # model persistence

# ──────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────
DATASET_DIR   = "dataset/celebrities"   # root folder: subfolders = class names
MODEL_PATH    = "celebrity_svm_model.pkl"
IMG_SIZE      = (128, 128)              # resize target (H, W)
TEST_SIZE     = 0.20                    # 80-20 split
RANDOM_STATE  = 42
MIN_FACE_IMGS = 10                      # skip celebrity if < N cropped images

# Haar cascades bundled with OpenCV
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# ──────────────────────────────────────────────────────────
# 1.  IMAGE UTILITIES
# ──────────────────────────────────────────────────────────

def load_image_cv(path: str) -> np.ndarray | None:
    """Read an image from disk; return BGR ndarray or None on failure."""
    img = cv2.imread(path)
    return img if img is not None else None


def convert_to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Standardize: BGR → grayscale for feature extraction."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def detect_face_with_eyes(bgr: np.ndarray):
    """
    Detect a face region that also contains at least 2 eyes.
    Returns (x, y, w, h) of the best face ROI, or None.

    Strategy:
      1. Detect all faces in the full image (frontal cascade).
      2. For each face candidate crop the ROI and detect eyes inside it.
      3. Accept only the first ROI that has ≥ 2 eyes detected.
    """
    gray  = convert_to_grayscale(bgr)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        eyes     = EYE_CASCADE.detectMultiScale(roi_gray, minNeighbors=3)
        if len(eyes) >= 2:
            return (x, y, w, h)
    return None


def crop_and_resize(bgr: np.ndarray, roi) -> np.ndarray:
    """Crop face ROI and resize to IMG_SIZE."""
    x, y, w, h = roi
    face        = bgr[y: y + h, x: x + w]
    return cv2.resize(face, IMG_SIZE[::-1])   # cv2 takes (W, H)


# ──────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  — Wavelet + Pixel hybrid
# ──────────────────────────────────────────────────────────

def get_wavelet_features(gray_face: np.ndarray) -> np.ndarray:
    """
    Apply a single-level 2-D Haar wavelet transform on the grayscale face.
    Concatenate the four sub-bands (LL, LH, HL, HH) after resizing each to
    a fixed shape, giving multi-frequency texture features.
    """
    cA, (cH, cV, cD) = pywt.dwt2(gray_face.astype(np.float32), "haar")
    # each sub-band is half the original size — flatten and concatenate
    features = np.concatenate([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()])
    return features


def extract_features(bgr_face: np.ndarray) -> np.ndarray:
    """
    Combined feature vector for one face image:
      • Wavelet coefficients from the grayscale face (multi-frequency texture)
      • Raw flattened grayscale pixels (spatial information)
    """
    gray      = convert_to_grayscale(bgr_face)
    wavelet_f = get_wavelet_features(gray)
    pixel_f   = gray.ravel().astype(np.float32) / 255.0
    return np.concatenate([wavelet_f, pixel_f])


# ──────────────────────────────────────────────────────────
# 3.  DATASET LOADING
# ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str):
    """
    Walk DATASET_DIR. Each sub-directory name is treated as a celebrity label.

    Returns
    -------
    X : np.ndarray  shape (N, feature_dim)
    y : list[str]   celebrity names (raw labels)
    meta : pd.DataFrame  per-image metadata
    """
    X, y, records = [], [], []

    celebrities = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    for celeb in celebrities:
        celeb_dir = os.path.join(dataset_dir, celeb)
        face_imgs = []

        for fname in os.listdir(celeb_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(celeb_dir, fname)
            bgr   = load_image_cv(fpath)
            if bgr is None:
                continue

            roi = detect_face_with_eyes(bgr)
            if roi is None:
                continue                         # skip non-face or ambiguous images

            face_bgr = crop_and_resize(bgr, roi)
            feat     = extract_features(face_bgr)
            face_imgs.append((feat, fpath))

        if len(face_imgs) < MIN_FACE_IMGS:
            print(f"  [SKIP] {celeb}: only {len(face_imgs)} valid images found.")
            continue

        for feat, fpath in face_imgs:
            X.append(feat)
            y.append(celeb)
            records.append({"celebrity": celeb, "filepath": fpath})

        print(f"  [OK]   {celeb}: {len(face_imgs)} images loaded.")

    X    = np.array(X)
    meta = pd.DataFrame(records)
    return X, y, meta


# ──────────────────────────────────────────────────────────
# 4.  MODEL PIPELINE  — PCA + StandardScaler + SVM
# ──────────────────────────────────────────────────────────

def build_pipeline(n_components: int = 100) -> Pipeline:
    """
    sklearn Pipeline:
      StandardScaler → PCA (dimensionality reduction) → SVM (RBF kernel)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, whiten=True,
                       random_state=RANDOM_STATE)),
        ("svm",    SVC(kernel="rbf", probability=True,
                       random_state=RANDOM_STATE)),
    ])


def tune_hyperparameters(pipeline, X_train, y_train) -> Pipeline:
    """
    Lightweight grid-search over SVM C and gamma.
    Returns the best estimator.
    """
    param_grid = {
        "svm__C":     [1, 10, 50],
        "svm__gamma": ["scale", "auto"],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="accuracy",
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"\n  Best params : {grid.best_params_}")
    print(f"  CV accuracy : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ──────────────────────────────────────────────────────────
# 5.  EVALUATION HELPERS
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm  = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Celebrity Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def print_report(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  TEST ACCURACY : {acc * 100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ──────────────────────────────────────────────────────────
# 6.  PREDICTION UTILITY  (single image at inference time)
# ──────────────────────────────────────────────────────────

def predict_celebrity(image_path: str, model: Pipeline,
                      le: LabelEncoder) -> dict:
    """
    Given a path to an image, detect the face, extract features,
    and return the top-3 predictions with confidence scores.
    """
    bgr = load_image_cv(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    roi = detect_face_with_eyes(bgr)
    if roi is None:
        raise ValueError("No valid face (with 2 eyes) detected in the image.")

    face   = crop_and_resize(bgr, roi)
    feat   = extract_features(face).reshape(1, -1)
    proba  = model.predict_proba(feat)[0]
    top3   = np.argsort(proba)[::-1][:3]

    results = {
        "predictions": [
            {"celebrity": le.classes_[i], "confidence": float(proba[i])}
            for i in top3
        ]
    }
    return results


# ──────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Celebrity Image Classifier — ML Pipeline")
    print("=" * 55 + "\n")

    # ── 7.1  Load & preprocess dataset ──────────────────────
    print("[1/5]  Loading dataset …")
    if not os.path.isdir(DATASET_DIR):
        print(f"\n  ⚠  Dataset folder '{DATASET_DIR}' not found.")
        print("     Please organise your data as:\n")
        print("       dataset/celebrities/")
        print("         ├── lionel_messi/   (*.jpg, *.png)")
        print("         ├── serena_williams/")
        print("         └── …\n")
        print("  Running in DEMO mode with synthetic data …\n")
        _run_demo()
        return

    X, y_raw, meta = load_dataset(DATASET_DIR)
    print(f"\n  Total samples : {len(X)}")
    print(f"  Feature dim   : {X.shape[1]}")
    print(f"  Classes       : {sorted(set(y_raw))}")
    print(f"\n  Dataset summary:\n{meta['celebrity'].value_counts().to_string()}\n")

    # ── 7.2  Encode labels ───────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # ── 7.3  Train / Test split ─────────────────────────────
    print("[2/5]  Splitting 80 / 20 …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # ── 7.4  Build & tune pipeline ──────────────────────────
    print("\n[3/5]  Building pipeline & tuning hyperparameters …")
    base_pipeline = build_pipeline(n_components=min(100, X_train.shape[0] - 1))
    best_model    = tune_hyperparameters(base_pipeline, X_train, y_train)

    # ── 7.5  Evaluate ────────────────────────────────────────
    print("\n[4/5]  Evaluating on held-out test set …")
    y_pred       = best_model.predict(X_test)
    class_names  = le.classes_
    print_report(le.inverse_transform(y_test),
                 le.inverse_transform(y_pred),
                 class_names)
    plot_confusion_matrix(le.inverse_transform(y_test),
                          le.inverse_transform(y_pred),
                          class_names)

    # ── 7.6  Persist model ───────────────────────────────────
    print("\n[5/5]  Saving model …")
    joblib.dump({"model": best_model, "label_encoder": le}, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # ── 7.7  Quick inference demo ────────────────────────────
    print("\n── Inference demo ──────────────────────────────────")
    demo_path = meta["filepath"].iloc[0]           # first image in dataset
    result    = predict_celebrity(demo_path, best_model, le)
    print(f"  Image : {demo_path}")
    for p in result["predictions"]:
        print(f"    {p['celebrity']:<25}  {p['confidence']*100:5.1f}%")


# ──────────────────────────────────────────────────────────
# 8.  DEMO MODE  (no real dataset — synthetic data)
# ──────────────────────────────────────────────────────────

def _run_demo():
    """
    Generates synthetic feature vectors to showcase the full pipeline
    without requiring real image data.
    """
    np.random.seed(RANDOM_STATE)
    celebrities = ["Lionel_Messi", "Serena_Williams", "Shah_Rukh_Khan",
                   "Virat_Kohli", "Priyanka_Chopra"]
    N_PER_CLASS = 400            # 2 000 total  → matches resume bullet
    FEAT_DIM    = 2048

    X_list, y_list = [], []
    for i, celeb in enumerate(celebrities):
        # Each class has a distinct cluster centre in feature space
        centre = np.random.randn(FEAT_DIM) * 3
        data   = centre + np.random.randn(N_PER_CLASS, FEAT_DIM) * 0.8
        X_list.append(data)
        y_list.extend([celeb] * N_PER_CLASS)

    X     = np.vstack(X_list)
    le    = LabelEncoder()
    y     = le.fit_transform(y_list)
    total = len(X)

    print(f"  Synthetic samples : {total}")
    print(f"  Feature dim       : {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}\n")

    pipeline   = build_pipeline(n_components=100)
    best_model = tune_hyperparameters(pipeline, X_train, y_train)

    y_pred = best_model.predict(X_test)
    print_report(le.inverse_transform(y_test),
                 le.inverse_transform(y_pred),
                 le.classes_)
    plot_confusion_matrix(le.inverse_transform(y_test),
                          le.inverse_transform(y_pred),
                          le.classes_)

    joblib.dump({"model": best_model, "label_encoder": le}, MODEL_PATH)
    print(f"\n  Demo model saved → {MODEL_PATH}")
    print("\n  ✅  Demo complete.  Replace DATASET_DIR with real data to train properly.")


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
