"""
main.py
End-to-end pipeline for the Celebrity Image Classifier.

Steps:
  1. Build (or reload) the feature dataset from raw images
  2. Train SVM, KNN, and Random Forest classifiers
  3. Select and save the best model
  4. Print metrics and save visualization charts
"""

import logging
import os
from pathlib import Path
import config
from src.dataset_builder import build_dataset, load_dataset
from src.model import train_and_evaluate, select_best_model, save_model
from src.visualizer import plot_confusion_matrix, plot_class_distribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure output directories exist
for d in [config.DATA_RAW_PATH, config.DATA_PROCESSED_PATH, config.MODEL_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)


def run(rebuild_dataset: bool = False) -> None:
    logger.info("═══ Celebrity Image Classifier Pipeline ═══")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    csv_path = config.DATASET_CSV_PATH
    if rebuild_dataset or not Path(csv_path).exists():
        logger.info("Building dataset from raw images…")
        build_dataset()
    else:
        logger.info(f"Using existing dataset: {csv_path}")

    X, y, classes = load_dataset(csv_path)
    logger.info(f"Classes: {classes}")

    # ── 2. Train All Classifiers ──────────────────────────────────────────────
    results = []
    for clf_name in ["svm", "knn", "random_forest"]:
        result = train_and_evaluate(X, y, classes, classifier_name=clf_name)
        results.append(result)
        print(f"\n── {clf_name.upper()} ──")
        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"Best Params: {result['best_params']}")
        print(result["report"])

    # ── 3. Select & Save Best Model ───────────────────────────────────────────
    best = select_best_model(results)
    print(f"\n🏆 Best model: {best['classifier_name'].upper()} — Accuracy: {best['accuracy']:.4f}")
    model_path = save_model(best["model"], classes)
    print(f"   Saved → {model_path}")

    # ── 4. Visualizations ─────────────────────────────────────────────────────
    plot_confusion_matrix(best["confusion_matrix"], classes)
    plot_class_distribution(y, classes)
    logger.info("Pipeline complete. Charts saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Celebrity Image Classifier")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the feature dataset from raw images.",
    )
    args = parser.parse_args()
    run(rebuild_dataset=args.rebuild)
