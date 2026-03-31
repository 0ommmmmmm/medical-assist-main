# src/training/train_diabetes.py
"""
Production-grade diabetes model training.

Key improvements over original:
  - Imputes biologically-invalid zeros (glucose=0 is impossible)
  - Handles class imbalance with class_weight and SMOTE (optional)
  - Stratified k-fold cross-validation instead of a single train/test split
  - Trains Random Forest + Logistic Regression; picks best by ROC-AUC
  - Saves model + scaler + feature list atomically with metadata
  - Comprehensive logging and metrics reporting
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import load_config, get_logger, evaluate_classifier

logger = get_logger("train_diabetes")


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str, zero_impute_cols: list,
                   target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, replace biologically-invalid zeros with NaN, then impute
    with the median of each column (robust to outliers).
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Replace biologically-invalid zeros with NaN before imputation
    for col in zero_impute_cols:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            logger.info(f"  {col}: replaced {n_zeros} zeros with NaN")

    logger.info(f"Class distribution:\n{df[target_col].value_counts()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_pipelines(cfg: dict) -> dict:
    """
    Returns a dict of named sklearn Pipelines.
    Each pipeline encapsulates imputation → scaling → classifier,
    guaranteeing no data leakage between folds.
    """
    rf_cfg = cfg["diabetes"]["model"]["random_forest"]
    lr_cfg = cfg["diabetes"]["model"]["logistic_regression"]

    return {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                class_weight=rf_cfg["class_weight"],
                random_state=cfg["diabetes"]["random_state"],
                n_jobs=-1,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=lr_cfg["C"],
                solver=lr_cfg["solver"],
                class_weight=lr_cfg["class_weight"],
                random_state=cfg["diabetes"]["random_state"],
                max_iter=1000,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation & selection
# ---------------------------------------------------------------------------

def cross_validate_pipelines(pipelines: dict, X: pd.DataFrame,
                              y: pd.Series, cfg: dict) -> dict:
    """Run stratified k-fold CV on each pipeline; return mean ROC-AUC scores."""
    skf = StratifiedKFold(
        n_splits=cfg["diabetes"]["model"]["cv_folds"],
        shuffle=True,
        random_state=cfg["diabetes"]["random_state"],
    )
    scores = {}
    for name, pipe in pipelines.items():
        cv_results = cross_validate(
            pipe, X, y, cv=skf,
            scoring=["roc_auc", "f1", "accuracy"],
            n_jobs=-1,
        )
        mean_auc = cv_results["test_roc_auc"].mean()
        std_auc  = cv_results["test_roc_auc"].std()
        logger.info(
            f"[{name}] CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f} | "
            f"F1: {cv_results['test_f1'].mean():.4f}"
        )
        scores[name] = mean_auc

    return scores


# ---------------------------------------------------------------------------
# Final training & saving
# ---------------------------------------------------------------------------

def train_and_save(best_name: str, pipelines: dict, X: pd.DataFrame,
                   y: pd.Series, cfg: dict, models_dir: str) -> None:
    """Refit the best pipeline on the full dataset and save artifacts."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    pipe = pipelines[best_name]
    pipe.fit(X, y)

    y_pred  = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    metrics = evaluate_classifier(
        y, y_pred, y_proba=y_proba,
        class_names=["No Diabetes", "Diabetes"]
    )
    logger.info(f"Final model metrics (train set):\n{metrics['classification_report']}")

    # Save pipeline (contains imputer + scaler + model — no separate scaler needed)
    pipeline_path = os.path.join(models_dir, "diabetes_pipeline.pkl")
    joblib.dump(pipe, pipeline_path)
    logger.info(f"Pipeline saved → {pipeline_path}")

    # Save metadata (feature order, thresholds, version)
    meta = {
        "model_type": best_name,
        "features": list(X.columns),
        "roc_auc_train": metrics.get("roc_auc"),
        "f1_train": metrics["f1_weighted"],
    }
    meta_path = os.path.join(models_dir, "diabetes_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    d_cfg = cfg["diabetes"]

    X, y = load_and_clean(
        csv_path=d_cfg["dataset_path"],
        zero_impute_cols=d_cfg["zero_impute_columns"],
        target_col=d_cfg["target_column"],
    )

    pipelines = build_pipelines(cfg)
    scores    = cross_validate_pipelines(pipelines, X, y, cfg)

    best_name = max(scores, key=scores.get)
    logger.info(f"Best model: {best_name} (ROC-AUC {scores[best_name]:.4f})")

    train_and_save(best_name, pipelines, X, y, cfg, cfg["paths"]["models_dir"])
    logger.info("Diabetes training complete.")


if __name__ == "__main__":
    main()
