# src/utils/helpers.py

import logging
import os
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that writes to both console and a rotating file.
    Using a shared logger prevents duplicate handlers on re-import.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray = None,
                         class_names: list = None) -> dict:
    """
    Returns a comprehensive metrics dict for a binary/multi-class classifier.

    Args:
        y_true:     Ground-truth labels.
        y_pred:     Hard predictions.
        y_proba:    Probability estimates for the positive class (binary)
                    or all classes (multi-class) – used for AUC.
        class_names: Optional list of label names for the report.

    Returns:
        dict with accuracy, f1, roc_auc (if y_proba supplied),
        confusion_matrix, and full classification_report string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    metrics = {
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba,
                                multi_class="ovr", average="weighted")
            metrics["roc_auc"] = round(float(auc), 4)
        except ValueError:
            pass   # single-class edge case during dev

    return metrics
