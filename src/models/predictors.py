# src/models/predictors.py
"""
Stateless predictor classes used by both the API and the GUI.
Loading and inference are clearly separated from training code.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import tensorflow as tf

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import get_logger

logger = get_logger("predictors")


class DiabetesPredictor:
    """
    Wraps the sklearn Pipeline (imputer + scaler + classifier).

    The pipeline was saved as a single artifact, so there is no risk of
    applying the wrong scaler to a new model – a common production bug.
    """

    def __init__(self, models_dir: str = "models"):
        pipeline_path = os.path.join(models_dir, "diabetes_pipeline.pkl")
        meta_path     = os.path.join(models_dir, "diabetes_meta.json")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Diabetes pipeline not found at '{pipeline_path}'. "
                "Run src/training/train_diabetes.py first."
            )

        self.pipeline = joblib.load(pipeline_path)
        logger.info(f"Loaded diabetes pipeline from {pipeline_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.features: list[str] = self.meta["features"]

    def predict(self, feature_values: list[float]) -> dict:
        """
        Args:
            feature_values: Values in the SAME order as self.features.

        Returns:
            {
              "prediction": 0 | 1,
              "label": "No Diabetes" | "Diabetes",
              "probability": float,   # P(Diabetes)
              "confidence": float,    # max(P(class))
            }
        """
        if len(feature_values) != len(self.features):
            raise ValueError(
                f"Expected {len(self.features)} features, got {len(feature_values)}."
            )

        X = np.array([feature_values])
        pred  = int(self.pipeline.predict(X)[0])
        proba = float(self.pipeline.predict_proba(X)[0][1])

        return {
            "prediction":  pred,
            "label":       "Diabetes" if pred == 1 else "No Diabetes",
            "probability": round(proba, 4),
            "confidence":  round(max(proba, 1 - proba), 4),
        }


class PneumoniaPredictor:
    """
    Wraps a TensorFlow SavedModel for pneumonia classification.

    Preprocessing (resize + normalise) is done here, not in the GUI/API
    layer, keeping inference logic in one place.
    """

    def __init__(self, models_dir: str = "models"):
        model_path = os.path.join(models_dir, "pneumonia_model.keras")
        meta_path  = os.path.join(models_dir, "pneumonia_meta.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pneumonia model not found at '{model_path}'. "
                "Run src/training/train_pneumonia.py first."
            )

        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded pneumonia model from {model_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.img_size: tuple[int, int] = tuple(self.meta["img_size"])
        self.class_names: list[str]    = self.meta["class_names"]

    def predict_from_path(self, img_path: str) -> dict:
        """Load an image from disk and return prediction."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
        return self._predict_img(img)

    def predict_from_bytes(self, img_bytes: bytes) -> dict:
        """Load an image from raw bytes (used by the API)."""
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).resize(self.img_size)
        return self._predict_img(img)

    def _predict_img(self, img) -> dict:
        """
        Internal helper – accepts a PIL Image or Keras image object.
        EfficientNetB0's built-in preprocessing expects uint8 [0,255].
        """
        arr   = tf.keras.utils.img_to_array(img)           # HxWx3, float32
        batch = tf.expand_dims(arr, 0)                      # 1xHxWx3
        score = float(self.model.predict(batch, verbose=0)[0][0])

        # class_names are ordered by directory name (alphabetical):
        # index 0 = NORMAL, index 1 = PNEUMONIA  (matches sigmoid > 0.5)
        pred  = 1 if score >= 0.5 else 0
        label = self.class_names[pred] if self.class_names else ("PNEUMONIA" if pred else "NORMAL")

        return {
            "prediction":  pred,
            "label":       label,
            "probability": round(score, 4),
            "confidence":  round(score if pred == 1 else 1 - score, 4),
        }
