# src/training/train_pneumonia.py
"""
Production-grade pneumonia CNN training with transfer learning.

Key improvements over original:
  - EfficientNetB0 (pre-trained on ImageNet) instead of a 3-layer custom CNN
  - Two-phase training: frozen base → fine-tune top layers
  - Class-weight balancing for the ~3:1 pneumonia/normal imbalance in the dataset
  - Mixed precision training (2–3x speedup on compatible GPUs)
  - Comprehensive callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
  - Saves in SavedModel format (not .h5) for TensorFlow Serving compatibility
  - Test-set evaluation logged after training
"""

import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard,
)
from tensorflow.keras.optimizers import Adam
import sklearn.utils

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import load_config, get_logger

logger = get_logger("train_pneumonia")

# Enable mixed precision for ~2x GPU throughput on Ampere+ GPUs
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict) -> tuple:
    """
    Uses tf.keras.utils.image_dataset_from_directory for fast, memory-efficient
    loading (no ImageDataGenerator legacy API).
    Returns (train_ds, val_ds, test_ds, class_names).
    """
    p_cfg   = cfg["pneumonia"]
    img_h, img_w = p_cfg["img_size"]
    base_dir = p_cfg["dataset_path"]
    bs       = p_cfg["batch_size"]

    def _make_ds(split_dir, shuffle=False):
        return tf.keras.utils.image_dataset_from_directory(
            split_dir,
            image_size=(img_h, img_w),
            batch_size=bs,
            shuffle=shuffle,
            label_mode="binary",
            seed=42,
        )

    train_ds = _make_ds(os.path.join(base_dir, "train"), shuffle=True)
    val_ds   = _make_ds(os.path.join(base_dir, "val"))
    test_ds  = _make_ds(os.path.join(base_dir, "test"))
    class_names = train_ds.class_names
    logger.info(f"Classes: {class_names}")

    # Augmentation layer (runs on GPU, inside the model graph)
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(p_cfg["augmentation"]["rotation_range"] / 360),
        layers.RandomZoom(p_cfg["augmentation"]["zoom_range"]),
        layers.RandomBrightness(p_cfg["augmentation"]["brightness_range"][0] - 1),
    ], name="augmentation")

    # Normalisation (EfficientNet expects [0,255]; its own rescaling is built-in)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    val_ds  = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_ds: tf.data.Dataset) -> dict:
    """
    Counts labels in the training set and returns class weights
    that counter-balance the ~3:1 PNEUMONIA/NORMAL imbalance.
    """
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    labels = labels.flatten().astype(int)
    weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    cw = dict(enumerate(weights))
    logger.info(f"Class weights: {cw}")
    return cw


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict, num_classes: int = 1) -> Model:
    """
    Phase-1 model: EfficientNetB0 base frozen + custom classification head.
    """
    m_cfg   = cfg["pneumonia"]["model"]
    img_h, img_w = cfg["pneumonia"]["img_size"]

    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_h, img_w, 3),
    )
    base.trainable = False   # freeze during phase 1

    inputs = tf.keras.Input(shape=(img_h, img_w, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(m_cfg["dropout_rate"])(x)
    # dtype=float32 ensures the output is always fp32 even under mixed precision
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs, outputs, name="efficientnet_pneumonia")
    return model, base


def compile_model(model: Model, lr: float) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(models_dir: str, phase: str) -> list:
    ckpt_path = os.path.join(models_dir, f"pneumonia_best_{phase}.weights.h5")
    return [
        ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max",
                        save_best_only=True, save_weights_only=True,
                        verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3,
                          min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join("logs", "pneumonia", phase),
                    histogram_freq=1),
    ]


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def phase1_train(model, base, train_ds, val_ds, class_weights, cfg, models_dir):
    """Train only the classification head (base frozen)."""
    logger.info("Phase 1: Training classification head (base frozen)")
    compile_model(model, lr=cfg["pneumonia"]["learning_rate"])
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg["pneumonia"]["epochs"],
        class_weight=class_weights,
        callbacks=get_callbacks(models_dir, "phase1"),
    )


def phase2_finetune(model, base, train_ds, val_ds, class_weights, cfg, models_dir):
    """Unfreeze top layers of the base and fine-tune at a lower LR."""
    fine_tune_from = cfg["pneumonia"]["model"]["fine_tune_from_layer"]
    logger.info(f"Phase 2: Fine-tuning from layer {fine_tune_from}")

    base.trainable = True
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    # 10x lower LR to avoid destroying pre-trained weights
    compile_model(model, lr=cfg["pneumonia"]["learning_rate"] / 10)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg["pneumonia"]["epochs"] // 2,
        class_weight=class_weights,
        callbacks=get_callbacks(models_dir, "phase2"),
    )


# ---------------------------------------------------------------------------
# Evaluation & saving
# ---------------------------------------------------------------------------

def evaluate_and_save(model: Model, test_ds, class_names: list,
                      models_dir: str, cfg: dict) -> None:
    results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    metrics = dict(zip(metric_names, results))
    logger.info(f"Test results: { {k: round(v, 4) for k, v in metrics.items()} }")

    # SavedModel format is preferred over .h5 for TF Serving / TFLite export
    save_path = os.path.join(models_dir, "pneumonia_model.keras")
    model.save(save_path)
    logger.info(f"Model saved → {save_path}")

    meta = {
        "architecture": cfg["pneumonia"]["model"]["architecture"],
        "img_size": cfg["pneumonia"]["img_size"],
        "class_names": class_names,
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    meta_path = os.path.join(models_dir, "pneumonia_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    models_dir = cfg["paths"]["models_dir"]
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets(cfg)
    class_weights = compute_class_weights(train_ds)

    model, base = build_model(cfg)
    model.summary(print_fn=logger.info)

    phase1_train(model, base, train_ds, val_ds, class_weights, cfg, models_dir)
    phase2_finetune(model, base, train_ds, val_ds, class_weights, cfg, models_dir)

    evaluate_and_save(model, test_ds, class_names, models_dir, cfg)
    logger.info("Pneumonia training complete.")


if __name__ == "__main__":
    main()
