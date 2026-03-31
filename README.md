# Medical Diagnosis Assistant v2.0

> ⚠️ **DISCLAIMER**: For educational purposes ONLY. Not a substitute for professional medical advice.

## Project Structure

```
medical-assist-pro/
├── configs/
│   └── config.yaml              # All hyperparameters and paths in one place
├── src/
│   ├── utils/
│   │   └── helpers.py           # Logger, config loader, metrics
│   ├── training/
│   │   ├── train_diabetes.py    # Diabetes model training
│   │   └── train_pneumonia.py   # Pneumonia CNN training
│   ├── models/
│   │   └── predictors.py        # Inference classes (used by API + GUI)
│   └── api/
│       └── main.py              # FastAPI REST server
├── models/                      # Saved model artifacts (git-ignored)
├── data/                        # Datasets (git-ignored)
├── logs/                        # Training + API logs
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Train models
python src/training/train_diabetes.py
python src/training/train_pneumonia.py

# 2. Start API
uvicorn src.api.main:app --reload --port 8000

# 3. Open interactive docs
open http://localhost:8000/docs
```

## Key Improvements Over v1

| Area | Before | After |
|---|---|---|
| **Pneumonia model** | 3-layer custom CNN | EfficientNetB0 (ImageNet pre-trained, 2-phase fine-tune) |
| **Diabetes preprocessing** | Raw zeros passed to model | Invalid zeros imputed with median |
| **Class imbalance** | Ignored | `class_weight="balanced"` + configurable SMOTE |
| **Validation** | Single 80/20 split | Stratified 5-fold cross-validation |
| **Model persistence** | Separate model + scaler files | Single Pipeline (no scaler mismatch bugs) |
| **Inference** | Scattered in GUI/CLI | Centralised `DiabetesPredictor` / `PneumoniaPredictor` |
| **API** | None | FastAPI with Pydantic validation + file-size guard |
| **Config** | Hardcoded magic numbers | `configs/config.yaml` |
| **Logging** | `print()` statements | Structured logger → file + console |
| **GPU** | Manual | Mixed precision auto-enabled |
| **Saving format** | `.h5` (legacy) | TF SavedModel (TF Serving compatible) |
# Medical-Assist-Main
