# Medical Diagnosis Assistant v2.0

> ⚠️ **DISCLAIMER**: For educational purposes ONLY. Not a substitute for professional medical advice.

Medical Assist is a portfolio-ready healthcare ML project with:

- A **FastAPI backend** for model inference
- A **Streamlit frontend** for a clean web demo
- A **diabetes prediction model** using tabular clinical inputs
- A **pneumonia detection model** using chest X-ray image uploads

## Streamlit UI

The Streamlit app includes:

- Home page with project intro and medical disclaimer
- Diabetes Prediction page with a guided input form
- Pneumonia Detection page with chest X-ray upload
- Results page with prediction, probability, confidence, and explanation cards
- About page covering tech stack, models, and limitations

## Screenshots

Add screenshots after running the app locally:

| Home | Diabetes Prediction | Pneumonia Detection |
|---|---|---|
| `assets/screenshots/home.png` | `assets/screenshots/diabetes.png` | `assets/screenshots/pneumonia.png` |

## Project Structure

```
medical-assist-pro/
├── app/
│   ├── prediction_client.py     # Streamlit prediction helpers
│   └── ui.py                    # Shared UI styling/components
├── pages/
│   ├── 1_Diabetes_Prediction.py
│   ├── 2_Pneumonia_Detection.py
│   ├── 3_Results.py
│   └── 4_About.py
├── streamlit_app.py             # Streamlit entry point
├── .streamlit/
│   └── config.toml              # Streamlit theme
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

# 1. Train models if model files are missing
python src/training/train_diabetes.py
python src/training/train_pneumonia.py

# 2. Start FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# 3. Open interactive API docs
open http://localhost:8000/docs
```

## Run the Streamlit Frontend

### Option A: Streamlit with local model files

This is the simplest path for Streamlit Cloud and local demos.

```bash
streamlit run streamlit_app.py
```

### Option B: Streamlit calling FastAPI

Run the API first:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then start Streamlit with the backend URL:

```bash
MEDICAL_ASSIST_API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

Use these settings:

- **Main file path:** `streamlit_app.py`
- **Python dependencies:** `requirements.txt`
- **Model files:** keep trained model artifacts in the `models/` folder or provide them through your deployment storage workflow
- **Optional secret/env var:** `MEDICAL_ASSIST_API_URL` if you deploy FastAPI separately

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

## Important Limitations

- This app is not clinically validated.
- Results are model outputs for educational demonstration only.
- Chest X-ray interpretation requires professional radiology review.
- Confidence is not the same as medical certainty.
# Med-Host
