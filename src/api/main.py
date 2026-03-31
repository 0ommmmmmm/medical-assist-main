# src/api/main.py
"""
FastAPI inference server.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict/diabetes   – JSON body with patient features
    POST /predict/pneumonia  – multipart/form-data with chest X-ray image
    GET  /health             – liveness check
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.predictors import DiabetesPredictor, PneumoniaPredictor
from src.utils.helpers import get_logger, load_config

logger = get_logger("api")

# ---------------------------------------------------------------------------
# Application lifespan – load models once at startup
# ---------------------------------------------------------------------------

predictors: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; release on shutdown."""
    cfg = load_config()
    models_dir = cfg["paths"]["models_dir"]

    logger.info("Loading models…")
    try:
        predictors["diabetes"]  = DiabetesPredictor(models_dir)
        logger.info("Diabetes model ready.")
    except FileNotFoundError as e:
        logger.warning(str(e))

    try:
        predictors["pneumonia"] = PneumoniaPredictor(models_dir)
        logger.info("Pneumonia model ready.")
    except FileNotFoundError as e:
        logger.warning(str(e))

    yield   # app runs here

    predictors.clear()
    logger.info("Models unloaded.")


app = FastAPI(
    title="Medical Diagnosis Assistant API",
    version="2.0.0",
    description=(
        "**⚠️ FOR EDUCATIONAL USE ONLY.** "
        "Not a substitute for professional medical advice."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DiabetesRequest(BaseModel):
    pregnancies:                float = Field(..., ge=0,  le=20,  description="Number of pregnancies")
    glucose:                    float = Field(..., ge=0,  le=300, description="Plasma glucose (mg/dL)")
    blood_pressure:             float = Field(..., ge=0,  le=200, description="Diastolic BP (mm Hg)")
    skin_thickness:             float = Field(..., ge=0,  le=100, description="Triceps skinfold (mm)")
    insulin:                    float = Field(..., ge=0,  le=900, description="2-hour serum insulin (mu U/ml)")
    bmi:                        float = Field(..., ge=0,  le=100, description="Body mass index")
    diabetes_pedigree_function: float = Field(..., ge=0,  le=3.0, description="Diabetes pedigree function")
    age:                        int   = Field(..., ge=1,  le=120, description="Age in years")

    @field_validator("glucose", "blood_pressure", "bmi")
    @classmethod
    def must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0 (0 is biologically invalid)")
        return v


class PredictionResponse(BaseModel):
    prediction:  int
    label:       str
    probability: float
    confidence:  float
    disclaimer:  str = "For educational use only. Consult a qualified healthcare professional."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "models_loaded": list(predictors.keys()),
        "timestamp": time.time(),
    }


@app.post("/predict/diabetes", response_model=PredictionResponse, tags=["Prediction"])
def predict_diabetes(req: DiabetesRequest):
    if "diabetes" not in predictors:
        raise HTTPException(503, "Diabetes model not loaded. Run training first.")

    features = [
        req.pregnancies, req.glucose, req.blood_pressure,
        req.skin_thickness, req.insulin, req.bmi,
        req.diabetes_pedigree_function, req.age,
    ]
    result = predictors["diabetes"].predict(features)
    return PredictionResponse(**result)


@app.post("/predict/pneumonia", response_model=PredictionResponse, tags=["Prediction"])
async def predict_pneumonia(file: UploadFile = File(...)):
    if "pneumonia" not in predictors:
        raise HTTPException(503, "Pneumonia model not loaded. Run training first.")

    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:   # 10 MB guard
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")

    result = predictors["pneumonia"].predict_from_bytes(img_bytes)
    return PredictionResponse(**result)
