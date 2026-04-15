from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import time
import logging

from src.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Tigrinya Harmful Content Detector API",
    version="1.2.0"
)

predictor = None

try:
    predictor = Predictor(
        model_path="models/xgb_model.json",
        feature_builder_path="models/feature_builder.pkl"
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")

class TextRequest(BaseModel):
    content: str = Field(..., min_length=3, max_length=2000)

class PredictionResponse(BaseModel):
    input_text: str
    label: str
    confidence: float
    explanation: str
    latency_ms: float

class BatchRequest(BaseModel):
    items: List[str]

class BatchItem(BaseModel):
    input_text: str
    label: str
    confidence: float

class BatchResponse(BaseModel):
    results: List[BatchItem]

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/v1/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start = time.time()
    text = request.content.strip()

    proba = float(predictor.predict_proba(text))
    label = "Harmful" if proba >= 0.5 else "Neutral"

    return PredictionResponse(
        input_text=text,
        label=label,
        confidence=proba,
        explanation="Harmful content detected" if label == "Harmful" else "Safe content",
        latency_ms=round((time.time() - start) * 1000, 2)
    )

@app.post("/v1/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []

    for text in request.items:
        text = text.strip()
        proba = float(predictor.predict_proba(text))
        label = "Harmful" if proba >= 0.5 else "Neutral"

        results.append(BatchItem(
            input_text=text,
            label=label,
            confidence=proba
        ))

    return BatchResponse(results=results)