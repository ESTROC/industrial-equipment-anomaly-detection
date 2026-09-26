from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import PredictionResponse, TelemetryRequest
from .service import load_metadata, load_model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_metadata()
    yield


app = FastAPI(
    title="Industrial Equipment Anomaly Detection API",
    version="1.0.0",
    description="ML inference API for synthetic industrial telemetry.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "industrial-equipment-anomaly-detection", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    meta = load_metadata()
    return {
        "status": "ok",
        "model_version": meta.get("model_version", "unknown"),
        "model_name": meta.get("model_name", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: TelemetryRequest):
    try:
        return predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
