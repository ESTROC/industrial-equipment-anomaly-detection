import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata():
    if not METADATA_PATH.exists():
        return {"model_version": "unknown", "decision_threshold": 0.5}
    return json.loads(METADATA_PATH.read_text())


def predict(payload: dict) -> dict:
    model = load_model()
    meta = load_metadata()
    threshold = float(meta.get("decision_threshold", 0.5))
    probability = float(model.predict_proba(pd.DataFrame([payload]))[:, 1][0])
    label = int(probability >= threshold)
    return {
        "prediction": label,
        "status": "ANOMALY" if label else "NORMAL",
        "anomaly_probability": round(probability, 6),
        "threshold": threshold,
        "model_version": str(meta.get("model_version", "unknown")),
    }
