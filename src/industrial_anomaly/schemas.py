from pydantic import BaseModel, Field


class TelemetryRequest(BaseModel):
    rpm: float = Field(..., ge=0, le=6000)
    load_pct: float = Field(..., ge=0, le=100)
    temperature_c: float = Field(..., ge=-20, le=180)
    vibration_mm_s: float = Field(..., ge=0, le=30)
    pressure_bar: float = Field(..., ge=0, le=15)
    voltage_v: float = Field(..., ge=0, le=400)
    current_a: float = Field(..., ge=0, le=150)
    lubrication_pct: float = Field(..., ge=0, le=100)
    ambient_temp_c: float = Field(..., ge=-40, le=80)
    runtime_hours: float = Field(..., ge=0, le=100000)


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    anomaly_probability: float
    threshold: float
    model_version: str
