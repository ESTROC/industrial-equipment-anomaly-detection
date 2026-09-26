from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "equipment_telemetry.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "best_pipeline.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", PROJECT_ROOT / "models" / "model_metadata.json"))
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

RANDOM_STATE = 42

RAW_FEATURES = [
    "rpm", "load_pct", "temperature_c", "vibration_mm_s", "pressure_bar",
    "voltage_v", "current_a", "lubrication_pct", "ambient_temp_c", "runtime_hours",
]
