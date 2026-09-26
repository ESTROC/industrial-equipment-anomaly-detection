from fastapi.testclient import TestClient

from industrial_anomaly.api import app


def test_health_and_prediction():
    payload = {
        "rpm": 2100, "load_pct": 52, "temperature_c": 78,
        "vibration_mm_s": 1.8, "pressure_bar": 4.4, "voltage_v": 231,
        "current_a": 27, "lubrication_pct": 83, "ambient_temp_c": 26,
        "runtime_hours": 2800,
    }
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0 <= response.json()["anomaly_probability"] <= 1

        invalid = {**payload, "load_pct": 140}
        assert client.post("/predict", json=invalid).status_code == 422
