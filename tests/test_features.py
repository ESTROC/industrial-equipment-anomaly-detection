import pandas as pd

from industrial_anomaly.features import EquipmentFeatureEngineer


def test_feature_engineer_adds_features():
    X = pd.DataFrame([{
        "rpm": 2000, "load_pct": 50, "temperature_c": 80,
        "vibration_mm_s": 2, "pressure_bar": 4, "voltage_v": 230,
        "current_a": 25, "lubrication_pct": 80, "ambient_temp_c": 25,
        "runtime_hours": 1000,
    }])
    out = EquipmentFeatureEngineer().fit_transform(X)
    assert "temp_rise_c" in out.columns
    assert "power_proxy" in out.columns
    assert out.loc[0, "temp_rise_c"] == 55
