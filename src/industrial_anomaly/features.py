import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EquipmentFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic domain-inspired feature engineering inside the sklearn Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        eps = 1e-6
        frame["temp_rise_c"] = frame["temperature_c"] - frame["ambient_temp_c"]
        frame["power_proxy"] = frame["voltage_v"] * frame["current_a"]
        frame["load_per_rpm"] = frame["load_pct"] / (frame["rpm"] + eps)
        frame["vibration_load"] = frame["vibration_mm_s"] * (1.0 + frame["load_pct"] / 100.0)
        frame["thermal_load"] = frame["temperature_c"] * (1.0 + frame["load_pct"] / 100.0)
        frame["pressure_load_ratio"] = frame["pressure_bar"] / (frame["load_pct"] + 1.0)
        frame["lubrication_stress"] = (100.0 - frame["lubrication_pct"]) * frame["vibration_mm_s"]
        frame["runtime_log"] = np.log1p(frame["runtime_hours"].clip(lower=0))
        return frame
