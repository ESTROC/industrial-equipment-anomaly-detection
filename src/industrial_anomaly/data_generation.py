import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    bounds = {
        "rpm": (400, 4200), "load_pct": (3, 100), "temperature_c": (35, 135),
        "vibration_mm_s": (0.2, 15), "pressure_bar": (0.8, 9), "voltage_v": (190, 260),
        "current_a": (1, 90), "lubrication_pct": (2, 100), "ambient_temp_c": (5, 50),
        "runtime_hours": (0, 25000),
    }
    for col, (lo, hi) in bounds.items():
        df[col] = df[col].clip(lo, hi)
    return df


def generate_dataset(n_samples: int = 12000, anomaly_rate: float = 0.18, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate noisy, overlapping synthetic telemetry with multiple failure mechanisms."""
    rng = np.random.default_rng(seed)
    n_anom = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anom

    ambient = rng.normal(27, 6, n_normal)
    load = np.clip(rng.beta(2.3, 2.0, n_normal) * 100, 5, 98)
    rpm = rng.normal(900 + load * 20, 220, n_normal)
    voltage = rng.normal(230, 5.5, n_normal)
    current = np.clip(4 + load * 0.43 + rng.normal(0, 4, n_normal), 1, None)
    temp = ambient + 24 + load * 0.43 + rng.normal(0, 5.5, n_normal)
    vibration = np.clip(0.8 + load * 0.025 + rng.normal(0, 0.65, n_normal), 0.2, None)
    pressure = 2.2 + load * 0.045 + rng.normal(0, 0.45, n_normal)
    lubrication = np.clip(88 - load * 0.08 + rng.normal(0, 6, n_normal), 25, 100)
    runtime = rng.gamma(2.2, 1700, n_normal)

    normal = pd.DataFrame({
        "rpm": rpm, "load_pct": load, "temperature_c": temp,
        "vibration_mm_s": vibration, "pressure_bar": pressure,
        "voltage_v": voltage, "current_a": current, "lubrication_pct": lubrication,
        "ambient_temp_c": ambient, "runtime_hours": runtime,
        "failure_mode": "normal", "label": 0,
    })

    idx = rng.integers(0, n_normal, n_anom)
    anomalous = normal.iloc[idx].copy().reset_index(drop=True)
    modes = rng.choice(["bearing", "overheat", "electrical", "lubrication"], n_anom,
                       p=[0.31, 0.28, 0.22, 0.19])
    anomalous["failure_mode"] = modes
    anomalous["label"] = 1

    for mode in ["bearing", "overheat", "electrical", "lubrication"]:
        mask = anomalous["failure_mode"].eq(mode)
        n = int(mask.sum())
        if not n:
            continue
        severity = rng.uniform(0.35, 1.0, n)

        if mode == "bearing":
            anomalous.loc[mask, "vibration_mm_s"] += severity * rng.normal(4.2, 1.2, n)
            anomalous.loc[mask, "temperature_c"] += severity * rng.normal(10, 4, n)
            anomalous.loc[mask, "current_a"] += severity * rng.normal(7, 3, n)
        elif mode == "overheat":
            anomalous.loc[mask, "temperature_c"] += severity * rng.normal(25, 7, n)
            anomalous.loc[mask, "pressure_bar"] += severity * rng.normal(0.8, 0.35, n)
            anomalous.loc[mask, "load_pct"] += severity * rng.normal(9, 6, n)
        elif mode == "electrical":
            direction = rng.choice([-1, 1], n)
            anomalous.loc[mask, "voltage_v"] += direction * severity * rng.normal(19, 6, n)
            anomalous.loc[mask, "current_a"] += severity * rng.normal(14, 5, n)
            anomalous.loc[mask, "temperature_c"] += severity * rng.normal(7, 3, n)
        else:
            anomalous.loc[mask, "lubrication_pct"] -= severity * rng.normal(38, 10, n)
            anomalous.loc[mask, "vibration_mm_s"] += severity * rng.normal(3.0, 1.0, n)
            anomalous.loc[mask, "temperature_c"] += severity * rng.normal(12, 4, n)

    anomalous = _clip(anomalous)
    data = pd.concat([normal, anomalous], ignore_index=True)

    sensor_cols = [c for c in data.columns if c not in {"label", "failure_mode"}]
    missing_mask = rng.random((len(data), len(sensor_cols))) < 0.004
    for j, col in enumerate(sensor_cols):
        data.loc[missing_mask[:, j], col] = np.nan

    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")
    print(f"Anomaly rate: {df['label'].mean():.2%}")


if __name__ == "__main__":
    main()
