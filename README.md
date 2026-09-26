# Industrial Equipment Anomaly Detection

Production-oriented machine-learning system for detecting abnormal industrial equipment telemetry with **Python, Scikit-learn, FastAPI, Docker and CI**.

The repository covers the full applied ML lifecycle:

`synthetic telemetry -> feature engineering -> train/validation/test split -> model comparison -> held-out evaluation -> serialized pipeline -> API inference -> Docker -> automated tests`

> The included data is synthetic and contains no proprietary industrial data. Reported metrics demonstrate the engineering workflow, not real-world safety performance.

## What it predicts

Given a snapshot of equipment telemetry such as RPM, load, temperature, vibration, pressure, voltage, current, lubrication and runtime, the API estimates whether the state is **NORMAL** or **ANOMALY** and returns an anomaly probability.

Simulated failure modes include bearing degradation, overheating, electrical faults and lubrication faults.

## ML pipeline

The project engineers domain-inspired features including:

- temperature rise over ambient
- electrical power proxy
- load-to-RPM ratio
- vibration-load interaction
- thermal-load interaction
- pressure/load ratio
- lubrication stress
- log-transformed runtime

Three models are compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

Preprocessing and inference are stored in a **single Scikit-learn Pipeline**, preventing training/serving drift and data leakage.

## Verified held-out result

The reproducible run uses **12,000 synthetic telemetry rows** and a **2,400-row held-out test set**.

| Metric | Test result |
| --- | ---: |
| Accuracy | 92.50% |
| Precision | 82.81% |
| Recall | 73.61% |
| F1 | 77.94% |
| ROC-AUC | 93.88% |

Selected model: **Random Forest**

Full metrics: `reports/metrics.json`

## API

- `GET /health`
- `POST /predict`
- Swagger/OpenAPI at `/docs`
- Pydantic request validation
- probability-based inference
- model version metadata

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
python -m industrial_anomaly.data_generation
python -m industrial_anomaly.train
pytest -q
uvicorn industrial_anomaly.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker compose up --build
```

The multi-stage Docker build regenerates the synthetic dataset and trains the model, so generated CSV files and model binaries do not need to be committed to Git.

## Repository structure

```text
src/industrial_anomaly/
  api.py
  config.py
  data_generation.py
  features.py
  schemas.py
  service.py
  train.py
tests/
reports/metrics.json
.github/workflows/ci.yml
Dockerfile
docker-compose.yml
MODEL_CARD.md
```

## Production-oriented engineering

- reproducible synthetic data generation
- train-only preprocessing
- explicit validation and untouched test set
- automated model comparison
- typed API contracts
- fail-fast model loading
- Docker health check
- non-root container runtime
- automated tests
- GitHub Actions CI
- model card and reproducible metrics

## Limitations

This is an ML engineering demonstration. A real industrial deployment would additionally require real sensor data, temporal validation, calibration, drift/performance monitoring, authentication, observability, model registry, load testing, rollback strategy and domain safety validation.

## License

MIT License.