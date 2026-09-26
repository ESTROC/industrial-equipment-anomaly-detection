# Model Card

## Intended use
Demonstration binary anomaly classification for synthetic industrial equipment telemetry.

## Data
- Rows: 12,000
- Raw features: 10
- Anomaly prevalence: 18.0%
- Held-out test rows: 2,400
- Source: reproducible synthetic generator in this repository

## Selected model
Random Forest

## Held-out test metrics
- Accuracy: 0.9250
- Precision: 0.8281
- Recall: 0.7361
- F1: 0.7794
- ROC-AUC: 0.9388

## Limitations
The dataset is synthetic and snapshot-based. This model is not validated for safety-critical industrial decisions. Real deployment requires real telemetry, temporal validation, drift monitoring, security controls, observability and domain validation.
