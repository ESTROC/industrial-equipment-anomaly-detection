import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, METADATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE, RAW_FEATURES
from .data_generation import generate_dataset
from .features import EquipmentFeatureEngineer


def metric_dict(y_true, pred, prob):
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def make_pipeline(model):
    return Pipeline([
        ("features", EquipmentFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    X = df[RAW_FEATURES]
    y = df["label"].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=350, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    validation = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        validation[name] = metric_dict(y_val, pred, prob)

    best_name = max(validation, key=lambda n: (validation[n]["f1"], validation[n]["roc_auc"]))

    best_model = make_pipeline(candidates[best_name])
    best_model.fit(X_trainval, y_trainval)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = metric_dict(y_test, test_pred, test_prob)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "features": len(RAW_FEATURES),
            "anomaly_rate": round(float(y.mean()), 4),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "synthetic": True,
        },
        "validation_model_comparison": validation,
        "selected_model": best_name,
        "test_metrics": test_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    METADATA_PATH.write_text(json.dumps({
        "model_name": best_name,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_features": RAW_FEATURES,
        "decision_threshold": 0.5,
        "synthetic_training_data": True,
        "test_metrics": test_metrics,
    }, indent=2))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
