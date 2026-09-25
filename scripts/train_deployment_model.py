from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "elasticnet_v1.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

TARGET = "critical_temp"
RANDOM_STATE = 42
ALPHA = 0.0001
L1_RATIO = 0.9
MODEL_VERSION = "1.0.0"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET]

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", ElasticNet(
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        max_iter=30000,
        random_state=RANDOM_STATE,
    )),
])

pipeline.fit(X, y)
joblib.dump(pipeline, MODEL_PATH)

stats = {}
for col in X.columns:
    s = X[col]
    stats[col] = {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
    }

metadata = {
    "model_name": "ElasticNet Regression",
    "model_version": MODEL_VERSION,
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "training_rows": int(len(df)),
    "feature_count": int(X.shape[1]),
    "target": TARGET,
    "alpha": ALPHA,
    "l1_ratio": L1_RATIO,
    "random_state": RANDOM_STATE,
    "features": list(X.columns),
    "feature_stats": stats,
    "validated_test_metrics": {
        "mae_k": 13.210,
        "rmse_k": 17.387,
        "r2": 0.737
    },
    "cv_metrics": {
        "rmse_mean_k": 17.699,
        "rmse_std_k": 0.257,
        "r2_mean": 0.734,
        "r2_std": 0.006
    }
}

META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved model: {MODEL_PATH}")
print(f"Saved metadata: {META_PATH}")
