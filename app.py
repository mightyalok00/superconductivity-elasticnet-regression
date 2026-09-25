from pathlib import Path
from threading import Lock
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

APP_TITLE = "Superconductivity Critical Temperature API"
APP_VERSION = "1.0.0"
TARGET = "critical_temp"
RANDOM_STATE = 42
BEST_ALPHA = 0.0001
BEST_L1_RATIO = 0.9

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

model = None
model_lock = Lock()

# Read only the header at import time so Railway can start immediately.
# The full dataset is loaded and ElasticNet is fitted lazily on the first /predict call.
if not DATA_PATH.exists():
    raise RuntimeError(f"Dataset not found: {DATA_PATH}")

_header = pd.read_csv(DATA_PATH, nrows=0)
if TARGET not in _header.columns:
    raise RuntimeError(f"Target column '{TARGET}' not found in dataset.")

feature_names: List[str] = [col for col in _header.columns if col != TARGET]
training_rows = 21263


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing all 81 model features and numeric values.",
    )


class PredictionResponse(BaseModel):
    predicted_critical_temp_k: float
    model: str
    feature_count: int


def ensure_model() -> Pipeline:
    """Fit the model once, on demand, and reuse it for later predictions."""
    global model

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        df = pd.read_csv(DATA_PATH)
        X = df[feature_names]
        y = df[TARGET]

        fitted_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=BEST_ALPHA,
                        l1_ratio=BEST_L1_RATIO,
                        max_iter=30000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        fitted_model.fit(X, y)
        model = fitted_model

    return model


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Railway-ready FastAPI service for predicting superconducting critical "
        "temperature using the project's tuned ElasticNet regression model."
    )
)


@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Superconductivity ML API</title>
  <style>
    :root {
      --bg: #07111f;
      --card: rgba(13, 27, 46, 0.82);
      --text: #eef6ff;
      --muted: #a9bbcf;
      --accent: #5eead4;
      --accent2: #60a5fa;
      --border: rgba(255,255,255,.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 20%, rgba(96,165,250,.18), transparent 34%),
        radial-gradient(circle at 85% 15%, rgba(94,234,212,.15), transparent 30%),
        linear-gradient(140deg, #050b14 0%, var(--bg) 55%, #081729 100%);
      padding: 32px 18px;
    }
    .wrap { max-width: 1050px; margin: 0 auto; }
    .hero {
      border: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(13,27,46,.92), rgba(9,20,36,.86));
      backdrop-filter: blur(16px);
      border-radius: 28px;
      padding: 46px;
      box-shadow: 0 30px 70px rgba(0,0,0,.35);
    }
    .badge {
      display: inline-block;
      padding: 8px 13px;
      border-radius: 999px;
      background: rgba(94,234,212,.10);
      color: var(--accent);
      border: 1px solid rgba(94,234,212,.25);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    h1 {
      font-size: clamp(38px, 7vw, 72px);
      line-height: .98;
      margin: 20px 0 18px;
      letter-spacing: -.045em;
    }
    .gradient {
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      color: transparent;
    }
    p {
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
      max-width: 760px;
    }
    .actions { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 28px; }
    a.button {
      text-decoration: none;
      color: #06121f;
      font-weight: 800;
      background: linear-gradient(90deg, var(--accent), #93c5fd);
      padding: 12px 18px;
      border-radius: 12px;
    }
    a.secondary {
      color: var(--text);
      background: rgba(255,255,255,.05);
      border: 1px solid var(--border);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-top: 18px;
    }
    .card {
      padding: 22px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: var(--card);
    }
    .card strong { display: block; font-size: 24px; margin-bottom: 6px; }
    .card span { color: var(--muted); font-size: 14px; }
    code {
      color: #d8f7ff;
      background: rgba(255,255,255,.06);
      padding: 2px 7px;
      border-radius: 6px;
    }
    footer { color: #71859b; text-align: center; padding: 28px 0 4px; font-size: 13px; }
    @media(max-width: 760px) {
      .hero { padding: 28px 22px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <span class="badge">🧲 Railway-ready FastAPI</span>
      <h1>Superconductivity<br><span class="gradient">Prediction API</span></h1>
      <p>
        Predict superconducting critical temperature using a tuned ElasticNet model
        trained on 21,263 materials and 81 engineered material-property features.
        Built from the same analysis used in the portfolio project.
      </p>

      <div class="actions">
        <a class="button" href="/docs">🚀 Open Swagger API</a>
        <a class="button secondary" href="/model-info">🧠 Model Info</a>
        <a class="button secondary" href="/features">🧬 Feature List</a>
        <a class="button secondary" href="/health">💚 Health</a>
      </div>

      <div class="grid">
        <div class="card">
          <strong>21,263</strong>
          <span>training materials</span>
        </div>
        <div class="card">
          <strong>81</strong>
          <span>engineered predictors</span>
        </div>
        <div class="card">
          <strong>R² ≈ 0.737</strong>
          <span>validated ElasticNet test score</span>
        </div>
      </div>
    </section>

    <footer>
      Built with FastAPI • scikit-learn • Railway • Alok Agarwal
    </footer>
  </main>
</body>
</html>
"""


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "model_ready": model is not None,
        "model_loading": "lazy_on_first_prediction",
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    return {
        "model": "ElasticNet Regression",
        "alpha": BEST_ALPHA,
        "l1_ratio": BEST_L1_RATIO,
        "training_rows": training_rows,
        "feature_count": len(feature_names),
        "validated_test_metrics": {
            "mae_k": 13.210,
            "rmse_k": 17.387,
            "r2": 0.737,
        },
        "cv_metrics": {
            "rmse_mean_k": 17.699,
            "rmse_std_k": 0.257,
            "r2_mean": 0.734,
            "r2_std": 0.006,
        },
        "note": "The deployed model is refit on all available train.csv rows using the tuned hyperparameters.",
    }


@app.get("/features", tags=["Model"])
async def features():
    return {
        "count": len(feature_names),
        "features": feature_names,
    }


@app.get("/sample", tags=["Prediction"])
async def sample_payload():
    df = pd.read_csv(DATA_PATH, nrows=1)
    sample = df.drop(columns=[TARGET]).iloc[0].to_dict()
    return {"features": {k: float(v) for k, v in sample.items()}}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(payload: PredictionRequest):
    active_model = await run_in_threadpool(ensure_model)

    incoming = payload.features
    missing = [name for name in feature_names if name not in incoming]
    extra = [name for name in incoming if name not in feature_names]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature mismatch.",
                "missing_features": missing,
                "extra_features": extra,
                "expected_feature_count": len(feature_names),
            },
        )

    row = pd.DataFrame(
        [[incoming[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = float(active_model.predict(row)[0])

    return PredictionResponse(
        predicted_critical_temp_k=round(prediction, 4),
        model="ElasticNet Regression",
        feature_count=len(feature_names),
    )
