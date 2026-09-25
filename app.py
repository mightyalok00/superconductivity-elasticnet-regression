from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
import io
import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_TITLE = "Superconductivity Critical Temperature API"
APP_VERSION = "1.1.0"
MODEL_VERSION = "1.0.0"
TARGET = "critical_temp"
RANDOM_STATE = 42
BEST_ALPHA = 0.0001
BEST_L1_RATIO = 0.9
STARTED_AT = time.time()

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "elasticnet_v1.joblib"
META_PATH = ROOT / "outputs" / "models" / "model_metadata.json"
FIGURE_DIR = ROOT / "outputs" / "figures"

model = None
model_lock = Lock()

if not DATA_PATH.exists():
    raise RuntimeError(f"Dataset not found: {DATA_PATH}")

metadata = {}
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

_header = pd.read_csv(DATA_PATH, nrows=0)
if TARGET not in _header.columns:
    raise RuntimeError(f"Target column '{TARGET}' not found in dataset.")

feature_names: List[str] = metadata.get(
    "features", [col for col in _header.columns if col != TARGET]
)
feature_stats = metadata.get("feature_stats", {})
training_rows = int(metadata.get("training_rows", 21263))


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing all 81 model features and numeric values.",
    )


class PredictionResponse(BaseModel):
    predicted_critical_temp_k: float
    model: str
    model_version: str
    feature_count: int
    expected_rmse_k: float
    warnings: List[str]


def _fit_fallback_model() -> Pipeline:
    df = pd.read_csv(DATA_PATH)
    X = df[feature_names]
    y = df[TARGET]
    fitted = Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNet(
            alpha=BEST_ALPHA,
            l1_ratio=BEST_L1_RATIO,
            max_iter=30000,
            random_state=RANDOM_STATE,
        )),
    ])
    fitted.fit(X, y)
    return fitted


def ensure_model() -> Pipeline:
    global model
    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            model = _fit_fallback_model()

    return model


def validate_feature_payload(incoming: Dict[str, float]) -> None:
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


def range_warnings(incoming: Dict[str, float]) -> List[str]:
    warnings = []
    for name, value in incoming.items():
        stats = feature_stats.get(name)
        if not stats:
            continue
        min_v = stats.get("min")
        max_v = stats.get("max")
        if min_v is not None and value < min_v:
            warnings.append(f"{name} is below the training minimum ({min_v:.4g}).")
        elif max_v is not None and value > max_v:
            warnings.append(f"{name} is above the training maximum ({max_v:.4g}).")
    return warnings


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Production-style FastAPI service for predicting superconducting critical "
        "temperature using the project's tuned ElasticNet regression pipeline."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if FIGURE_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURE_DIR)), name="figures")


@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Interactive superconductivity critical temperature prediction API built with FastAPI and ElasticNet." />
  <meta property="og:title" content="Superconductivity ML API" />
  <meta property="og:description" content="Predict critical temperature using an ElasticNet model trained on 21,263 superconducting materials." />
  <title>Superconductivity ML API</title>
  <style>
    :root {
      --bg:#07111f; --panel:#0e1d31; --panel2:#10243c; --text:#eef6ff;
      --muted:#9fb2c8; --accent:#5eead4; --accent2:#60a5fa; --border:rgba(255,255,255,.10);
      --good:#4ade80; --warn:#fbbf24;
    }
    *{box-sizing:border-box} body{margin:0;background:
      radial-gradient(circle at 12% 8%,rgba(96,165,250,.18),transparent 27%),
      radial-gradient(circle at 88% 12%,rgba(94,234,212,.14),transparent 28%),
      linear-gradient(145deg,#050a12,var(--bg) 55%,#09182a);color:var(--text);
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .wrap{max-width:1160px;margin:auto;padding:28px 18px 50px}
    .hero,.section{border:1px solid var(--border);background:rgba(10,23,40,.86);border-radius:24px;
      box-shadow:0 28px 65px rgba(0,0,0,.28)}
    .hero{padding:44px}.section{padding:28px;margin-top:20px}
    .eyebrow{display:flex;align-items:center;gap:8px;color:var(--accent);font-weight:800;font-size:13px;text-transform:uppercase;letter-spacing:.08em}
    .dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 18px var(--good)}
    h1{font-size:clamp(40px,7vw,72px);line-height:.98;letter-spacing:-.045em;margin:18px 0}
    h2{margin:0 0 14px;font-size:26px} h3{margin:0 0 8px}
    .gradient{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;color:transparent}
    p{color:var(--muted);line-height:1.7}.lead{font-size:18px;max-width:820px}
    .buttons{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}
    .btn{border:0;border-radius:12px;padding:12px 16px;font-weight:800;cursor:pointer;text-decoration:none;display:inline-block}
    .primary{background:linear-gradient(90deg,var(--accent),#93c5fd);color:#06121f}
    .secondary{background:rgba(255,255,255,.055);border:1px solid var(--border);color:var(--text)}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:24px}
    .metric,.card{background:linear-gradient(145deg,rgba(17,38,63,.95),rgba(11,27,47,.95));border:1px solid var(--border);border-radius:16px;padding:18px}
    .metric b{font-size:26px;display:block}.metric span,.small{font-size:13px;color:var(--muted)}
    .grid2{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}
    textarea{width:100%;min-height:280px;background:#07111f;color:#dff7ff;border:1px solid var(--border);border-radius:14px;padding:14px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
    input[type=file]{width:100%;padding:14px;border:1px dashed rgba(255,255,255,.22);border-radius:12px;color:var(--muted);background:#081626}
    .result{margin-top:14px;padding:18px;border-radius:14px;background:rgba(94,234,212,.08);border:1px solid rgba(94,234,212,.25);display:none}
    .result strong{font-size:30px;color:var(--accent)}
    .warn{color:var(--warn);font-size:13px;margin-top:8px}
    .charts{display:grid;grid-template-columns:1fr 1fr;gap:15px}.charts img{width:100%;border-radius:14px;border:1px solid var(--border);background:white}
    .full{grid-column:1/-1}
    code{background:rgba(255,255,255,.07);padding:2px 6px;border-radius:6px;color:#d8f7ff}
    footer{text-align:center;color:#71859b;padding:26px 0 0;font-size:13px}
    @media(max-width:850px){.metrics{grid-template-columns:1fr 1fr}.grid2,.charts{grid-template-columns:1fr}.full{grid-column:auto}.hero{padding:28px}.section{padding:22px}}
    @media(max-width:520px){.metrics{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="eyebrow"><span class="dot"></span> Live FastAPI deployment on Railway</div>
      <h1>Superconductivity<br><span class="gradient">Critical Temperature Predictor</span></h1>
      <p class="lead">A deployable machine-learning application that estimates superconducting critical temperature from 81 engineered material-property features using a tuned ElasticNet pipeline.</p>
      <div class="buttons">
        <a class="btn primary" href="#predictor">Try Live Prediction</a>
        <a class="btn secondary" href="/docs">Swagger API</a>
        <a class="btn secondary" href="https://github.com/mightyalok00/superconductivity-elasticnet-regression">GitHub Repository</a>
      </div>
      <div class="metrics">
        <div class="metric"><b>21,263</b><span>training materials</span></div>
        <div class="metric"><b>81</b><span>engineered predictors</span></div>
        <div class="metric"><b>0.737</b><span>ElasticNet test R²</span></div>
        <div class="metric"><b>17.387 K</b><span>ElasticNet test RMSE</span></div>
      </div>
    </section>

    <section class="section" id="predictor">
      <h2>Interactive Prediction</h2>
      <p>Load a real sample from the dataset, edit any values, and send it directly to the versioned API.</p>
      <div class="grid2">
        <div>
          <textarea id="payload" aria-label="Prediction JSON payload">{}</textarea>
          <div class="buttons">
            <button class="btn secondary" onclick="loadSample()">Load Sample</button>
            <button class="btn primary" onclick="predict()">Predict Critical Temperature</button>
          </div>
          <div class="result" id="result"></div>
        </div>
        <div class="card">
          <h3>How it works</h3>
          <p><b>Input</b> → validation → training-range check → StandardScaler → ElasticNet → predicted critical temperature.</p>
          <p class="small">Inputs outside the training range are accepted but returned with an out-of-distribution warning.</p>
          <hr style="border-color:var(--border);border-width:1px 0 0">
          <h3>Model version</h3>
          <p><code>ElasticNet v1.0.0</code><br>alpha = 0.0001<br>l1_ratio = 0.9</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Batch CSV Prediction</h2>
      <p>Upload a CSV containing the same 81 feature columns. The API returns a downloadable CSV with predictions and out-of-distribution warning counts.</p>
      <input id="csvFile" type="file" accept=".csv,text/csv">
      <div class="buttons"><button class="btn primary" onclick="uploadCsv()">Predict CSV</button></div>
      <div class="small" id="batchStatus"></div>
    </section>

    <section class="section">
      <h2>Model Evaluation</h2>
      <div class="charts">
        <img src="/figures/q10_model_comparison.png" alt="Model comparison chart">
        <img src="/figures/q14_residuals.png" alt="ElasticNet residual distribution">
        <img class="full" src="/figures/q14_actual_vs_predicted.png" alt="Actual versus predicted critical temperature">
      </div>
    </section>

    <section class="section">
      <h2>API Resources</h2>
      <div class="grid2">
        <div class="card"><h3>For developers</h3><p><code>/docs</code> Swagger UI<br><code>/api/v1/predict</code> single prediction<br><code>/api/v1/predict-batch</code> CSV batch prediction<br><code>/api/v1/schema</code> feature ranges and metadata</p></div>
        <div class="card"><h3>System</h3><p><code>/health</code> Railway health check<br><code>/model-info</code> validated model metrics<br><code>/features</code> expected feature names<br><code>/sample</code> example request payload</p></div>
      </div>
    </section>

    <footer>Built by Alok Agarwal • FastAPI • scikit-learn • Railway • MIT License</footer>
  </main>
<script>
async function loadSample(){
  const r=await fetch('/api/v1/sample');
  const data=await r.json();
  document.getElementById('payload').value=JSON.stringify(data,null,2);
}
async function predict(){
  const box=document.getElementById('result');
  box.style.display='block';
  box.innerHTML='Running prediction…';
  try{
    const body=JSON.parse(document.getElementById('payload').value);
    const r=await fetch('/api/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await r.json();
    if(!r.ok){box.innerHTML='<b>Request error</b><pre>'+JSON.stringify(data,null,2)+'</pre>';return;}
    const warningHtml=data.warnings.length?'<div class="warn">'+data.warnings.slice(0,6).join('<br>')+'</div>':'';
    box.innerHTML='<div class="small">Predicted critical temperature</div><strong>'+data.predicted_critical_temp_k.toFixed(2)+' K</strong><div class="small">Expected model RMSE ≈ '+data.expected_rmse_k+' K</div>'+warningHtml;
  }catch(e){box.innerHTML='<b>Invalid JSON</b><div class="warn">'+e.message+'</div>';}
}
async function uploadCsv(){
  const f=document.getElementById('csvFile').files[0];
  const status=document.getElementById('batchStatus');
  if(!f){status.textContent='Choose a CSV file first.';return;}
  status.textContent='Uploading and predicting…';
  const form=new FormData(); form.append('file',f);
  const r=await fetch('/api/v1/predict-batch',{method:'POST',body:form});
  if(!r.ok){status.textContent='Batch request failed: '+await r.text();return;}
  const blob=await r.blob();
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download='superconductivity_predictions.csv'; a.click();
  URL.revokeObjectURL(url);
  status.textContent='Prediction file downloaded.';
}
loadSample();
</script>
</body>
</html>
"""


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "api_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "artifact_available": MODEL_PATH.exists(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.get("/api/v1/model-info", tags=["Model"])
@app.get("/model-info", tags=["Model"], include_in_schema=False)
async def model_info():
    return {
        "model": "ElasticNet Regression",
        "model_version": MODEL_VERSION,
        "alpha": BEST_ALPHA,
        "l1_ratio": BEST_L1_RATIO,
        "training_rows": training_rows,
        "feature_count": len(feature_names),
        "artifact_available": MODEL_PATH.exists(),
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
    }


@app.get("/api/v1/features", tags=["Model"])
@app.get("/features", tags=["Model"], include_in_schema=False)
async def features():
    return {"count": len(feature_names), "features": feature_names}


@app.get("/api/v1/schema", tags=["Model"])
async def schema():
    return {
        "target": TARGET,
        "feature_count": len(feature_names),
        "features": feature_stats or {name: {} for name in feature_names},
    }


@app.get("/api/v1/sample", tags=["Prediction"])
@app.get("/sample", tags=["Prediction"], include_in_schema=False)
async def sample_payload():
    df = pd.read_csv(DATA_PATH, nrows=1)
    sample = df.drop(columns=[TARGET]).iloc[0].to_dict()
    return {"features": {k: float(v) for k, v in sample.items()}}


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], include_in_schema=False)
async def predict(payload: PredictionRequest):
    incoming = payload.features
    validate_feature_payload(incoming)
    warnings = range_warnings(incoming)

    active_model = await run_in_threadpool(ensure_model)
    row = pd.DataFrame([[incoming[name] for name in feature_names]], columns=feature_names)
    prediction = float(active_model.predict(row)[0])

    return PredictionResponse(
        predicted_critical_temp_k=round(prediction, 4),
        model="ElasticNet Regression",
        model_version=MODEL_VERSION,
        feature_count=len(feature_names),
        expected_rmse_k=17.387,
        warnings=warnings,
    )


@app.post("/api/v1/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = [name for name in feature_names if name not in df.columns]
    extra = [name for name in df.columns if name not in feature_names]
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV feature mismatch.",
                "missing_features": missing,
                "extra_features": extra,
                "expected_feature_count": len(feature_names),
            },
        )

    active_model = await run_in_threadpool(ensure_model)
    ordered = df[feature_names].copy()
    predictions = active_model.predict(ordered)

    warning_counts = []
    for _, row in ordered.iterrows():
        warning_counts.append(len(range_warnings({k: float(row[k]) for k in feature_names})))

    result = ordered.copy()
    result["predicted_critical_temp_k"] = predictions
    result["ood_warning_count"] = warning_counts

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    response = StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = 'attachment; filename="superconductivity_predictions.csv"'
    return response
