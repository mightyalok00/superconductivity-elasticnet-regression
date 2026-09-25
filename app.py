from pathlib import Path
from threading import Lock
from typing import Dict, List
import io
import json
import logging
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings

APP_TITLE = settings.app_name
APP_VERSION = "1.2.0"
MODEL_VERSION = "1.0.0"
TARGET = settings.target
RANDOM_STATE = settings.random_state
BEST_ALPHA = 0.0001
BEST_L1_RATIO = 0.9
STARTED_AT = time.time()

ROOT = Path(__file__).resolve().parent
DATA_PATH = settings.train_path
MODEL_PATH = settings.model_path
META_PATH = settings.metadata_path
FIGURE_DIR = ROOT / "outputs" / "figures"

model = None
model_lock = Lock()
REQUEST_COUNT = 0
PREDICTION_COUNT = 0
RATE_BUCKETS: dict[tuple[str, int], int] = {}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("superconductivity_api")

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


class CompareRequest(BaseModel):
    material_a: Dict[str, float]
    material_b: Dict[str, float]


class PredictionResponse(BaseModel):
    predicted_critical_temp_k: float
    model: str
    model_version: str
    feature_count: int
    expected_rmse_k: float
    warnings: List[str]
    ood_risk_score: float
    top_contributors: List[dict]
    response_time_ms: float
    request_id: str


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


def feature_group(name: str) -> str:
    lowered = name.lower()
    groups = [
        ("atomic_mass", "Atomic Mass"),
        ("fie", "First Ionization Energy"),
        ("atomic_radius", "Atomic Radius"),
        ("density", "Density"),
        ("electronaffinity", "Electron Affinity"),
        ("fusionheat", "Fusion Heat"),
        ("thermalconductivity", "Thermal Conductivity"),
        ("valence", "Valence"),
    ]
    if lowered == "number_of_elements":
        return "Composition"
    compact = lowered.replace("_", "")
    for key, label in groups:
        if key.replace("_", "") in compact:
            return label
    return "Other"


def feature_description(name: str) -> str:
    prefix = "Weighted " if name.startswith("wtd_") else ""
    cleaned = name.replace("wtd_", "")
    statistic = ""
    for key, label in [
        ("mean_", "mean"),
        ("gmean_", "geometric mean"),
        ("entropy_", "entropy"),
        ("range_", "range"),
        ("std_", "standard deviation"),
    ]:
        if cleaned.startswith(key):
            statistic = label
            cleaned = cleaned[len(key):]
            break
    property_name = cleaned.replace("_", " ")
    if name == "number_of_elements":
        return "Number of chemical elements represented in the material."
    return f"{prefix}{statistic} summary of {property_name}".strip().capitalize() + "."


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


def ood_score(incoming: Dict[str, float]) -> float:
    checked = 0
    violations = 0
    for name, value in incoming.items():
        stats = feature_stats.get(name)
        if not stats:
            continue
        checked += 1
        if value < stats.get("min", value) or value > stats.get("max", value):
            violations += 1
    return round((violations / checked) * 100, 2) if checked else 0.0


def explain_row(active_model: Pipeline, incoming: Dict[str, float], limit: int = 10) -> List[dict]:
    scaler = active_model.named_steps["scale"]
    estimator = active_model.named_steps["model"]
    row = pd.DataFrame([[incoming[name] for name in feature_names]], columns=feature_names)
    standardized = scaler.transform(row)[0]
    contributions = standardized * estimator.coef_
    items = [
        {
            "feature": name,
            "group": feature_group(name),
            "contribution_k": round(float(value), 4),
            "direction": "higher" if value >= 0 else "lower",
        }
        for name, value in zip(feature_names, contributions)
    ]
    return sorted(items, key=lambda item: abs(item["contribution_k"]), reverse=True)[:limit]


def predict_one(active_model: Pipeline, incoming: Dict[str, float]) -> dict:
    validate_feature_payload(incoming)
    row = pd.DataFrame([[incoming[name] for name in feature_names]], columns=feature_names)
    prediction = float(active_model.predict(row)[0])
    warnings = range_warnings(incoming)
    return {
        "predicted_critical_temp_k": round(prediction, 4),
        "warnings": warnings,
        "ood_risk_score": ood_score(incoming),
        "top_contributors": explain_row(active_model, incoming, 10),
    }


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Production-style FastAPI service for predicting superconducting critical "
        "temperature using the project's tuned ElasticNet regression pipeline."
    ),
)

cors_origins = (
    ["*"]
    if settings.cors_origins.strip() == "*"
    else [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if FIGURE_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURE_DIR)), name="figures")


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    if request.url.path.startswith("/api/v1/"):
        if settings.require_api_key:
            supplied_key = request.headers.get("X-API-Key")
            if not settings.api_key or supplied_key != settings.api_key:
                return JSONResponse(
                    {"detail": "A valid X-API-Key header is required."},
                    status_code=401,
                    headers={"X-Request-ID": request_id},
                )

        if settings.enable_rate_limit:
            client_ip = request.client.host if request.client else "unknown"
            minute_bucket = int(time.time() // 60)
            bucket_key = (client_ip, minute_bucket)
            current = RATE_BUCKETS.get(bucket_key, 0) + 1
            RATE_BUCKETS[bucket_key] = current

            if current > settings.requests_per_minute:
                return JSONResponse(
                    {"detail": "Rate limit exceeded. Try again shortly."},
                    status_code=429,
                    headers={
                        "X-Request-ID": request_id,
                        "Retry-After": "60",
                    },
                )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(json.dumps({
            "event": "request_error",
            "request_id": request_id,
            "path": request.url.path,
        }))
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    logger.info(json.dumps({
        "event": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(elapsed_ms, 2),
    }))
    return response


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(
            "<html><body style='font-family:system-ui;background:#07111f;color:#eef6ff;padding:50px'>"
            "<h1>404</h1><p>That page does not exist.</p><a style='color:#5eead4' href='/'>Return home</a>"
            "</body></html>",
            status_code=404,
        )
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
async def robots():
    return "User-agent: *\nAllow: /\n"


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <rect width="64" height="64" rx="14" fill="#07111f"/>
    <circle cx="32" cy="32" r="20" fill="none" stroke="#5eead4" stroke-width="5"/>
    <path d="M18 34h28M24 22l16 20M40 22L24 42" stroke="#60a5fa" stroke-width="4" stroke-linecap="round"/>
    </svg>"""
    return Response(svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def home():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Interactive superconductivity critical temperature prediction API built with FastAPI and ElasticNet." />
  <meta property="og:title" content="Superconductivity ML API" />
  <meta property="og:description" content="Predict critical temperature using an ElasticNet model trained on 21,263 superconducting materials." />
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <title>Superconductivity ML API</title>
  <style>
    :root{--bg:#07111f;--panel:#0e1d31;--panel2:#10243c;--text:#eef6ff;--muted:#9fb2c8;--accent:#5eead4;--accent2:#60a5fa;--border:rgba(255,255,255,.10);--good:#4ade80;--warn:#fbbf24;--danger:#fb7185}
    body.light{--bg:#eef4fb;--panel:#fff;--panel2:#f7fbff;--text:#102033;--muted:#52677f;--border:rgba(15,35,60,.13)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 12% 8%,rgba(96,165,250,.18),transparent 27%),radial-gradient(circle at 88% 12%,rgba(94,234,212,.14),transparent 28%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;transition:.25s}
    .wrap{max-width:1180px;margin:auto;padding:28px 18px 50px}.hero,.section{border:1px solid var(--border);background:color-mix(in srgb,var(--panel) 92%,transparent);border-radius:24px;box-shadow:0 28px 65px rgba(0,0,0,.18)}.hero{padding:44px}.section{padding:28px;margin-top:20px}
    .topbar{display:flex;justify-content:space-between;align-items:center;gap:10px}.eyebrow{display:flex;align-items:center;gap:8px;color:var(--accent);font-weight:800;font-size:13px;text-transform:uppercase;letter-spacing:.08em}.dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 18px var(--good)}
    h1{font-size:clamp(40px,7vw,72px);line-height:.98;letter-spacing:-.045em;margin:18px 0}h2{margin:0 0 14px;font-size:26px}h3{margin:0 0 8px}.gradient{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;color:transparent}p{color:var(--muted);line-height:1.7}.lead{font-size:18px;max-width:840px}
    .buttons{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.btn{border:0;border-radius:12px;padding:12px 16px;font-weight:800;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:7px;transition:.18s}.btn:hover{transform:translateY(-2px);filter:brightness(1.06)}.btn:disabled{opacity:.55;cursor:not-allowed;transform:none}.primary{background:linear-gradient(90deg,var(--accent),#93c5fd);color:#06121f}.secondary{background:rgba(255,255,255,.055);border:1px solid var(--border);color:var(--text)}.danger{background:rgba(251,113,133,.12);color:#fecdd3;border:1px solid rgba(251,113,133,.22)}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:24px}.metric,.card{background:var(--panel2);border:1px solid var(--border);border-radius:16px;padding:18px}.metric b{font-size:26px;display:block}.metric span,.small{font-size:13px;color:var(--muted)}
    .grid2{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}textarea{width:100%;min-height:280px;background:color-mix(in srgb,var(--bg) 88%,black);color:#dff7ff;border:1px solid var(--border);border-radius:14px;padding:14px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}body.light textarea{color:#102033;background:white}
    input[type=file],input[type=number]{width:100%;padding:12px;border:1px solid var(--border);border-radius:12px;color:var(--text);background:var(--panel2)}.result{margin-top:14px;padding:18px;border-radius:14px;background:rgba(94,234,212,.08);border:1px solid rgba(94,234,212,.25);display:none}.result strong{font-size:30px;color:var(--accent)}.warn{color:var(--warn);font-size:13px;margin-top:8px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:15px}.charts img{width:100%;border-radius:14px;border:1px solid var(--border);background:white}.full{grid-column:1/-1}
    code{background:rgba(255,255,255,.07);padding:2px 6px;border-radius:6px;color:var(--accent)}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid var(--border)}th{color:var(--accent)}.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(96,165,250,.12);font-size:12px;color:#bfdbfe}.bar{height:8px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden}.bar>span{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
    details{border:1px solid var(--border);border-radius:12px;padding:12px 14px;background:var(--panel2);margin:8px 0}summary{cursor:pointer;font-weight:800}.feature-row{display:grid;grid-template-columns:1.3fr .7fr .7fr .7fr;gap:8px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px}.history{max-height:300px;overflow:auto}.socials{display:flex;gap:9px;flex-wrap:wrap}footer{text-align:center;color:#71859b;padding:26px 0 0;font-size:13px}
    @media(max-width:850px){.metrics,.grid3{grid-template-columns:1fr 1fr}.grid2,.charts{grid-template-columns:1fr}.full{grid-column:auto}.hero{padding:28px}.section{padding:22px}.feature-row{grid-template-columns:1fr 1fr}}
    @media(max-width:520px){.metrics,.grid3{grid-template-columns:1fr}.topbar{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="topbar">
        <div class="eyebrow"><span class="dot"></span> Live FastAPI deployment on Railway</div>
        <button class="btn secondary" onclick="toggleTheme()">☀ / ☾ Theme</button>
      </div>
      <h1>Superconductivity<br><span class="gradient">Critical Temperature Predictor</span></h1>
      <p class="lead">Interactive ML application for predicting superconducting critical temperature, explaining the strongest feature contributions, comparing materials, and ranking candidate materials from CSV uploads.</p>
      <div class="buttons">
        <a class="btn primary" href="#predictor">▶ Try Prediction</a>
        <a class="btn secondary" href="#compare">⇄ Compare Materials</a>
        <a class="btn secondary" href="/docs">API Docs</a>
        <a class="btn secondary" href="https://github.com/mightyalok00/superconductivity-elasticnet-regression">GitHub</a>
        <a class="btn secondary" href="https://www.linkedin.com/in/alok-agarwal-seo-ai-ml/">LinkedIn</a>
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
      <p>Load a real material sample, edit values if needed, predict, and inspect the most influential feature contributions.</p>
      <div class="grid2">
        <div>
          <textarea id="payload" aria-label="Prediction JSON payload">{}</textarea>
          <div class="buttons">
            <button class="btn secondary" onclick="loadSample()">Load Sample</button>
            <button class="btn primary" id="predictBtn" onclick="predict()">Predict</button>
            <button class="btn secondary" onclick="resetPrediction()">Reset</button>
            <button class="btn secondary" onclick="downloadSampleJson()">Download JSON</button>
            <button class="btn secondary" onclick="copyEndpoint('/api/v1/predict')">Copy Endpoint</button>
          </div>
          <div class="result" id="result"></div>
        </div>
        <div class="card">
          <h3>Top feature contributions</h3>
          <p class="small">These are model-based linear contributions after standardization. They explain the model output, not physical causation.</p>
          <div id="contributors"><span class="small">Run a prediction to see contributors.</span></div>
        </div>
      </div>
    </section>

    <section class="section" id="historySection">
      <h2>Prediction History</h2>
      <p>Stored only in this browser session.</p>
      <div class="buttons"><button class="btn danger" onclick="clearHistory()">Clear History</button></div>
      <div class="history"><table><thead><tr><th>Time</th><th>Prediction</th><th>OOD Risk</th><th>Latency</th><th>Request ID</th></tr></thead><tbody id="historyBody"></tbody></table></div>
    </section>

    <section class="section" id="compare">
      <h2>Compare Two Materials</h2>
      <p>Compare two complete feature payloads side-by-side using the same deployed model.</p>
      <div class="grid2">
        <div><h3>Material A</h3><textarea id="compareA">{}</textarea></div>
        <div><h3>Material B</h3><textarea id="compareB">{}</textarea></div>
      </div>
      <div class="buttons">
        <button class="btn secondary" onclick="loadCompareSamples()">Load Samples</button>
        <button class="btn primary" onclick="compareMaterials()">Compare</button>
      </div>
      <div class="result" id="compareResult"></div>
    </section>

    <section class="section">
      <h2>Batch Candidate Ranking</h2>
      <p>Upload a CSV with all 81 feature columns. Results are ranked from highest to lowest predicted critical temperature. You can optionally filter candidates below a threshold.</p>
      <div class="grid2">
        <div><input id="csvFile" type="file" accept=".csv,text/csv"></div>
        <div><input id="minTemp" type="number" value="80" step="1" placeholder="Minimum predicted temperature (K)"></div>
      </div>
      <div class="buttons">
        <button class="btn secondary" onclick="downloadSampleCsv()">Download Sample CSV</button>
        <button class="btn primary" id="batchBtn" onclick="uploadCsv()">Run Batch Prediction</button>
      </div>
      <div class="small" id="batchStatus"></div>
    </section>

    <section class="section">
      <h2>Feature Reference</h2>
      <p>Grouped feature definitions with training minimum, median and maximum values.</p>
      <div id="featureGroups"><span class="small">Loading feature schema…</span></div>
    </section>

    <section class="section">
      <h2>What the Model Learned</h2>
      <div class="grid3">
        <div class="card"><h3>Strongest positive signal</h3><p><code>wtd_std_ThermalConductivity</code><br>Correlation ≈ <b>+0.721</b></p></div>
        <div class="card"><h3>Strongest negative signal</h3><p><code>wtd_mean_Valence</code><br>Correlation ≈ <b>-0.632</b></p></div>
        <div class="card"><h3>Composition experiment</h3><p>Adding elemental composition improved RMSE by <b>0.876 K</b> and R² by <b>+0.0258</b>.</p></div>
      </div>
      <div class="card" style="margin-top:14px">
        <h3>Top 10 globally important ElasticNet features</h3>
        <div id="topFeatures"><span class="small">Loading…</span></div>
      </div>
    </section>

    <section class="section">
      <h2>Model Comparison</h2>
      <div class="grid3">
        <div class="card"><h3>Linear Regression</h3><p><b>RMSE 17.378 K</b><br>R² 0.738<br><span class="pill">Best test RMSE</span></p></div>
        <div class="card"><h3>Ridge</h3><p>Best α = 0.1<br>Performance nearly identical to Linear.</p></div>
        <div class="card"><h3>Lasso</h3><p>α = 0.0001<br>0 / 81 coefficients reduced exactly to zero.</p></div>
        <div class="card"><h3>ElasticNet</h3><p><b>RMSE 17.387 K</b><br>R² 0.737<br>α = 0.0001, l1_ratio = 0.9</p></div>
      </div>
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
      <h2>Case Study</h2>
      <div class="grid3">
        <div class="card"><h3>Problem</h3><p>Estimate critical temperature from material descriptors and explore whether composition carries additional predictive signal.</p></div>
        <div class="card"><h3>Approach</h3><p>EDA, multicollinearity analysis, Linear/Ridge/Lasso/ElasticNet, GridSearchCV, 5-fold CV, residual diagnostics and deployment.</p></div>
        <div class="card"><h3>Outcome</h3><p>Stable regularized model, transparent comparison with Linear Regression, composition uplift and deployable FastAPI service.</p></div>
      </div>
    </section>

    <section class="section">
      <h2>Developer & Technical Resources</h2>
      <div class="buttons">
        <a class="btn primary" href="/docs">Open Swagger Docs</a>
        <a class="btn secondary" href="/api/v1/model-info">Model Info</a>
        <a class="btn secondary" href="/api/v1/schema">Feature Schema</a>
        <a class="btn secondary" href="/api/v1/metrics">Live Metrics</a>
        <a class="btn secondary" href="https://github.com/mightyalok00/superconductivity-elasticnet-regression/blob/main/docs/MODEL_CARD.md">Model Card</a>
        <a class="btn secondary" href="https://github.com/mightyalok00/superconductivity-elasticnet-regression/blob/main/docs/DATA_CARD.md">Data Card</a>
        <a class="btn secondary" href="https://github.com/mightyalok00/superconductivity-elasticnet-regression/blob/main/docs/ARCHITECTURE.md">Architecture</a>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>cURL example</h3>
        <code id="curlExample">curl https://superconductivity-ml-api.up.railway.app/api/v1/model-info</code>
        <div class="buttons"><button class="btn secondary" onclick="copyText(document.getElementById('curlExample').innerText)">Copy cURL</button></div>
      </div>
    </section>

    <section class="section">
      <h2>Live Service Status</h2>
      <div class="metrics">
        <div class="metric"><b id="uptime">—</b><span>uptime</span></div>
        <div class="metric"><b id="requests">—</b><span>requests</span></div>
        <div class="metric"><b id="predictions">—</b><span>predictions</span></div>
        <div class="metric"><b id="artifact">—</b><span>model artifact</span></div>
      </div>
    </section>

    <footer>
      <div class="socials" style="justify-content:center">
        <a class="btn secondary" href="https://github.com/mightyalok00">GitHub</a>
        <a class="btn secondary" href="https://www.linkedin.com/in/alok-agarwal-seo-ai-ml/">LinkedIn</a>
      </div>
      <p>Built by Alok Agarwal • FastAPI • scikit-learn • Railway • MIT License</p>
    </footer>
  </main>
<script>
const $=id=>document.getElementById(id);

function toggleTheme(){document.body.classList.toggle('light');sessionStorage.setItem('theme',document.body.classList.contains('light')?'light':'dark')}
if(sessionStorage.getItem('theme')==='light')document.body.classList.add('light');

async function copyText(text){await navigator.clipboard.writeText(text)}
function copyEndpoint(path){copyText(location.origin+path)}

async function loadSample(){
  const r=await fetch('/api/v1/sample'); const data=await r.json();
  $('payload').value=JSON.stringify(data,null,2);
}
function resetPrediction(){$('payload').value='{}';$('result').style.display='none';$('contributors').innerHTML='<span class="small">Run a prediction to see contributors.</span>'}

async function downloadSampleJson(){
  const r=await fetch('/api/v1/sample-download.json'); const blob=await r.blob();
  downloadBlob(blob,'superconductivity_sample.json');
}
async function downloadSampleCsv(){
  const r=await fetch('/api/v1/sample.csv'); const blob=await r.blob();
  downloadBlob(blob,'superconductivity_sample.csv');
}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url)}

function renderContributors(items){
  $('contributors').innerHTML=items.map(x=>{
    const width=Math.min(100,Math.abs(x.contribution_k)*4);
    const sign=x.contribution_k>=0?'+':'';
    return '<div style="margin:11px 0"><div class="small"><b>'+x.feature+'</b> · '+x.group+' · '+sign+x.contribution_k.toFixed(3)+' K</div><div class="bar"><span style="width:'+width+'%"></span></div></div>';
  }).join('');
}

function addHistory(data){
  const arr=JSON.parse(sessionStorage.getItem('predictionHistory')||'[]');
  arr.unshift({time:new Date().toLocaleTimeString(),temp:data.predicted_critical_temp_k,ood:data.ood_risk_score,lat:data.response_time_ms,id:data.request_id});
  sessionStorage.setItem('predictionHistory',JSON.stringify(arr.slice(0,20)));renderHistory();
}
function renderHistory(){
  const arr=JSON.parse(sessionStorage.getItem('predictionHistory')||'[]');
  $('historyBody').innerHTML=arr.length?arr.map(x=>'<tr><td>'+x.time+'</td><td>'+x.temp.toFixed(2)+' K</td><td>'+x.ood.toFixed(1)+'%</td><td>'+x.lat.toFixed(1)+' ms</td><td><code>'+x.id.slice(0,8)+'</code></td></tr>').join(''):'<tr><td colspan="5" class="small">No predictions in this session yet.</td></tr>';
}
function clearHistory(){sessionStorage.removeItem('predictionHistory');renderHistory()}

async function predict(){
  const btn=$('predictBtn'),box=$('result');btn.disabled=true;box.style.display='block';box.innerHTML='Running prediction…';
  try{
    const body=JSON.parse($('payload').value);
    const r=await fetch('/api/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await r.json();
    if(!r.ok){box.innerHTML='<b>Request error</b><pre>'+JSON.stringify(data,null,2)+'</pre>';return}
    const warnings=data.warnings.length?'<div class="warn">'+data.warnings.slice(0,6).join('<br>')+'</div>':'';
    box.innerHTML='<div class="small">Predicted critical temperature</div><strong>'+data.predicted_critical_temp_k.toFixed(2)+' K</strong><div class="small">Expected RMSE ≈ '+data.expected_rmse_k+' K · OOD risk '+data.ood_risk_score.toFixed(1)+'% · '+data.response_time_ms.toFixed(1)+' ms</div>'+warnings;
    renderContributors(data.top_contributors);addHistory(data);loadMetrics();
  }catch(e){box.innerHTML='<b>Invalid request</b><div class="warn">'+e.message+'</div>'}finally{btn.disabled=false}
}

async function loadCompareSamples(){
  const r=await fetch('/api/v1/samples?count=2');const data=await r.json();
  $('compareA').value=JSON.stringify({features:data.samples[0]},null,2);
  $('compareB').value=JSON.stringify({features:data.samples[1]},null,2);
}
async function compareMaterials(){
  const box=$('compareResult');box.style.display='block';box.innerHTML='Comparing…';
  try{
    const a=JSON.parse($('compareA').value).features;const b=JSON.parse($('compareB').value).features;
    const r=await fetch('/api/v1/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({material_a:a,material_b:b})});
    const d=await r.json();if(!r.ok){box.innerHTML='<pre>'+JSON.stringify(d,null,2)+'</pre>';return}
    box.innerHTML='<strong>'+d.material_a.predicted_critical_temp_k.toFixed(2)+' K</strong> vs <strong>'+d.material_b.predicted_critical_temp_k.toFixed(2)+' K</strong><p>'+d.higher_predicted_material+' is higher by '+Math.abs(d.difference_k).toFixed(2)+' K.</p>';
  }catch(e){box.innerHTML='<div class="warn">'+e.message+'</div>'}
}

async function uploadCsv(){
  const f=$('csvFile').files[0],status=$('batchStatus'),btn=$('batchBtn');if(!f){status.textContent='Choose a CSV file first.';return}
  btn.disabled=true;status.textContent='Uploading, validating, ranking and predicting…';
  const form=new FormData();form.append('file',f);
  const minTemp=$('minTemp').value||'';
  const r=await fetch('/api/v1/predict-batch?min_temp='+encodeURIComponent(minTemp),{method:'POST',body:form});
  if(!r.ok){status.textContent='Batch request failed: '+await r.text();btn.disabled=false;return}
  const blob=await r.blob();downloadBlob(blob,'ranked_superconductivity_predictions.csv');status.textContent='Ranked prediction file downloaded.';btn.disabled=false;loadMetrics();
}

async function loadFeatureSchema(){
  const r=await fetch('/api/v1/schema');const d=await r.json();const groups={};
  for(const [name,info] of Object.entries(d.features)){const g=info.group||'Other';(groups[g]??=[]).push([name,info])}
  $('featureGroups').innerHTML=Object.entries(groups).map(([g,items])=>'<details><summary>'+g+' ('+items.length+')</summary>'+items.map(([name,i])=>'<div class="feature-row" title="'+(i.description||'')+'"><b>'+name+'</b><span>min '+fmt(i.min)+'</span><span>median '+fmt(i.median)+'</span><span>max '+fmt(i.max)+'</span></div>').join('')+'</details>').join('');
}
function fmt(v){return typeof v==='number'?Number(v).toPrecision(4):'—'}

async function loadTopFeatures(){
  const r=await fetch('/api/v1/top-features?limit=10');const d=await r.json();
  $('topFeatures').innerHTML=d.features.map((x,i)=>'<div style="margin:10px 0"><span class="small">'+(i+1)+'. <b>'+x.feature+'</b> · '+x.group+' · |coefficient| '+Math.abs(x.coefficient).toFixed(3)+'</span><div class="bar"><span style="width:'+Math.min(100,Math.abs(x.coefficient))+'%"></span></div></div>').join('');
}

async function loadMetrics(){
  const r=await fetch('/api/v1/metrics');const d=await r.json();
  $('uptime').textContent=Math.floor(d.uptime_seconds/60)+' min';$('requests').textContent=d.request_count;$('predictions').textContent=d.prediction_count;$('artifact').textContent=d.artifact_available?'Ready':'Fallback';
}

loadSample();loadCompareSamples();renderHistory();loadFeatureSchema();loadTopFeatures();loadMetrics();
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
        "environment": settings.app_env,
        "rate_limit_enabled": settings.enable_rate_limit,
        "api_key_required": settings.require_api_key,
    }


@app.get("/api/v1/metrics", tags=["System"])
async def live_metrics():
    return {
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "request_count": REQUEST_COUNT,
        "prediction_count": PREDICTION_COUNT,
        "artifact_available": MODEL_PATH.exists(),
        "model_loaded": model is not None,
        "api_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.get("/api/v1/model-info", tags=["Model"])
@app.get("/model-info", tags=["Model"], include_in_schema=False)
async def model_info():
    return {
        "model": "ElasticNet Regression",
        "model_version": MODEL_VERSION,
        "api_version": APP_VERSION,
        "alpha": BEST_ALPHA,
        "l1_ratio": BEST_L1_RATIO,
        "training_rows": training_rows,
        "feature_count": len(feature_names),
        "artifact_available": MODEL_PATH.exists(),
        "validated_test_metrics": {"mae_k": 13.210, "rmse_k": 17.387, "r2": 0.737},
        "cv_metrics": {"rmse_mean_k": 17.699, "rmse_std_k": 0.257, "r2_mean": 0.734, "r2_std": 0.006},
    }


@app.get("/api/v1/features", tags=["Model"])
@app.get("/features", tags=["Model"], include_in_schema=False)
async def features():
    return {"count": len(feature_names), "features": feature_names}


@app.get("/api/v1/schema", tags=["Model"])
async def schema():
    enriched = {}
    for name in feature_names:
        stats = feature_stats.get(name, {})
        enriched[name] = {
            **stats,
            "group": feature_group(name),
            "description": feature_description(name),
        }
    return {"target": TARGET, "feature_count": len(feature_names), "features": enriched}


@app.get("/api/v1/top-features", tags=["Model"])
async def top_features(limit: int = Query(10, ge=1, le=30)):
    active_model = await run_in_threadpool(ensure_model)
    estimator = active_model.named_steps["model"]
    features_out = [
        {
            "feature": name,
            "group": feature_group(name),
            "coefficient": round(float(coef), 6),
        }
        for name, coef in zip(feature_names, estimator.coef_)
    ]
    features_out.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
    return {"features": features_out[:limit]}


@app.get("/api/v1/sample", tags=["Prediction"])
@app.get("/sample", tags=["Prediction"], include_in_schema=False)
async def sample_payload():
    df = pd.read_csv(DATA_PATH, nrows=1)
    sample = df.drop(columns=[TARGET]).iloc[0].to_dict()
    return {"features": {k: float(v) for k, v in sample.items()}}


@app.get("/api/v1/samples", tags=["Prediction"])
async def samples(count: int = Query(2, ge=1, le=10)):
    df = pd.read_csv(DATA_PATH, nrows=count)
    return {"samples": [{k: float(v) for k, v in row.items()} for row in df.drop(columns=[TARGET]).to_dict(orient="records")]}


@app.get("/api/v1/sample.csv", tags=["Prediction"])
async def sample_csv():
    df = pd.read_csv(DATA_PATH, nrows=1).drop(columns=[TARGET])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    response = StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = 'attachment; filename="superconductivity_sample.csv"'
    return response


@app.get("/api/v1/sample-download.json", tags=["Prediction"])
async def sample_json_download():
    df = pd.read_csv(DATA_PATH, nrows=1).drop(columns=[TARGET])
    payload = {"features": {k: float(v) for k, v in df.iloc[0].to_dict().items()}}
    response = JSONResponse(payload)
    response.headers["Content-Disposition"] = 'attachment; filename="superconductivity_sample.json"'
    return response


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], include_in_schema=False)
async def predict(payload: PredictionRequest, request: Request):
    global PREDICTION_COUNT
    started = time.perf_counter()
    active_model = await run_in_threadpool(ensure_model)
    result = predict_one(active_model, payload.features)
    PREDICTION_COUNT += 1
    return PredictionResponse(
        **result,
        model="ElasticNet Regression",
        model_version=MODEL_VERSION,
        feature_count=len(feature_names),
        expected_rmse_k=17.387,
        response_time_ms=round((time.perf_counter() - started) * 1000, 2),
        request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
    )


@app.post("/api/v1/compare", tags=["Prediction"])
async def compare_materials(payload: CompareRequest):
    global PREDICTION_COUNT
    active_model = await run_in_threadpool(ensure_model)
    a = predict_one(active_model, payload.material_a)
    b = predict_one(active_model, payload.material_b)
    PREDICTION_COUNT += 2
    diff = a["predicted_critical_temp_k"] - b["predicted_critical_temp_k"]
    return {
        "material_a": a,
        "material_b": b,
        "difference_k": round(diff, 4),
        "higher_predicted_material": "Material A" if diff >= 0 else "Material B",
    }


@app.post("/api/v1/predict-batch", tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(...),
    min_temp: float | None = Query(None, description="Optional minimum predicted critical temperature in Kelvin."),
):
    global PREDICTION_COUNT
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
    risk_scores = []
    for _, row in ordered.iterrows():
        payload = {k: float(row[k]) for k in feature_names}
        warning_counts.append(len(range_warnings(payload)))
        risk_scores.append(ood_score(payload))

    result = ordered.copy()
    result["predicted_critical_temp_k"] = predictions
    result["ood_warning_count"] = warning_counts
    result["ood_risk_score"] = risk_scores
    result = result.sort_values("predicted_critical_temp_k", ascending=False).reset_index(drop=True)
    result.insert(0, "candidate_rank", np.arange(1, len(result) + 1))

    if min_temp is not None:
        result = result[result["predicted_critical_temp_k"] >= min_temp].copy()

    PREDICTION_COUNT += len(ordered)
    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    response = StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = 'attachment; filename="ranked_superconductivity_predictions.csv"'
    return response
