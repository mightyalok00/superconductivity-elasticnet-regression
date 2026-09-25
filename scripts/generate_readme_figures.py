from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "critical_temp"

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

models = {
    "Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=0.1)),
    ]),
    "Lasso": Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.0001, max_iter=30000)),
    ]),
    "ElasticNet": Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=0.0001, l1_ratio=0.9, max_iter=30000, random_state=RANDOM_STATE)),
    ]),
}

rows = []
elastic_pred = None
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    rows.append({"Model": name, "RMSE": rmse})
    if name == "ElasticNet":
        elastic_pred = pred

comparison = pd.DataFrame(rows).sort_values("RMSE", ascending=True)

plt.figure(figsize=(9, 5.4))
bars = plt.barh(comparison["Model"], comparison["RMSE"])
plt.xlabel("Test RMSE (K) — lower is better")
plt.title("Regression Model Comparison")
plt.xlim(comparison["RMSE"].min() - 0.02, comparison["RMSE"].max() + 0.02)
for bar, value in zip(bars, comparison["RMSE"]):
    plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "q10_model_comparison.png", dpi=180, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6.6, 6.2))
plt.scatter(y_test, elastic_pred, alpha=0.28, s=16)
lims = [min(float(y_test.min()), float(elastic_pred.min())),
        max(float(y_test.max()), float(elastic_pred.max()))]
plt.plot(lims, lims, "--", linewidth=1.5)
plt.xlabel("Actual critical_temp (K)")
plt.ylabel("Predicted critical_temp (K)")
plt.title("Actual vs Predicted — ElasticNet")
plt.tight_layout()
plt.savefig(OUT / "q14_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
plt.close()

residuals = y_test.to_numpy() - elastic_pred
plt.figure(figsize=(9, 5.4))
sns.histplot(residuals, kde=True, bins=40)
plt.axvline(0, linestyle="--", linewidth=1.3)
plt.xlabel("Residual (Actual − Predicted) K")
plt.ylabel("Count")
plt.title("ElasticNet Residual Distribution")
plt.tight_layout()
plt.savefig(OUT / "q14_residuals.png", dpi=180, bbox_inches="tight")
plt.close()

print(comparison.to_string(index=False))
print("Saved README figures to", OUT)
