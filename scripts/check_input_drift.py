from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "outputs" / "models" / "model_metadata.json"


def classify_shift(z: float) -> str:
    value = abs(z)
    if value >= 2.0:
        return "high"
    if value >= 1.0:
        return "moderate"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate-data feature means against training metadata."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "drift_report.json",
    )
    args = parser.parse_args()

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    expected = metadata["features"]
    stats = metadata["feature_stats"]

    df = pd.read_csv(args.csv_path)
    missing = [name for name in expected if name not in df.columns]
    if missing:
        raise SystemExit(f"Missing required features: {missing}")

    report = []
    for name in expected:
        train_mean = float(stats[name]["mean"])
        train_std = float(stats[name].get("std", 0.0))
        candidate_mean = float(df[name].mean())

        if train_std > 0:
            standardized_mean_shift = (candidate_mean - train_mean) / train_std
        else:
            standardized_mean_shift = 0.0

        report.append(
            {
                "feature": name,
                "training_mean": train_mean,
                "candidate_mean": candidate_mean,
                "standardized_mean_shift": round(float(standardized_mean_shift), 4),
                "drift_level": classify_shift(standardized_mean_shift),
            }
        )

    report.sort(key=lambda item: abs(item["standardized_mean_shift"]), reverse=True)

    summary = {
        "rows_checked": int(len(df)),
        "feature_count": len(expected),
        "high_drift_features": sum(x["drift_level"] == "high" for x in report),
        "moderate_drift_features": sum(x["drift_level"] == "moderate" for x in report),
        "features": report,
        "note": (
            "This is a lightweight mean-shift diagnostic, not a substitute for "
            "full production drift monitoring or statistical hypothesis testing."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Drift report written to {args.output}")


if __name__ == "__main__":
    main()
