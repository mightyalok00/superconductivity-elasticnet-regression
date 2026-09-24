from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (WINDOWS_TRAIN_PATH, WINDOWS_UNIQUE_PATH, LOCAL_TRAIN_PATH,
                    LOCAL_UNIQUE_PATH, TARGET, RANDOM_STATE, TEST_SIZE)
from src.data_loader import load_datasets
from src.modeling import build_models
from src.evaluation import regression_metrics

# Load the two data files from the requested Windows paths or bundled fallback files.
train_df, unique_df, train_path, unique_path = load_datasets(
    WINDOWS_TRAIN_PATH, WINDOWS_UNIQUE_PATH, LOCAL_TRAIN_PATH, LOCAL_UNIQUE_PATH
)
print(f"Loaded train: {train_path} -> {train_df.shape}")
print(f"Loaded unique_m: {unique_path} -> {unique_df.shape}")

# Split predictors and target once so all baseline models use the same untouched test set.
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Fit and compare the four baseline/regularized regression models.
rows = []
for name, model in build_models(RANDOM_STATE).items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    row = {"Model": name, **regression_metrics(y_test, pred)}
    rows.append(row)

results = pd.DataFrame(rows).sort_values("RMSE")
print("\nModel comparison:\n", results.to_string(index=False))
