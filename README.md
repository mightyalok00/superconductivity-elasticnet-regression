# Superconductivity Critical Temperature Prediction — ElasticNet Regression

This project answers **16 structured questions** using the UCI Superconductivity dataset.

## Dataset
- `train.csv`: 21,263 rows × 82 columns (81 predictors + `critical_temp` target)
- `unique_m.csv`: 21,263 rows × 88 columns (elemental composition + `critical_temp` + `material`)

## Windows paths used by default
The loader first tries the exact paths requested:

```text
C:\Users\Alok Agarwal\Downloads\superconductivty+data\train.csv
C:\Users\Alok Agarwal\Downloads\superconductivty+data\unique_m.csv
```

If they are unavailable, it automatically falls back to `data/raw/` inside this project.

## Project structure
```text
superconductivity_elasticnet_project/
├── data/raw/
│   ├── train.csv
│   └── unique_m.csv
├── docs/
│   └── 16_questions.md
├── notebooks/
│   └── Superconductivity_ElasticNet_16_Questions.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   └── modeling.py
├── config.py
├── run_analysis.py
├── requirements.txt
└── README.md
```

## Run order
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `notebooks/Superconductivity_ElasticNet_16_Questions.ipynb`
4. Optional quick script: `python run_analysis.py`

## Models covered
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- ElasticNetCV

## Metrics
- MAE
- MSE
- RMSE
- R²
- 5-fold cross-validation mean ± standard deviation

## Notes
The notebook uses an untouched test set for final evaluation. Standardization is performed inside scikit-learn pipelines to reduce leakage risk.
