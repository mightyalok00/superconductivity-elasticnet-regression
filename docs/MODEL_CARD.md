# Model Card — ElasticNet Superconductivity Predictor

## Model Details

- **Model:** ElasticNet Regression
- **Version:** 1.0.0
- **Target:** `critical_temp` in Kelvin
- **Input features:** 81 engineered material-property features
- **Training rows:** 21,263
- **Deployment:** FastAPI on Railway

## Hyperparameters

- `alpha = 0.0001`
- `l1_ratio = 0.9`
- `random_state = 42`

## Validation Results

| Metric | Value |
|---|---:|
| Test MAE | 13.210 K |
| Test RMSE | 17.387 K |
| Test R² | 0.737 |
| 5-fold CV RMSE | 17.699 ± 0.257 K |
| 5-fold CV R² | 0.734 ± 0.006 |

## Intended Use

This model is intended for portfolio demonstration, exploratory scientific analysis, and candidate-screening decision support.

## Out-of-Scope Use

The model should not be used as a substitute for laboratory validation, safety-critical scientific decisions, or causal inference.

## Important Limitations

- Linear model assumptions may miss nonlinear material interactions.
- Training-distribution performance does not guarantee performance on unseen material families.
- Highly correlated features make individual coefficient interpretation difficult.
- Prediction error can be substantial relative to high-precision experimental applications.
