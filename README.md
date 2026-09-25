# Superconductivity Critical Temperature Prediction — ElasticNet Regression

Machine-learning analysis of the **UCI Superconductivity dataset** using Linear, Ridge, Lasso, and ElasticNet regression. The project answers **16 structured analytical questions** and connects model results to a realistic superconducting-materials R&D screening workflow.

## Project Snapshot

- **21,263 materials**
- **81 engineered predictors**
- Target: `critical_temp`
- Models: Linear Regression, Ridge, Lasso, ElasticNet
- Validation: untouched test set + 5-fold cross-validation
- Analysis: target distribution, correlations, multicollinearity, weighted vs unweighted features, regularization, coefficients, residuals, and R&D interpretation

## Key Results

| Finding | Result |
|---|---:|
| Mean critical temperature | **34.42 K** |
| Median critical temperature | **20 K** |
| Target skewness | **0.86** |
| IQR target outliers | **1** |
| Strongest positive target correlation | `wtd_std_ThermalConductivity` ≈ **+0.721** |
| Strongest negative target correlation | `wtd_mean_Valence` ≈ **-0.632** |
| Predictor pairs with `|r| >= 0.90` | **74** |
| Weighted means stronger than unweighted | **7 of 8** property groups |
| Linear Regression test RMSE | **17.38 K** |
| Linear Regression test R² | **0.738** |
| Ridge best alpha | **0.1** |
| Ridge test RMSE | **17.38 K** |
| Lasso best alpha | **0.0001** |
| Lasso coefficients exactly zero | **0 / 81** |

> **Interpretation:** the dataset contains substantial multicollinearity, so regularized regression is justified. Ridge provides coefficient stability but does not materially beat the linear baseline in the verified run. Lasso selects a very weak penalty, indicating that stronger sparsity reduced predictive performance within the tested grid.

## Feature Interpretation

The analysis identifies **thermal conductivity, atomic radius, valence, atomic mass, ionization energy, fusion heat, density, and electron affinity** as useful physical-property families.

Important interpretation rules:

- Correlation measures association, **not causation**.
- ElasticNet coefficients describe relationships within the fitted standardized linear model.
- Large coefficient magnitude does not prove that experimentally changing that property will cause a higher or lower critical temperature.
- Highly correlated predictors can share coefficient weight, so physical conclusions require domain expertise and laboratory validation.

The notebook extracts the largest positive and negative ElasticNet coefficients to help explain which engineered variables the final model relies on most.

## Why the Lasso Result Is Important

The tuned Lasso model selects `alpha = 0.0001` and shrinks **zero predictors exactly to zero**.

That is a meaningful result: cross-validation preferred predictive performance over sparsity. It would be incorrect to claim that Lasso found many unnecessary variables when the selected model did not.

## R&D / Scientific Value

This model is best viewed as a **candidate-screening tool**, not a replacement for experimental physics.

A practical R&D workflow could be:

1. Generate or collect candidate superconducting materials.
2. Compute the same engineered material features.
3. Predict critical temperature.
4. Rank candidates by predicted potential.
5. Apply domain constraints such as stability, cost, toxicity, and manufacturability.
6. Send only the most promising candidates for laboratory validation.
7. Feed experimental results back into future model versions.

This can potentially reduce low-value screening work, prioritize researcher attention, and make candidate comparison more systematic.

## Recruiter-Facing Skills Demonstrated

This repository demonstrates:

- exploratory data analysis
- regression modeling
- StandardScaler + scikit-learn pipelines
- train/test separation
- GridSearchCV
- 5-fold cross-validation
- Ridge, Lasso, and ElasticNet regularization
- multicollinearity analysis
- feature interpretation
- residual analysis
- model comparison
- reproducible project organization
- translation of ML output into scientific/R&D decision support


## Question-by-Question Coverage

All **16 project questions** are implemented in the notebook, and each section now ends with an explicit `CONCLUSION Q1` through `CONCLUSION Q16` statement. These conclusions are generated from the analysis results when the notebook is run, so the written answer stays tied to the actual metrics rather than a hard-coded claim.

This includes direct conclusions for:

- target distribution and outliers,
- strongest positive/negative relationships,
- multicollinearity,
- weighted vs unweighted properties,
- number of elements,
- physical-property groups,
- Linear Regression baseline,
- Ridge comparison,
- Lasso sparsity,
- ElasticNet model comparison,
- best `alpha` and `l1_ratio`,
- 5-fold CV stability,
- largest positive/negative coefficients,
- final test metrics and residual behavior,
- whether `unique_m.csv` improves prediction,
- and the R&D decision-support interpretation.

## Dataset

- `train.csv`: 21,263 rows × 82 columns (81 predictors + `critical_temp`)
- `unique_m.csv`: 21,263 rows × 88 columns (elemental composition + `critical_temp` + `material`)

## Project Structure

```text
superconductivity_elasticnet_project/
├── data/raw/
│   ├── train.csv
│   └── unique_m.csv
├── docs/
│   ├── 16_questions.md
│   └── RESULTS_AND_INTERPRETATION.md
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

## Run Order

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```text
notebooks/Superconductivity_ElasticNet_16_Questions.ipynb
```

4. Optional quick model comparison:

```bash
python run_analysis.py
```

## Models Covered

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- ElasticNetCV

## Evaluation Metrics

- MAE
- MSE
- RMSE
- R²
- 5-fold CV mean ± standard deviation
- residual analysis

## Detailed Interpretation

See **[Results & Interpretation](docs/RESULTS_AND_INTERPRETATION.md)** for:

- feature-level conclusions,
- correlation and coefficient interpretation,
- multicollinearity explanation,
- the Lasso sparsity result,
- R&D decision-support workflow,
- limitations,
- suggested next experiments.

## Reproducibility Notes

The notebook keeps the final test set untouched during model selection. Scaling is performed inside scikit-learn pipelines to reduce leakage risk.

The loader first tries the original Windows source paths and falls back to `data/raw/` when necessary.

## Limitations

- Results are associative rather than causal.
- Linear models may miss nonlinear relationships and interactions.
- Highly correlated predictors make individual coefficient interpretation harder.
- Performance on this dataset does not guarantee performance on unseen material families.
- Real scientific use requires independent data and laboratory validation.
