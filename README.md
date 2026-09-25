# 🧲 Superconductivity Critical Temperature Prediction

<p align="center">
  <b>ElasticNet • Ridge • Lasso • Linear Regression • Scientific ML Analysis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

> 🔬 Predicting superconducting `critical_temp` from engineered material properties and elemental composition while answering **16 structured analytical questions** with reproducible modeling, cross-validation, feature interpretation, residual analysis, and R&D-oriented conclusions.

---

## 🚀 Project Highlights

- 📊 **21,263 superconducting materials**
- 🧪 **81 engineered predictors**
- 🎯 Target: **`critical_temp`**
- 🧠 Models: **Linear Regression, Ridge, Lasso, ElasticNet**
- 🔁 Validation: **5-fold cross-validation + untouched test set**
- 🧬 Additional experiment: **elemental composition from `unique_m.csv`**
- 📚 **16/16 questions answered**
- ✅ Notebook executed successfully on GitHub
- 🤖 GitHub Actions workflow included for reproducible notebook execution

---

## 🏆 Key Results

| Metric / Finding | Result |
|---|---:|
| 📌 Mean critical temperature | **34.42 K** |
| 📍 Median critical temperature | **20.00 K** |
| 📈 Target skewness | **0.860** |
| ⚠️ IQR extreme values | **1** |
| 🔥 Strongest positive correlation | `wtd_std_ThermalConductivity` ≈ **+0.721** |
| 🧊 Strongest negative correlation | `wtd_mean_Valence` ≈ **-0.632** |
| 🔗 Predictor pairs with `|r| ≥ 0.90` | **74** |
| ⚖️ Weighted means stronger than simple means | **7 / 8** groups |
| 📉 Linear Regression RMSE | **17.378 K** |
| 🎯 Linear Regression R² | **0.738** |
| 🧱 Ridge best alpha | **0.1** |
| ✂️ Lasso zero coefficients | **0 / 81** |
| 🧬 ElasticNet best alpha | **0.0001** |
| 🎚️ ElasticNet best `l1_ratio` | **0.9** |
| 🔁 ElasticNet CV RMSE | **17.699 ± 0.257 K** |
| 🔁 ElasticNet CV R² | **0.734 ± 0.006** |
| 🧪 ElasticNet test RMSE | **17.387 K** |
| 🧪 ElasticNet test R² | **0.737** |
| 🧬 Adding elemental composition | **RMSE improves by 0.876 K** |
| 📈 Elemental composition R² gain | **+0.0258** |

---

## 🧠 What the Model Actually Found

### 🔥 Strong positive signal
`wtd_std_ThermalConductivity` shows the strongest positive linear relationship with critical temperature:

**r ≈ +0.721**

### 🧊 Strong negative signal
`wtd_mean_Valence` shows the strongest negative linear relationship:

**r ≈ -0.632**

### 🔗 Strong multicollinearity
The dataset contains **74 predictor pairs with |r| ≥ 0.90**.

That makes regularization especially relevant because ordinary linear-regression coefficients can become unstable when predictors strongly overlap.

### ⚖️ Weighted features matter
Weighted material-property means outperform their unweighted counterparts in **7 of 8 matched property groups**.

This suggests composition-aware weighting preserves useful information that simple averages can miss.

---

## 🤖 Model Comparison

| Model | Main takeaway |
|---|---|
| 📏 Linear Regression | **Best test RMSE among the compared models** |
| 🧱 Ridge Regression | Similar accuracy; useful for coefficient stabilization |
| ✂️ Lasso Regression | CV selected very weak regularization; no features were eliminated |
| 🧬 ElasticNet Regression | Stable, interpretable regularized model with mixed L1/L2 penalty |

> 💡 Important: ElasticNet was **not forced to “win.”** The project reports the measured result honestly: **Linear Regression achieved the lowest test RMSE in this comparison.**

---

## 🎯 Final ElasticNet Performance

- **MAE:** 13.210 K
- **MSE:** 302.295
- **RMSE:** 17.387 K
- **R²:** 0.737
- **Residual mean:** -0.160 K
- **Residual SD:** 17.386 K

### 🔁 5-Fold Cross-Validation

- **RMSE:** 17.699 ± 0.257 K
- **R²:** 0.734 ± 0.006

The low fold-to-fold variation suggests reasonably stable performance across the training folds.

---

## 🧬 Q15 — Does Elemental Composition Help?

Yes.

Adding elemental-composition features from `unique_m.csv`:

- ✅ improves RMSE by **0.876 K**
- ✅ improves R² by **+0.0258**

### Interpretation

The elemental-composition variables provide additional predictive information beyond the 81 engineered features in `train.csv`.

This is one of the strongest practical findings in the project because it shows that raw composition information contains signal that is not fully captured by the engineered feature set.

---

## 🔬 Q16 — R&D / Scientific Interpretation

The highest-magnitude standardized ElasticNet coefficients include:

- `wtd_gmean_atomic_radius`
- `wtd_mean_atomic_radius`
- `entropy_fie`
- `wtd_mean_atomic_mass`
- `std_ElectronAffinity`
- `wtd_mean_FusionHeat`
- `entropy_Valence`
- `wtd_gmean_Valence`

### 🧪 Potential R&D workflow

1. 🧬 Generate or collect candidate superconducting materials.
2. ⚙️ Compute the same engineered features used in training.
3. 🤖 Predict critical temperature.
4. 📊 Rank candidates by predicted potential.
5. 🧠 Apply domain constraints such as stability, toxicity, cost, and manufacturability.
6. 🔬 Send the strongest candidates to laboratory validation first.
7. ♻️ Feed new experimental outcomes back into future model versions.

### 💰 Potential value

The model can help:

- reduce low-potential lab screening,
- prioritize promising material candidates,
- organize experimental decision-making,
- compare many candidates consistently,
- reduce repetitive screening effort,
- support scientific hypothesis generation.

> ⚠️ This is **decision support**, not proof of causation and not a substitute for laboratory experiments.

---

## ✅ 16 / 16 Questions Answered

The notebook includes a direct **`CONCLUSION Q1` → `CONCLUSION Q16`** after every analysis section.

| # | Question Area | Status |
|---:|---|:---:|
| 1 | Target distribution, skewness, extreme values | ✅ |
| 2 | Strongest positive and negative relationships | ✅ |
| 3 | Multicollinearity | ✅ |
| 4 | Weighted vs unweighted features | ✅ |
| 5 | Number of elements vs critical temperature | ✅ |
| 6 | Physical-property importance | ✅ |
| 7 | Linear Regression baseline | ✅ |
| 8 | Ridge vs Linear Regression | ✅ |
| 9 | Lasso feature shrinkage | ✅ |
| 10 | ElasticNet vs other models | ✅ |
| 11 | Best `alpha` and `l1_ratio` | ✅ |
| 12 | 5-fold CV stability | ✅ |
| 13 | Positive / negative ElasticNet coefficients | ✅ |
| 14 | Test metrics + residual analysis | ✅ |
| 15 | Elemental-composition experiment | ✅ |
| 16 | R&D / business-scientific interpretation | ✅ |

📄 Full question list: [docs/16_questions.md](docs/16_questions.md)

📘 Detailed interpretation: [docs/RESULTS_AND_INTERPRETATION.md](docs/RESULTS_AND_INTERPRETATION.md)

---

## 🛠️ Tech Stack

<p>
  🐍 Python &nbsp;•&nbsp;
  🐼 Pandas &nbsp;•&nbsp;
  🔢 NumPy &nbsp;•&nbsp;
  📊 Matplotlib &nbsp;•&nbsp;
  🎨 Seaborn &nbsp;•&nbsp;
  🤖 scikit-learn &nbsp;•&nbsp;
  📓 Jupyter Notebook &nbsp;•&nbsp;
  ⚙️ GitHub Actions
</p>

---

## 📁 Project Structure

~~~text
superconductivity-elasticnet-regression/
├── 📂 data/
│   └── raw/
│       ├── train.csv
│       └── unique_m.csv
│
├── 📂 docs/
│   ├── 16_questions.md
│   └── RESULTS_AND_INTERPRETATION.md
│
├── 📂 notebooks/
│   └── Superconductivity_ElasticNet_16_Questions.ipynb
│
├── 📂 outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│       └── final_summary.json
│
├── 📂 src/
│   ├── data_loader.py
│   ├── evaluation.py
│   └── modeling.py
│
├── ⚙️ config.py
├── ▶️ run_analysis.py
├── 📦 requirements.txt
└── 📘 README.md
~~~

---

## ▶️ How to Run

### 1️⃣ Clone the repository

~~~bash
git clone https://github.com/mightyalok00/superconductivity-elasticnet-regression.git
cd superconductivity-elasticnet-regression
~~~

### 2️⃣ Create a virtual environment

~~~bash
python -m venv .venv
~~~

### 3️⃣ Activate it

**Windows**

~~~bash
.venv\Scripts\activate
~~~

**macOS / Linux**

~~~bash
source .venv/bin/activate
~~~

### 4️⃣ Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 5️⃣ Run the notebook

Open:

~~~text
notebooks/Superconductivity_ElasticNet_16_Questions.ipynb
~~~

Or run the quick comparison script:

~~~bash
python run_analysis.py
~~~

---

## 🔁 Reproducibility

The project includes a GitHub Actions workflow that can execute the notebook and save rendered outputs automatically.

The modeling pipeline uses:

- ✅ consistent train/test splitting
- ✅ scaling inside scikit-learn pipelines
- ✅ GridSearchCV
- ✅ 5-fold cross-validation
- ✅ untouched final test data
- ✅ fixed random state
- ✅ saved final model summary

---

## 🧠 ML Skills Demonstrated

- 📊 Exploratory Data Analysis
- 🔗 Correlation & multicollinearity analysis
- 🧹 Feature comparison
- 📏 Linear Regression
- 🧱 Ridge Regression
- ✂️ Lasso Regression
- 🧬 ElasticNet Regression
- 🎛️ Hyperparameter tuning
- 🔁 Cross-validation
- 📉 Regression evaluation
- 🧪 Residual analysis
- 🔍 Feature interpretation
- 🧬 Multi-source feature integration
- 🏭 R&D decision-support thinking
- 🗂️ Reproducible ML project structure

---

## ⚠️ Limitations

- Correlation does **not** establish causation.
- The models are linear and may miss nonlinear material interactions.
- Highly correlated predictors complicate coefficient interpretation.
- Small sample sizes exist for some high element-count groups.
- Dataset performance does not guarantee performance on unseen material families.
- Real scientific use requires independent datasets and laboratory validation.

---

## 🔮 Future Improvements

- 🌲 Random Forest
- 🚀 Gradient Boosting
- ⚡ XGBoost / LightGBM
- 🔍 SHAP explanations
- 🔁 Repeated cross-validation
- 📉 Lasso regularization path
- 🧪 Error analysis by critical-temperature range
- 🌐 Streamlit prediction app
- 📦 Model deployment API
- 🔬 Independent experimental validation

---

## 👨‍💻 Author

**Alok Agarwal**

SEO & Digital Marketing professional transitioning into **Data Science, AI & Machine Learning**, building practical projects that combine analytics, modeling, business interpretation, and reproducible workflows.

<p align="left">
  <a href="https://github.com/mightyalok00">
    <img src="https://img.shields.io/badge/GitHub-mightyalok00-181717?logo=github&logoColor=white" />
  </a>
</p>

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ star**.

It helps make the project easier to discover and supports continued development.
