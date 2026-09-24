# 16 Project Questions

1. What is the distribution of superconducting critical temperature (`critical_temp`), and does it contain skewness or extreme values?
2. Which material-property features have the strongest positive and negative relationships with `critical_temp`?
3. How much multicollinearity exists among the 81 predictor variables, and which groups of features are highly correlated?
4. Do weighted material-property features such as `wtd_mean_atomic_mass`, `wtd_mean_fie`, and `wtd_mean_ThermalConductivity` provide stronger predictive information than their unweighted counterparts?
5. How does the number of elements (`number_of_elements`) in a material relate to its superconducting critical temperature?
6. Which physical-property groups—atomic mass, first ionization energy, atomic radius, density, electron affinity, fusion heat, thermal conductivity, or valence—are most important for predicting `critical_temp`?
7. How does a baseline Linear Regression model perform when predicting superconducting critical temperature?
8. Does Ridge Regression improve generalization compared with ordinary Linear Regression when highly correlated predictors are present?
9. Which features does Lasso Regression shrink to zero, and what does this reveal about potentially redundant predictors?
10. Does ElasticNet Regression outperform Linear, Ridge, and Lasso Regression by combining L1 feature selection with L2 regularization?
11. What combination of `alpha` and `l1_ratio` produces the best ElasticNet model using cross-validation?
12. How stable is the best ElasticNet model under 5-fold cross-validation, and what are its mean and standard deviation for RMSE and R²?
13. Which variables retain the largest positive and negative coefficients after ElasticNet regularization, and what might these coefficients indicate about critical temperature?
14. How well does the final ElasticNet model perform on an untouched test set using MAE, MSE, RMSE, and R², and what do its residuals reveal about model errors?
15. Can the elemental-composition information in `unique_m.csv` be combined with the engineered features in `train.csv` to improve critical-temperature prediction beyond using `train.csv` alone?
16. If this ElasticNet model were used by a superconducting-materials R&D team, which material properties should researchers prioritize when screening new candidate materials, and how could the model help reduce experimental cost, testing time, and the number of low-potential materials sent for laboratory validation?
