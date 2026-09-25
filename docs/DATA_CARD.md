# Data Card — Superconductivity Dataset

## Overview

This project uses the UCI superconductivity dataset, including:

- `train.csv` — engineered material-property features and `critical_temp`
- `unique_m.csv` — elemental composition information

## Main Training Table

- **Rows:** 21,263
- **Predictors:** 81
- **Target:** `critical_temp`
- **Missing values:** none identified in the project analysis

## Feature Families

The engineered features include material descriptors related to:

- atomic mass
- first ionization energy
- atomic radius
- density
- electron affinity
- fusion heat
- thermal conductivity
- valence
- composition-weighted summary statistics

## Known Considerations

- The predictors contain substantial multicollinearity.
- Some feature distributions are strongly skewed.
- Correlation does not imply causation.
- Training data may not represent all possible superconducting material families.

## Licensing

The dataset is third-party data and is not relicensed by this repository. Reuse should follow the original source terms and attribution requirements.
