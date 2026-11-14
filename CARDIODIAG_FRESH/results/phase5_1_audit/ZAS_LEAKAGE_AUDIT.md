# ZAS leakage and validation audit

## Implementation evidence

1. The target is mapped with `frame["Cath"].map({"Cad": 1, "Normal": 0})` at `runner/phase5/zas_track/run.py:31-35`.
2. The six feature groups and six model families are declared before the loops at `run.py:20-28`. `eligible_combined` is not assembled dynamically from test outcomes.
3. The outer design is `RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=5107)` at `run.py:38`. Each outer training partition is separated from its test partition at `run.py:41-43`.
4. `GridSearchCV` performs three-fold inner selection at `run.py:44-46`. The estimator is a pipeline; imputation, scaling, spline transformation and one-hot encoding are therefore fitted inside each inner training fold by `runner/phase5/common/models.py:11-40`.
5. No automated feature selection occurs. The feature list is fixed before each model fit. Consequently, feature-set selection across the six groups is **not** nested as a separate model-selection layer.
6. Predictions are written at `run.py:50-52`; repeated predictions are averaged per `(group, model, row_index)` at `run.py:53`. This correctly returns one aggregated prediction per the 303 source rows for each group/model, but repeated folds are not independent patients.
7. Metrics are computed from the aggregated predictions at `run.py:54-58` using `runner/phase5/common/metrics.py:19-40`. Calibration intercept and slope are descriptive evaluation statistics, not tuning operations.

## Integrity conclusions

- Fold-local preprocessing: **supported by code**.
- Inner hyperparameter selection: **nested within each outer training partition**.
- Outer test leakage during fitting: **not identified**.
- Fold-local feature selection: **not applicable; features were fixed**.
- Patient-level aggregation: **one row per source patient after repeat averaging**; no patient identifier exists beyond the source row index.
- Independent evaluation unit: **303 source rows**, not 606 repeated predictions.
- Direct target leakage in `eligible_combined`: **not identified**.

## Material limitation

The output compares six feature groups and six model families, producing 36 benchmark rows and 288 outer selection records. The row with ROC-AUC 0.928 was identified after viewing this benchmark table. Inner nesting protects hyperparameter selection within a candidate, but it does not protect the subsequent choice of the best feature group/model from selection optimism. The result is therefore not an unbiased estimate of a prespecified final model.

The output also records `optimism_correction` as not implemented. Repeated out-of-fold averaging reduces dependence from repeated predictions, but it does not remove candidate-selection optimism or resolve feature-timing uncertainty.
