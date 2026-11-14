# Executive Results Report

This report summarizes the leakage-controlled internal validation experiment. It is not external validation or a clinical deployment assessment.

- Records: 303; CAD cases: 216; normal cases: 87.
- Strict features: 5 (`Age, Sex, Current Smoker, EX-Smoker, FH`).
- Validation: 5 repeats of stratified 5-fold outer CV with 4-fold inner selection.
- Highest mean calibrated ROC-AUC (descriptive): `logistic_regression_none` = 0.737 +/- 0.059.

## Interpretation

All preprocessing and sigmoid calibration were fit without outer test labels. The model ranking and calibration estimates are subject to substantial small-cohort uncertainty. The statistical comparison file reports paired bootstrap effects without confirmatory independent-fold p-values. Feature importance is predictive association, not causation.

## Artifact references

- `MODEL_COMPARISON_RESULTS.csv` contains outer-fold and summary metrics.
- `OUTER_FOLD_PREDICTIONS.csv` contains every evaluated outer-test prediction.
- `CALIBRATION_RESULTS.csv`, `UNCERTAINTY_RESULTS.csv`, and `STATISTICAL_COMPARISONS.csv` contain reliability and uncertainty analyses.
