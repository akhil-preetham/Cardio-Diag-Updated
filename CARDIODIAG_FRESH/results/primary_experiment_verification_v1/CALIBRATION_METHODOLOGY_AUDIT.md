# Calibration Methodology Audit

- **Verified correct:** `CalibratedClassifierCV` is fitted using only each outer training partition and its inner folds; outer test labels are not passed to calibration fitting.
- **Verified correct:** calibrated and uncalibrated predictions are stored separately and keyed by repeat, fold, model, and row index.
- **Correct with limitations:** the same inner training data support hyperparameter selection and calibration, so calibration uncertainty includes neither a separate calibration-selection layer nor independent calibration data. Outer-test isolation remains intact.
- **Requires correction in interpretation:** for SVM configurations, the uncalibrated column is `expit(decision_function)`, a monotone score mapping, not a native probability estimate. Its Brier/log-loss values should not be interpreted as calibrated probability quality.
- **Correct with limitations:** calibration intercept, slope, Brier score, log loss, and ECE are calculated on outer test labels as evaluation summaries, not as fitting inputs.
- **Correct with limitations:** ten equal-width ECE bins are unstable for outer test folds of approximately 60 observations. ECE is descriptive and should not be treated as a precise reliability estimate.
