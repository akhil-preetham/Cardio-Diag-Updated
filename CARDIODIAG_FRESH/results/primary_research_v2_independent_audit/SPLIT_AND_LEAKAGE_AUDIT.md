# Split and Leakage Audit

## Classification

- Outer manifest equality: **Verified correct**.
- Inner manifest equality: **Verified correct**.
- Prediction coverage and duplicate-key checks are verified from the saved records.
- The code partitions data with `x.iloc[outer_train]` and `x.iloc[outer_test]`; preprocessing is inside `Pipeline` and `GridSearchCV`.
- Cross-fitted calibration fits each base pipeline on inner training rows and predicts its inner test rows. The calibrator is fitted on those outer-training scores and `y[outer_train]`, then applied to outer-test scores.
- No outer-test labels are passed to model selection, preprocessing, calibration fitting, or threshold selection.

## Limitation

Hyperparameters are selected by `GridSearchCV` using all four inner folds before the same folds are used to generate calibration scores. Thus each calibration fold's base-model hyperparameters were selected with that fold's labels included in the selection process. This creates selection-induced optimism in the calibration training scores. It is not outer-test leakage, but it means calibration is not fully nested with respect to hyperparameter selection.
