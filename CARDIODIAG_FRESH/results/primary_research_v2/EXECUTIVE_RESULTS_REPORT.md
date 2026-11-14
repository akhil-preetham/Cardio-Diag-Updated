# Executive Results Report

This is the corrected V2 internal-validation experiment. It is not external validation or clinical deployment.

- Records: 303; class counts: {'Cad': 216, 'Normal': 87}.
- Strict features: Age, Sex, Current Smoker, EX-Smoker, FH.
- Persisted Stage 6 manifest used exactly: yes.
- Nested cross-fitted sigmoid calibration: yes.
- Descriptive highest calibrated ROC-AUC: `logistic_regression_balanced` = 0.735 +/- 0.060.

Model selection and calibration used only outer-training data. SVM decision scores are not treated as native probabilities. Comparisons are paired descriptive summaries without confirmatory p-values. Bootstrap outputs are conditional source-row resampling summaries, not population confidence intervals.
