# Executive V3 Results Report

V3 is a fully nested calibration correction using the exact Stage 6 outer and inner assignments. Calibration-fold labels were excluded from the hyperparameter selection that generated their calibration scores.

- Records: 303; class counts: {'Cad': 216, 'Normal': 87}.
- Strict features: Age, Sex, Current Smoker, EX-Smoker, FH.
- Highest descriptive calibrated ROC-AUC: `logistic_regression_balanced` = 0.735 +/- 0.060.
- This remains internal validation, not external validation or clinical utility evidence.
