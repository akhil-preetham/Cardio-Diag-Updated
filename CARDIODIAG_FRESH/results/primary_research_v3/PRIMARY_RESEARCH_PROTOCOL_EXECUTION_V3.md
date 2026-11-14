# CARDIODIAG Primary Research Protocol V3 Execution

V3 preserves the Stage 6 outer and inner assignments. For each outer fold and each calibration fold, hyperparameter selection uses only the other three inner folds. The held-out calibration fold's labels therefore cannot influence the hyperparameters used to generate its score. A sigmoid calibrator is fitted on the resulting out-of-fold calibration scores, then applied to the final estimator selected and refitted on the complete outer-training partition.

Estimated and observed pipeline fit budget: approximately 16325 fits. Execution used n_jobs=1 and no new dataset or predictor.

Highest descriptive calibrated ROC-AUC: `logistic_regression_balanced` = 0.735 +/- 0.060.
