# Final Calibration Protocol

Use V3 fully nested calibration. For each calibration fold, hyperparameters are selected only on the other calibration-training folds; the held-out calibration fold generates the out-of-fold score. A sigmoid calibrator is fitted to the collected scores, and the final base estimator is independently selected and refit on the complete outer-training partition. Outer-test labels are never used. V4 preserves this procedure. A calibrated fold-model ensemble was not accepted because it was not independently implemented and prespecified.
