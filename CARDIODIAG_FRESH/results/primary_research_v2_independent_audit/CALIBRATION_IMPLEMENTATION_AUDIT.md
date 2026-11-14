# Calibration Implementation Audit

## Classification: Requires correction

The implementation correctly performs a separate calibration fit inside every outer fold, uses sigmoid calibration, excludes outer-test labels, and labels SVM outputs as `decision_score`. Probability clipping to `[1e-6, 1-1e-6]` is used only for calibration-slope/intercept evaluation.

However, `GridSearchCV` first selects `best_params` over all four persisted inner folds. `cross_fitted_calibration` then refits that selected configuration on each inner training subset and predicts its inner calibration subset. Because the selected configuration was chosen using the held-out subset's labels, these are not fully independent calibration-training predictions. A corrected design should perform hyperparameter selection separately inside each calibration fold (using only the remaining inner-training data), or reserve a calibration layer before model selection. This audit does not rerun or alter V2.

Calibration evaluation on outer-test predictions is appropriate as evaluation, but the approximately 60 observations per outer fold make ten-bin ECE unstable. Calibration and discrimination are distinct: ROC-AUC assesses ranking, whereas Brier score, log loss, intercept, slope, reliability, and ECE assess probability quality.
