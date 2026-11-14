# Calibration Implementation V3

V3 uses nested cross-fitting rather than wrapping a single global search in a calibrator. For calibration fold k, the other three persisted inner test partitions form the calibration-training data. A three-fold GridSearchCV selects hyperparameters using only those observations, fits the selected pipeline on them, and scores fold k. This is repeated for all four calibration folds.

The sigmoid calibrator is fit on the four held-out score blocks. The final outer model is selected independently with all four inner folds and refitted on all outer-training rows. It is a single refitted estimator; the fold-specific calibration models are used only to create calibration-training scores, not as an ensemble for outer-test prediction. SVM decision scores remain scores until the sigmoid transform.

This design prevents calibration-fold label reuse in the hyperparameter selection that generates its own score. It costs approximately 16,325 pipeline fits for 25 outer folds and nine configurations, with one bounded worker. Small-sample calibration and score-scale differences between fold-specific models remain limitations.
