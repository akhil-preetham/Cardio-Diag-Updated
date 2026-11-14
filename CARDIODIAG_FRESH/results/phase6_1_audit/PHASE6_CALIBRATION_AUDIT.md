# Phase 6 calibration audit

## What the code does

Within each outer training partition, `choose_candidate` (or `choose_v3`) first uses all three inner folds to select the group/family/hyperparameters. Then `crossfit_selected_scores` holds out each of those three inner folds in turn and predicts it from a model fitted on the other two inner folds, with the selected parameters held fixed. `fit_sigmoid` fits a logistic map on the resulting outer-training OOF scores and labels. `choose_threshold` chooses Youden J on those same calibrated training scores. Finally the selected model is refit on all outer-training rows and the learned sigmoid is applied to the outer-test predictions.

## Calibration limitations

**The cross-fit predictions are selection-dependent.** Hyperparameter and feature-group selection uses labels from all outer-training rows, including labels for rows that later appear in the inner validation blocks used to generate the sigmoid-training scores. Thus each base-model score is out-of-fold with respect to the estimator fit, but not independent of the model-selection procedure. The sigmoid and threshold are also fitted/selected on those same training OOF scores.

There is a training-size/score-scale mismatch: each inner scoring model is fitted on roughly two thirds of the outer-training rows, while the final outer predictor is refit on all outer-training rows. For this split, the first outer training partition has 242 rows; inner fits have sizes 161;161;162, inner validation blocks have sizes 81;81;80, and the final model has 242 training rows. Across folds the sizes vary by one row. The saved outer-test calibration results therefore evaluate the practical effect of applying this calibrator to the full-training refit, but they do not remove the calibration construction limitation.

## Outer test use and validity

Outer-test labels are not passed into model selection, preprocessing, sigmoid fitting, or threshold choice. They are used after the model/calibrator/threshold are fixed to produce performance metrics. The runner also uses outer-test features and labels for a post-fit permutation-importance analysis; those results do not feed back into selection. Thus the test set is not literally untouched by all analysis, but its labels do not determine the saved predictions or selected procedure. The paired outer discrimination estimate remains valid as an evaluation of the complete training/selection/calibration procedure represented by the code, subject to source-row independence and the three-repeat limitation.

Fold- and repeat-level calibration intercept/slope are recomputed from outer-test predictions only. The fields `calibration_intercept_training` and `calibration_slope_training` are fitted sigmoid parameters on training OOF scores, not test calibration metrics. The saved outer test estimates reconcile exactly to the independent calculations. These estimates can be unstable for folds of about 60 rows and do not validate a fixed final deployable model.

## Classification

Calibration is **limited, not independently cross-fitted with respect to feature/model selection**. This is a calibration limitation, not a discovered outer ROC-AUC leakage path. The outer folds still assess the whole selection procedure without outer-label selection.
