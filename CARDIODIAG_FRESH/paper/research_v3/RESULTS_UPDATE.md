# CARDIODIAG Version 3 Results Update

Version 3 is a separately versioned fully nested calibration correction. It preserves
the Version 1 and Version 2 experiments and uses the exact Stage 6 outer and inner split
manifest.

For each held-out inner calibration fold, hyperparameters were selected using only the
other three inner folds. The selected pipeline then generated scores for the held-out
calibration fold. A sigmoid calibrator was fit to the collected out-of-fold scores.
Final outer-test scores came from a single estimator selected and refitted on the full
outer-training partition. SVM decision scores were not treated as native probabilities.

The highest descriptive fully nested calibrated outer-fold ROC-AUC was logistic
regression with balanced class weighting: **0.735 +/- 0.060**. The unweighted logistic
regression result was **0.735 +/- 0.061**. These are repeated internal-validation
summaries, not external validation, population confidence intervals, prospective
prediction, or clinical utility evidence.

The fully nested design removes the V2 calibration-fold hyperparameter selection
interaction. Remaining limitations include small-sample calibration, score-scale
differences between fold-specific calibration models and the final refitted estimator,
repeated-participant validation, unavailable patient identifiers, and conditional rather
than population-level uncertainty summaries.
