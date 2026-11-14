# CARDIODIAG Version 2 Results Update

This is a separate Version 2 results update. The Version 1 manuscript was not modified.

The corrected experiment used the persisted Stage 6 nested split manifest exactly:
five repeats, five outer folds, and four inner folds. The target was `Cath` with the
strict features `Age`, `Sex`, `Current Smoker`, `EX-Smoker`, and `FH`.

All nine Version 1 model/imbalance configurations were retained. Sigmoid calibration
was fit to cross-fitted inner-training predictions only. SVM decision scores were
retained as scores and were not evaluated as native probabilities using Brier score or
log loss.

The highest descriptive calibrated outer-fold ROC-AUC was:

- Logistic Regression, balanced class weighting: **0.735 +/- 0.060**
- Logistic Regression, unweighted: **0.735 +/- 0.061**

These are repeated internal-validation summaries, not external validation, population
confidence intervals, clinical utility evidence, or prospective prediction. Comparisons
are descriptive paired analyses without confirmatory p-values. The source derivative has
no patient identifier, so patient-level grouping cannot be independently verified.
