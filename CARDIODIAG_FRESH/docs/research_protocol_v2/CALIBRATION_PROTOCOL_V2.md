# Calibration Protocol v2

## Primary rule

For every model, hyperparameters are selected inside outer-training data. A calibration
method is then fit using only outer-training data and inner calibration folds. Outer test
labels are used only once for evaluation.

Native probabilities, decision scores, and calibrated probabilities are separate output
types. For SVM, `decision_function` is not a probability. It must either remain a ranking
score or be transformed by a calibration model trained without outer test labels.

## Methods

Sigmoid/Platt calibration is the primary method because the cohort is small. Isotonic
calibration is exploratory only and requires a predeclared minimum calibration sample
and monotonicity check; it must not be selected after viewing outer results.

## Metrics

Primary calibration metrics are held-out Brier score and log loss, with calibration
intercept and slope as complementary summaries. Reliability diagrams use fixed bins
chosen before examining outer labels. Ten-bin ECE is not a primary metric because outer
folds contain approximately 60 observations; use fewer bins or adaptive minimum-count
bins as a prespecified sensitivity analysis.

Calibration quality is distinct from discrimination. A high ROC-AUC does not establish
trustworthy probabilities or clinical utility.
