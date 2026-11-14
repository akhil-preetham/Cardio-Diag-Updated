# Correction and Acceptance Decisions

## Overall classification: Requires correction before final calibrated baseline acceptance

V2 is a valid and reproducible internal-validation artifact for discrimination and descriptive probability analysis, and its outer-test isolation is verified. It should not yet be accepted as the final calibrated baseline because the calibration-training predictions use hyperparameters selected with the same calibration fold labels included in model selection.

## Required correction

Use a fully nested calibration design: for each calibration fold, select hyperparameters using only the other inner-training partitions, fit that selected pipeline without the calibration fold, generate the calibration score, and fit the sigmoid calibrator only after all such scores are assembled. Preserve the outer test set for final evaluation. Do not reinterpret the current V2 outputs as fully nested calibration results.

The current metrics are independently reproduced. Statistical comparisons and uncertainty remain descriptive, and feature importance does not support causal or substantive clinical conclusions.
