# Independent Experimental Verification Report v1

## Overall classification

The implementation is **correct with limitations**, not publication-ready by execution alone.

### Verified correct

- Outer validation structure: five repeats, five stratified outer folds, deterministic seeds, no train/test overlap.
- Fold-local preprocessing and inner hyperparameter search.
- No outer-test labels used to fit models, select hyperparameters, calibrate, or select thresholds.
- Nine model/imbalance configurations and zero failed model folds.
- Predictions are complete and independently recomputed metrics match `6300/6300` saved metric values.
- No external validation, prospective prediction, or clinical utility claim is made.

### Correct with limitations

- Sigmoid calibration is outer-test isolated but reuses outer-training observations used in inner model selection.
- Bootstrap intervals use aggregated repeated-CV row predictions and are not population confidence intervals.
- Held-out permutation importance is valid as post-hoc explanation but not causal evidence.
- Ten-bin ECE is unstable at this sample size.

### Requires correction in interpretation

- SVM uncalibrated values are sigmoid-transformed decision scores, not native probabilities.
- Statistical comparisons are descriptive paired bootstrap summaries, not full dependent-CV hypothesis tests.
- Inner-fold assignments were not persisted.

### Unresolved

- Patient-level grouping cannot be independently verified because the modeling derivative has no patient identifier.
- Broader feature timing and measurement availability remain unresolved; the strict five-feature set is appropriately conservative.

## Primary result verification

The saved best calibrated configuration is `logistic_regression_none` with mean outer ROC-AUC
`0.737` and SD `0.059`. This ranking is descriptive
and must not be interpreted as an independently selected final clinical model.

## Outputs

All findings and recomputations are stored in this verification directory. The original
primary experiment directory and manuscript were not modified.
