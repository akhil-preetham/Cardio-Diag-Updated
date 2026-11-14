# Reusable implementation status

## Implemented and verified

- Dataset loading, schema checks, duplicate-row counts, identifier detection, and SHA-256 recording.
- Fold-safe preprocessing with median imputation, standardization, and one-hot encoding.
- An `imbalanced-learn` pipeline helper that places SMOTE after preprocessing and therefore inside training folds.
- Source-shaped OVEC (`VotingClassifier`) and OSEC (`StackingClassifier`) builders.
- Source-package LoRAS wrapper (`loras==0.0.22b0`) with training-only usage.
- XAIH-DRC correlation/LIME top-k thresholding, XOR PCA branch, and leakage guard.
- Explicit model-search helpers for training data supplied by the caller.
- Classification, calibration, paired bootstrap, subgroup, and decision-curve utilities.
- Optional LIME integration that fails explicitly when the dependency is unavailable.

## Reproduction boundaries

XAIH-DRC requires training-fold global LIME importances as an explicit input;
it will not compute importances from the full dataset. LoRAS requires the
source-used `loras` package and preserves its majority/minority API and
optional downsampling behavior. Both must be called only on training data.

The OVEC/OSEC builders reproduce the visible sklearn ensemble shapes, but
their exact notebook-specific estimator lists and search spaces must be
supplied by each experiment configuration. The optimization helpers use only
the caller-provided training data; they do not discover or execute notebooks.

## Verification

The focused suite passes with 8 tests. Python bytecode compilation of the new
`src` tree passes. No notebook or kernel was started. The optional `lime` and
source-used `loras` packages were installed only in `.venv-repro` so their
controlled analyses could execute; `.venv` was not modified.
