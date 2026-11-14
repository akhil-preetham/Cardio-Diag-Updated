# Phase 6 implementation audit

## Calibration metadata count

The runner has a three-fold inner design (`StratifiedKFold(n_splits=3)`), and `crossfit_selected_scores` iterates exactly over those three splits. However, the reporting list `calibration_selection` (and its V3 analogue) is constructed with `range(1, 5)`, creating four metadata entries with calibration_fold labels 1–4. The fourth entry duplicates the same selected group, family and parameters; it does not correspond to a score block or a fourth model fit. `calibration_crossfit_rows` and the actual calibration scores are produced from the three loop iterations.

**Finding: metadata/reporting error only.** The saved four-entry JSON does not drive calibration, outer probability generation, threshold selection, metrics, or candidate selection. `CALIBRATION_SELECTION_METADATA_CORRECTED.csv` supplies a corrected three-row display for each of the 30 outer model records and explicitly records that selection occurred once over all three inner folds rather than once per calibration fold. Historical Phase 6 CSVs remain untouched.

## Split, selection and output checks

- Frozen split manifest: 15 outer partition/model pairs verified over the first three of five repeats.
- Unique source rows per repeat/model: 303; original target labels match: all verified.
- Feature-group/model metadata agrees between predictions and outer selections: all verified.
- Phase 6 result/output and input hashes: 15 outputs and 6 inputs checked; all match: True.
- The original Phase 6 output manifest itself is not self-hashed; this audit verifies every file enumerated by that manifest.

No predictive-classifier fitting, hyperparameter search, threshold optimization or Phase 6 command was run by this audit. Logistic regressions were fit only to recompute calibration intercept/slope as evaluation metrics from saved outer predictions; these fits did not produce predictions or influence selection.

## Project environment

The protocol commands invoke the project-local `.venv\Scripts\python.exe`. Its runtime resolves `sys.executable` and `sys.prefix` to `CARDIODIAG_FRESH\.venv`, with base Python 3.11.16. Verified versions: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, pytest 9.2.0, and Ruff 0.17.4. `requirements.txt` and `requirements-runner.txt` contain ranges/unpinned packages rather than a lock file; these observed virtual-environment versions are the exact versions used here. No installation was performed.
