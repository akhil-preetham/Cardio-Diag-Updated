# CARDIODIAG final completion report

## Completed

- Static audit of all 18 notebooks and final status inventory.
- Author-faithful Five States base execution.
- Corrected validation on Five States and Mendeley.
- Results extraction into JSON and CSV records.
- Conditional bootstrap, calibration, and descriptive Sex subgroup analyses from persisted predictions.
- Hash, environment, runner, dataset, and output manifests.
- New LaTeX manuscript source using only verified results.
- Focused safe Ruff cleanup in `runner/corrected_validation.py`.

## Exact measured results

- Author-faithful Five States: accuracy 0.8922, precision 0.9052, recall 0.9052, F1 0.9052, ROC-AUC 0.9555.
- Corrected Five States: accuracy 0.8783 mean, ROC-AUC 0.9332 mean, PR-AUC 0.9322 mean, Brier 0.0954 mean.
- Corrected Mendeley: accuracy 0.9772 mean, ROC-AUC 0.9964 mean, PR-AUC 0.9970 mean, Brier 0.0182 mean.

## Incomplete

Controlled Python analyses for XAIH-DRC, OVEC, OSEC, LoRAS, LIME, fairness,
clinical utility, and paired statistical comparisons are available in their
machine-readable output directories. The supplementary author notebooks were
not executed as notebooks, the author-faithful OVEC remains blocked by its
source-invalid candidate, and the approximately 70,000-row benchmark dataset
is unavailable. The exact paper tables and figures could not be extracted with
available local tools. No unsupported paper comparison is reported.

## Quality status

Tests passed (`8 passed`) and Python compilation passed. New implementation and
experiment files pass Ruff; the broader legacy runner set retains 38
pre-existing findings. Execution gates are disabled.

## Manuscript

`paper/manuscript.tex` is a new source manuscript. It contains no invented authorship, affiliation, funding, ethics approval, or publication claim. PDF compilation was not available because no LaTeX executable was installed.

## Final status

`IMPLEMENTATION COMPLETE — EXTERNAL DATA OR VALIDATION STILL REQUIRED`
