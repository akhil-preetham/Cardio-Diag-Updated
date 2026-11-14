# CARDIODIAG project completion status

## Executed evidence

### Author-faithful base

- Notebook: `reference/reproduction_runner/work/author_faithful/base_reproduction/Daza_Paper_Exact_Accuracy.ipynb`
- Executed output: `reference/reproduction_runner/author_faithful/executed_notebooks/Daza_Paper_Exact_Accuracy_a788cef8a6_executed.ipynb`
- Dataset: `reference/author_release/heart5.csv`
- Dataset SHA-256: `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2`
- Metrics: accuracy 0.8922, precision 0.9052, recall 0.9052, F1 0.9052, ROC-AUC 0.9555
- Interpretation: executed in the current environment with the released methodological behavior; not leakage-controlled and not proof of historical environment recovery.

### Corrected validation

- Five States: mean accuracy 0.8783, balanced accuracy 0.8755, ROC-AUC 0.9332, PR-AUC 0.9322, Brier 0.0954.
- Mendeley: mean accuracy 0.9772, balanced accuracy 0.9759, ROC-AUC 0.9964, PR-AUC 0.9970, Brier 0.0182.
- Ten repeated holdouts were used for each dataset. These repeats overlap and are not independent external validation cohorts.

## Separate tracks

The compatibility-adapted OVEC/OSEC track was executed through separate controlled
Python runners. The final OVEC rerun uses the recovered `n_iter=2` count and
records the finite `n_estimators=2` value as an engineering adaptation. The
supplementary author notebooks themselves were not executed; their equivalent
controlled analyses are not historical notebook reproduction.

## Methods and claims

The released source uses SMOTE before splitting, non-nested model selection, and
test-set-driven stopping. The corrected comparator addresses these concerns but
is a different experiment. OVEC/OSEC, XAIH-DRC, LoRAS, LIME, bootstrap/statistical
comparisons, fairness, and clinical utility were validated in separate controlled
analyses with documented limitations.

## Software and quality

The `.venv-repro` environment and current `.venv` inventory are recorded. Tests passed (`8 passed`) and Python compilation passed. New implementation and
experiment files pass Ruff; the broader legacy runner set retains 38 pre-existing
findings.

## Data and manuscript limitations

Only Five States and Mendeley data were present. The requested approximately 70,000-row Kaggle dataset was absent. A new `paper/manuscript.tex` was created from verified project results; it contains no invented authorship or publication claim. The available PDF could not be fully text-extracted with installed local dependencies, so no exact paper table value was invented.

## Safety state

`config/runner_config.json` now has `notebook_execution.enabled=false` and `approval_granted=false`. The allowlist remains restricted to the prepared working copy. Original and prepared notebook artifacts remain preserved.

## Status

`IMPLEMENTATION COMPLETE — EXTERNAL DATA OR VALIDATION STILL REQUIRED`
