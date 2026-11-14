# Implementation audit

## Module findings

| Area | Evidence | Assessment |
|---|---|---|
| Data loading | `src/cardiodiag/data/loading.py`, `runner/ensemble_experiments.py` | Targets are dataset-specific and identifier-like columns are excluded in controlled runners. |
| Preprocessing | `src/cardiodiag/preprocessing/pipeline.py` | Corrected transformer uses median numeric imputation, scaling, most-frequent categorical imputation, and one-hot encoding. |
| Imbalance | `src/cardiodiag/imbalance/sampling.py`, `runner/dataset_integrity_loras.py` | LoRAS/SMOTE comparison exists; corrected runner places SMOTE inside the training pipeline. |
| XAIH-DRC | `src/cardiodiag/feature_selection/xaih_drc.py` | A guarded selector exists, requires supplied training-fold LIME importances, and uses correlation/LIME thresholds plus PCA. It is not proof of full paper equivalence. |
| OVEC/OSEC | `src/cardiodiag/ensembles/classifiers.py`, `runner/ensemble_experiments.py` | Source-shaped classifiers and controlled runs exist. The author-faithful OVEC failure is preserved; compatibility adaptation is separate. |
| Evaluation | `src/cardiodiag/evaluation/metrics.py` | Broad classification metrics are implemented and persisted. |
| Statistics | `src/cardiodiag/statistics/paired.py` | Paired bootstrap and DeLong code require aligned vectors; saved artifact provenance must be reconciled. |
| Explainability | `src/cardiodiag/explainability/lime_workflow.py` | LIME workflow exists, but explanation/model/input provenance requires a single authoritative registry. |
| Reliability/utility | `src/cardiodiag/calibration/metrics.py`, `src/cardiodiag/clinical_utility/decision_curve.py` | ECE/MCE and net benefit are implemented as descriptive diagnostics. |
| Fairness | `src/cardiodiag/fairness/subgroups.py`, `runner/fairness_ensemble.py` | Descriptive subgroup metrics with minimum-size filtering; no certification. |

## Important correctness observations

- `runner/ensemble_experiments.py::_split` intentionally preserves the source order for author-faithful and compatibility tracks: transformed data and SMOTE are applied before splitting. This is a methodological risk, not an accidental correction.
- The corrected branch splits first and fits the estimator pipeline on training data. Its search is performed using training-only transformed data.
- `XAIHDRCSelector.fit` correctly refuses absent LIME importances, but it is a component, not an integrated fold-local selector in the corrected ensemble runner.
- The codebase has focused tests for metrics, preprocessing construction, XAIH-DRC behavior, paired statistics, LoRAS, and runner configuration. Tests do not establish paper-level equivalence or external validity.

## Missing or incomplete evidence

- Historical dependency lockfile and exact author environment are unavailable.
- The approximately 70,000-row benchmark dataset is absent.
- Exact paper tables/figures could not be extracted locally.
- No complete end-to-end nested, patient-grouped, independent external evaluation is present.

## Severity

- **Critical:** author-faithful result is not leakage-controlled.
- **High:** OVEC exact path is blocked; statistical provenance is contradictory.
- **Medium:** specialized-method equivalence and fold-local XAIH-DRC integration are not fully demonstrated.
- **Low:** documentation and registry fragmentation.
