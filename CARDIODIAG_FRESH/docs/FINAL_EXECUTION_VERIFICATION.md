# Final execution verification

## Executed components

| Component | Implemented | Tested/executed | Evidence |
|---|---:|---:|---|
| OVEC | Yes | Compatibility-adapted exact-count rerun and corrected tracks; author-faithful failed on invalid `n_estimators=None` | `results/ensemble_experiments_compat_v2/*/*/ovec/` and `results/ensemble_experiments/*/*/ovec/` |
| OSEC | Yes | All datasets and tracks completed | `results/ensemble_experiments/*/*/osec/` |
| LoRAS | Yes, source package `0.0.22b0` | Training-only Five States and Mendeley comparison | `results/data_integrity_and_loras/dataset_integrity_loras_report.json` |
| XAIH-DRC | Yes | Training-only LIME importance and recovered threshold/PCA procedure | `results/xaih_drc_experiments/xaih_drc_manifest.json` |
| LIME | Yes | Representative positive/negative explanations for persisted models | `results/lime_experiments/` |
| Paired statistics | Yes | Paired bootstrap, Wilcoxon, DeLong, Holm correction | `results/ensemble_postprocessing/paired_model_comparisons.csv` |
| Calibration/utility | Yes | Ten-bin calibration and threshold sweep 0.05--0.95 | `results/ensemble_postprocessing/` |
| Subgroup analysis | Yes | Descriptive Sex/gender subgroup metrics, minimum n=20 | `results/ensemble_postprocessing/ensemble_subgroup_performance.csv` |

## Key ensemble results

| Dataset | Track | Method | Accuracy | ROC-AUC |
|---|---|---|---:|---:|
| Five States | corrected | OVEC | 0.8957 | 0.9421 |
| Five States | corrected | OSEC | 0.9174 | 0.9424 |
| Mendeley | corrected | OVEC | 0.9760 | 0.9982 |
| Mendeley | corrected | OSEC | 0.9840 | 0.9995 |

These are single fixed corrected holdouts, not independent external validation.
The previously executed ten-repeat corrected-validation results remain the
primary repeated comparator.

The final compatibility OVEC rerun used the recovered source `n_iter=2`
search count and the finite `n_estimators=2` engineering adaptation. It
achieved Five States accuracy/ROC-AUC 0.8725/0.9430 and Mendeley
0.9871/0.9987. Earlier compatibility outputs using `n_iter=10` are preserved
but superseded for source-count comparison.

## Verification checks

- Full tests: 8 passed.
- Ruff: passed for `src/cardiodiag/` and all new experiment runners.
- Python compilation: passed for `src`, `runner`, and `tests`.
- Saved prediction metrics were independently recalculated and matched within serialized JSON precision.
- Persisted models were reloaded and produced identical predictions.
- Notebook execution was not used for these experiments; the notebook runner remains disabled.
- Original datasets and author notebooks were not modified.
- Independent machine-readable verification: `results/final_independent_verification.json`.
- Consolidated ensemble comparison: `results/ensemble_comparison.csv`.

## Remaining limitations

- Author-faithful OVEC cannot complete in the current environment because the
  recovered source search space contains invalid `RandomForestClassifier`
  candidates.
- Fairness results remain descriptive subgroup summaries, not certification.
- The approximately 70,000-row benchmark dataset remains unavailable.
- Historical package equivalence and exact paper table extraction remain unresolved.

**Final status: IMPLEMENTATION COMPLETE — EXTERNAL DATA OR VALIDATION STILL REQUIRED**
