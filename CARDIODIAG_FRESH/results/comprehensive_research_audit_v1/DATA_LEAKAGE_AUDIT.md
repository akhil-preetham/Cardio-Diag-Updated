# Data leakage audit

## Severity-ranked findings

### Critical — author-faithful pre-split resampling and preprocessing

**Evidence:** `runner/ensemble_experiments.py::_split`; `results/experiment_metrics.json`; `reference/reproduction_runner/author_faithful/`. The source-shaped track transforms data and applies SMOTE before the 80:20 split. The prepared notebook also records test-set-driven stopping.

**Impact:** Synthetic samples and fitted transformations can incorporate information from observations later assigned to test. Test-set stopping makes the reported result selection-biased. The 0.8922/0.9555 Five States author-faithful metrics are execution evidence, not unbiased generalization estimates.

**Correction:** Preserve the source track as-is and report it as methodologically risky. Use fold-local preprocessing, resampling, selection, and a single untouched evaluation boundary in the enhanced track. Rerun corrected experiments if the protocol changes.

### High — repeated holdout dependence

**Evidence:** `runner/corrected_validation.py`; corrected reports and `docs/EXPERIMENT_RESULTS.md`.

**Impact:** Ten seeds reuse observations across runs. Means and standard deviations describe split sensitivity, not ten independent patient cohorts. Naive inference treating runs as independent is invalid.

**Correction:** Report per-observation or grouped nested evaluation, and describe repeated holdouts as dependent resampling diagnostics.

### High — statistical-artifact provenance contradiction

**Evidence:** `docs/EXPERIMENT_RESULTS.md` says DeLong/Wilcoxon/paired comparisons were not performed, while `docs/FINAL_EXECUTION_VERIFICATION.md`, `results/final_independent_verification.json`, and `results/ensemble_postprocessing/paired_model_comparisons.csv` state they were completed.

**Impact:** Readers cannot determine which claims and artifacts are authoritative.

**Correction:** Reconcile the registry and manuscript before publication; verify identical row ordering and prediction-vector provenance for every comparison.

### Medium — identifier/group leakage not established for all historical tracks

**Evidence:** `patientid` is excluded in corrected runners; Mendeley has no repeated IDs in the duplicate audit. Historical notebook paths and track-specific data handling are not uniformly documented as grouped splits.

**Impact:** Unique IDs do not prove patient independence in external or transformed sources.

**Correction:** Make identifier policy explicit in every manifest and use patient/group splits whenever repeated measurements are possible.

## Negative findings

No exact, predictor-only, near-duplicate, or reproduced train/test duplicate overlap was found in the supplied datasets. These negative findings do not establish absence of other leakage mechanisms.
