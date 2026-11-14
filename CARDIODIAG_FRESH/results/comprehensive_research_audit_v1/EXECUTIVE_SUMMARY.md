# CARDIODIAG comprehensive research audit

## Overall status

**Condition: substantial implementation and evidence infrastructure, but not publication-ready for external-validity or clinical claims.** The repository preserves separate author-faithful, compatibility-adapted, and corrected tracks; records source hashes; has executable runners and tests; and contains saved predictions/models for the controlled ensemble experiments. The evidence supports a reproducibility study and a leakage-controlled comparator, not an exact historical reproduction or independent clinical validation.

## Completed evidence

- Eighteen extracted notebooks were inventoried in `reference/reproduction_runner/audit/notebook_inventory.json`.
- The prepared author-faithful Five States notebook executed in 404 seconds with accuracy 0.8922 and ROC-AUC 0.9555.
- Corrected repeated validation used ten stratified 75:25 holdouts, fold-local preprocessing, training-fold SMOTE, and ROC-AUC selection.
- Corrected mean ROC-AUC was 0.9332 for Five States and 0.9964 for Mendeley (`patientid` excluded).
- Controlled OVEC/OSEC, LoRAS, XAIH-DRC, LIME, calibration, subgroup, decision-curve, and statistical artifacts exist, but their evidentiary status differs and is documented below.
- The duplicate audit found no exact duplicates, predictor duplicates, near-duplicate candidates, repeated Mendeley IDs, or train/test duplicate overlap in the reproduced holdouts.

## Critical limitations

1. The author-faithful notebook applies preprocessing/label encoding and SMOTE before the 80:20 split and uses test-set-driven stopping. Its result is a faithful execution artifact, not an unbiased estimate.
2. The author-faithful OVEC path fails on the recovered `RandomForestClassifier(n_estimators=None)` candidate. The finite-parameter rerun is compatibility-adapted and must not be reported as exact reproduction.
3. Historical package/environment equivalence and exact paper table extraction remain unavailable.
4. Repeated holdouts overlap; they are not independent cohorts or external validation.
5. `docs/EXPERIMENT_RESULTS.md` says paired tests were not performed, while `docs/FINAL_EXECUTION_VERIFICATION.md`, `results/final_independent_verification.json`, and `results/ensemble_postprocessing/paired_model_comparisons.csv` say they were completed. This is a material provenance/documentation contradiction.
6. Fairness, calibration, decision-curve, and LIME artifacts are descriptive internal analyses, not evidence of fairness certification, causal explanation, clinical utility, or deployment readiness.

## Recommended research direction

Use the corrected pipeline as the enhanced track, prespecify a nested/group-aware evaluation protocol, reconcile the statistical-artifact contradiction, and obtain the missing benchmark dataset and an independent external cohort. The defensible contribution is a transparent leakage-controlled re-evaluation of the published method plus a reproducible comparison of source-shaped ensembles, not a claim that the original method is clinically validated.

## First implementation task

Create one authoritative experiment registry that maps every reported metric to a source prediction file, row-index manifest, protocol, and status (`executed`, `static`, `blocked`, or `superseded`). Resolve the paired-statistics contradiction before any new optimization.
