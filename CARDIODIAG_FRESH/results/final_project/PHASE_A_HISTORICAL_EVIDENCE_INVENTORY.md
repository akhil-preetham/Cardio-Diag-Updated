# Phase A — Historical evidence inventory

Scope: source files, frozen protocol records, predictions, result tables, manifests and prior independent audits in the local CARDIODIAG_FRESH workspace. No historical training run was repeated.

| Phase | Evidence inspected | Status for this completion pass | Authority/qualification |
|---|---|---|---|
| Phase 1 / author release | notebook inventories, execution report, outputs, prepared-run manifest | Existing run retained; not rerun | Author-faithful output is a reproduction artifact; code reports pre-split SMOTE, test-set-driven stopping and invalid candidates. |
| Corrected comparator | corrected prediction files, reports, manifest | Historical outputs retained | Repeated holdouts overlap; Five States and Mendeley endpoints are separate. |
| Phase 2 | eligibility, protocol and QA reports | Existing evidence reviewed | ZAS pre-diagnostic timing is not documented feature-by-feature. |
| Phase 3 | acquisition, compatibility and external-validation reports | Existing evidence reviewed | No identified compatible untouched cohort for frozen V3. |
| Phase 4 | four-centre protocol, configurations, predictions, reports and source manifest | Existing evidence reviewed; output has one material audit defect | UCI stages are internal-external centre-held-out tasks, not external validation of ZAS. Duplicate sensitivity uses a non-global row key. |
| Phase 4.1–4.3 | staged predictions, paired comparisons, center metrics, duplicate report, independent audit manifest | Metrics and key designs inspected | Pair keys include centre and source row. Duplicate sensitivity counts and exclusion logic are defective as detailed in the metric report. |
| Phase 5 | ZAS/UCI runners, result tables, predictions, data registry, reliability, reports and output manifest | ZAS metric and all 40 declared output hashes reconciled | ZAS benchmark contains duplicate Logistic/Ridge model labels. Phase 5 UCI prediction metadata calls validation `stratified_inner` despite the runner using a fixed-parameter centre-held-out fit with no inner CV. |
| Phase 5.1 | timing/leakage runner and four generated audit artifacts | Findings partly supported; documentation needs correction | The audit runner overwrites fixed outputs on rerun; its “read-only” description is inaccurate. It says “patients” without verified IDs and calls age/sex confirmed based on plausibility. |
| Phase 5.1.1 | No separately named runner, report or directory found | No discrete implementation located | The Phase 5.1.1 questions in the user's completion brief were answered from existing predictions/code below; this does not prove a previous formal 5.1.1 run occurred. |

## Phase 5.1.1 metric reconstruction

For each of `eligible_combined/logistic` and `eligible_combined/ridge`, the prediction file contains 606 rows (two repeated outer-test predictions for each of 303 source rows). Averaging by source row gives 303 rows, 216 positive labels, no label disagreements and two repeats per row. Recomputed metrics match `ZAS_MODEL_BENCHMARK.csv`: ROC-AUC 0.9282141337, PR-AUC 0.9668302657, Brier 0.0964242607 and log loss 0.3141080938.

The logistic and ridge saved probabilities are exactly identical at every `(repeat_fold, row_index)` key. In `runner/phase5/common/models.py`, both labels construct the same L2 `LogisticRegression` with the same C grid. Thus there are 36 result rows but no more than 30 distinct label/configuration combinations after deduplicating those identical aliases; other families remain separate estimator classes.

No code selects a unique winner: the runner enumerates six groups and six model labels, then writes all 36 benchmark rows. Choosing 0.928 after reviewing that table is post hoc. Hyperparameter selection is nested within each candidate; group/model selection is not nested. No model-development-level optimism correction is present.

## Historical source identity

- The frozen V3 experiment configuration lists exactly Age, Sex, Current Smoker, EX-Smoker and FH and preserves five outer and four inner folds with fold-specific nested sigmoid calibration.
- V3's report states the highest descriptive calibrated ROC-AUC as 0.735 ± 0.060. This is internal validation on source rows; the dataset supplies no patient identifier.
- Phase 4.3 reports 3,680 UCI stage predictions across four stages and four centres, with paired comparisons keyed by `(held_out_centre, source_row)`.
- Phase 5 output manifest: 40 declared outputs, 40 found, 0 SHA-256 mismatches at inspection.

## Files to consult

- `results/primary_research_v3/EXPERIMENT_CONFIGURATION.json`
- `results/primary_research_v3/PRIMARY_RESEARCH_PROTOCOL_EXECUTION_V3.md`
- `results/phase4_validation_audit/PHASE4_3_FINAL_REPORT.md`
- `results/phase5/manifests/PHASE5_MANIFEST.json`
- `results/phase5/zas_track/ZAS_NESTED_VALIDATION_PREDICTIONS.csv`
- `results/phase5/zas_track/ZAS_MODEL_BENCHMARK.csv`
