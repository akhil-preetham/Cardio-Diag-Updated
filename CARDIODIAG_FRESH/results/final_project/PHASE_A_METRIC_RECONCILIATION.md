# Phase A — Metric reconciliation

## ZAS Phase 5 eligible-combined

Source: `results/phase5/zas_track/ZAS_NESTED_VALIDATION_PREDICTIONS.csv`. Predictions were grouped by `(group, model, row_index)`, repeated probabilities averaged per source row, and source labels required to agree.

| Label | Outer prediction rows | Aggregated source rows | Events | ROC-AUC | PR-AUC | Brier | Log loss | Reconciles to benchmark |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| logistic | 606 | 303 | 216 | 0.928214 | 0.966830 | 0.096424 | 0.314108 | Yes |
| ridge | 606 | 303 | 216 | 0.928214 | 0.966830 | 0.096424 | 0.314108 | Yes |

No source patient ID exists; “303 source rows” is the verified unit. The rows cannot be asserted to be 303 unique patients based solely on this CSV.

## UCI Phase 4 stage results

The Phase 4.3 independent audit reports 920 records per stage, four centres and 3,680 saved predictions in total. The pooled out-of-centre results are:

| Stage | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Stage 1 basic risk | 0.717900 | 0.729290 | 0.211205 | 0.611668 |
| Stage 2 symptoms | 0.822525 | 0.824077 | 0.168070 | 0.510644 |
| Stage 3 resting measurements | 0.821505 | 0.819088 | 0.168381 | 0.513922 |
| Stage 4 exercise test | 0.845033 | 0.864469 | 0.158443 | 0.486086 |

Phase 4.3's paired Stage 3-to-Stage 4 comparison uses the same 920 `(centre, source_row)` pairs. Differences (Stage 4 minus Stage 3) were ROC-AUC +0.02353 (record-bootstrap 95% interval +0.00608 to +0.04426), PR-AUC +0.04538 (+0.01960 to +0.07062), Brier -0.00994 (-0.01861 to -0.00151), and log loss -0.02784 (-0.05139 to -0.00562). Intervals resample records, not centres or verified patient clusters.

### Duplicate-sensitivity correction required

The source cohort has two within-centre duplicate profiles: one later occurrence at Hungary source row 102 and one at VA Long Beach source row 187. The saved sensitivity table's “drop duplicates” scenario reports `n=913` pooled and `excluded_n=0`. The runner forms `exclusions` from row numbers alone, then applies those numbers to all centres even though row numbers repeat by centre; it also calculates `excluded_n` after removing those rows, so the count is always zero. This incorrectly drops five additional records from other centres beyond the two later duplicate occurrences. The correct paired exclusion identity is `(centre, source_row)`, giving 918 records if only the two later within-centre duplicates are omitted. Historical metrics are left untouched; the contaminated sensitivity scenario is not used to support robustness claims.

The saved Phase 4.3 base metrics and paired stage comparisons are separate from that incorrect sensitivity subset and their source-row/centre pairing is valid. The duplicate-sensitivity robustness claim itself is not accepted without a versioned correction.

## Prior ensemble paired-statistics documentation

`docs/EXPERIMENT_RESULTS.md` says paired tests were not performed for the corrected repeated-holdout models because models lacked common persisted observations. `docs/FINAL_EXECUTION_VERIFICATION.md` reports Wilcoxon/DeLong/paired bootstrap/Holm analyses for a different, fixed-holdout OVEC/OSEC comparison, and `paired_model_comparisons.csv` contains four such comparisons. These statements are scope-specific, not contradictory; the older wording should say “not performed for the ten-run corrected comparator.”
