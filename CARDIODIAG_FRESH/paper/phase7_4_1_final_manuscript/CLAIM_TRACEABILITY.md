# Claim and number traceability

All local references are relative to the project root `CARDIODIAG_FRESH`. Phase 6.1 is the independent arithmetic/prediction audit; underlying saved results remain numerical sources.

| Manuscript claim | Exact evidence source | Location / interpretation |
|---|---|---|
| ZAS n=303; 216 Cad and 87 Normal; prevalence 0.712871 | `data/interim/validated/z_alizadeh_sani.csv`; `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv`; `results/phase6_zas_selection/PHASE6_MANIFEST.json` | Target is `Cath`, map Cad=1/Normal=0; source rows, not verified unique patients. |
| Official ZAS threshold >=50%; UCI catalog says 54 features; local CSV has 56 columns | UCI [Z-Alizadeh Sani page](https://archive.ics.uci.edu/dataset/412/z); `results/final_project/FINAL_RESEARCH_DECISION.md`; `results/phase5_1_audit/ZAS_FEATURE_TIMING_AUDIT.md` | Schema discrepancy is disclosed unresolved. UCI metadata source verified 2026-11-14. |
| Phase6 candidate pool 6 groups × 5 unique families; duplicate Logistic/Ridge removed | `runner/phase6_zas_selection/run.py` lines 46–60; `results/phase6_zas_selection/PHASE6_MANIFEST.json` | `historical_0928_used_for_selection:false`. |
| 3 repeats × 5 paired outer folds; 3 inner folds; first three frozen V3 partitions | `results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`, “Split and nested selection”; `results/phase6_zas_selection/PHASE6_MANIFEST.json` | 15 outer partitions. The 25-fold plan exceeded 35 CPU minutes; the reduced design is reported openly. |
| Abstract AUCs 0.9134206, 0.8999042, 0.9176245; V3-style 0.7370158, 0.7419647, 0.7299649 | `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv`; independently `results/phase6_1_audit/RECONSTRUCTED_REPEAT_METRICS.csv` | Each metric is from a complete repeat-specific 303-source-row OOF set, not fold-average AUC. Rounded abstract: 0.9134, 0.8999, 0.9176 and 0.7370, 0.7420, 0.7300. |
| Paired repeat AUC differences +0.1764049, +0.1579395, +0.1876596; mean +0.1740013 | `results/phase6_zas_selection/PAIRED_REPEAT_DIFFERENCES.csv`; `results/phase6_zas_selection/PERFORMANCE_COMPARISON_SUMMARY.csv`; independent `results/phase6_1_audit/RECONSTRUCTED_PAIRED_REPEAT_DIFFERENCES.csv` | Paired on exact same outer repeat partitions. No p-value/significance claim. |
| Split-variation interval +0.1367266 to +0.2112761 | `results/phase6_zas_selection/PERFORMANCE_COMPARISON_SUMMARY.csv`; `results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`, “Performance and uncertainty” | Student t interval over 3 repeat-level differences; overlapping source rows; not patient/population CI. |
| Phase6 secondary mean AP .957291/.846862, Brier .107616/.174605, log loss .346740/.528566 | `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv` (three rows per procedure) | Means calculated across repeat summaries for description; no pooled prediction analysis or inference. AP baseline prevalence 0.712871. |
| Selection group `eligible_combined` 15/15; model families 9/15, 4/15, 2/15 | `results/phase6_zas_selection/MODEL_SELECTION_FREQUENCY.csv`; `OUTER_MODEL_SELECTIONS.csv`; `ROBUSTNESS_REPORT.md` | Counts of 15 outer selection fits; group inclusion does not establish every feature’s necessity. |
| Phase 6 group/family selection frequencies (15 outer fits) | `results/phase6_zas_selection/OUTER_MODEL_SELECTIONS.csv`; `MODEL_SELECTION_FREQUENCY.csv`; `PHASE6_MANIFEST.json` | Summarized in `TABLES/TABLE6_SELECTION_FREQUENCY.csv`; broad group frequency is not individual feature necessity. |
| Permutation top feature Typical Chest Pain decrease .125 (SD .050), Age .042 (.017), DM .019 (.011) | `results/phase6_zas_selection/ROBUSTNESS_REPORT.md`, “Outer-fold variability and feature dependence”; underlying `FEATURE_PERMUTATION_IMPORTANCE.csv` | Descriptive marginal importance using outer test after model frozen; not causal and not selection. |
| Age/Sex-only AUC range .726–.731 | `results/phase6_zas_selection/DEMOGRAPHIC_ONLY_SENSITIVITY.csv`; `ROBUSTNESS_REPORT.md` | Timing-uncertain sensitivity, not eligible-confirmed baseline. |
| Phase5 exploratory metrics .928214/.966830/.096424/.314108; 36 rows | `results/phase5/zas_track/ZAS_MODEL_BENCHMARK.csv`; `results/phase5_1_audit/ZAS_0928_EVIDENCE_DECISION.md` | Selected post hoc. Never presented as unbiased primary estimate. |
| UCI n=920; centres n=303/294/123/200; events 139/106/115/149 | `results/phase4_uci_multicentre/CENTRE_COHORT_PROFILE.csv`; `results/phase4_improvement/IECV_FINAL_PREDICTIONS.csv`; Phase 4.3 report | Counts from reconstructed source and saved predictions. Switzerland n=123, 115 events, 8 negatives. |
| UCI Stage pooled ROC-AUC .717900/.822525/.821505/.845033; AP .729290/.824077/.819088/.864469; Brier .211205/.168070/.168381/.158443; log loss .611668/.510644/.513922/.486086 | `results/phase4_validation_audit/PHASE4_3_FINAL_REPORT.md`; `FINAL_STAGE_COMPARISON.csv`; independently recomputed metric table | Different UCI endpoint/task; not compared as ZAS external validation. |
| UCI centre AUC ranges per stage; heatmap values | `results/phase4_validation_audit/INDEPENDENT_METRIC_REPRODUCTION.csv` filtered to `metric=roc_auc` | Figure 3 shows saved audited centre metric values and denominators. |
| Stage 3→4 paired differences and intervals | `results/phase4_validation_audit/PAIRED_STAGE_COMPARISONS.csv`, pooled row `stage3_resting_measurements` → `stage4_exercise_test`; `runner/phase4_validation_audit/run.py` lines 208–234 | Paired record bootstrap, 400 draws, same `(held_out_centre, source_row)` sample indexes; not patient or centre uncertainty. |
| Feature timing and excluded fields | `results/phase5_1_audit/ZAS_FEATURE_TIMING_AUDIT.md`; `ZAS_FEATURE_TIMING_TABLE.csv`; `ZAS_LEAKAGE_AUDIT.md`; Phase6 protocol | The “Phase 5.1.1” named namespace was not found; relevant audit is under `phase5_1_audit`. No predictor timestamp confirms decision-time availability. |
| Calibration cross-fit selection dependence, fit-size mismatch, outer-test calibration evaluation | `results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md`; `CALIBRATION_TRAINING_SIZE_AUDIT.csv`; `OUTER_CALIBRATION_METRIC_CHECKS.csv` | Calibration limitation is separated from outer discrimination validity. |
| Four-vs-three calibration metadata count is reporting-only | `results/phase6_1_audit/PHASE6_IMPLEMENTATION_AUDIT.md`; `CALIBRATION_SELECTION_METADATA_CORRECTED.csv` | No effect on model fits/predictions. |
| Citation [1] original ZAS article metadata | PubMed PMID 23537611; DOI 10.1016/j.cmpb.2013.03.004 | Verified bibliographic title/authors/journal/year/pages/DOI. |
| Citation [2] ZAS repository metadata | Official UCI page, DOI 10.24432/C5Q31T | Verified dataset description and schema note. |
| Citation [3] model-selection bias | JMLR publisher page, volume 11 (2010), pp. 2079–2107 | Claim restricted to general selection optimism. |
| Citation [4] calibration | BMC Medicine DOI 10.1186/s12916-019-1466-7 | Claim restricted to discrimination/calibration distinction. |
| Citations [5–6] reporting/appraisal recommendations | BMJ DOI 10.1136/bmj-2023-078378 and 10.1136/bmj-2024-082505 | Used as reporting/risk-appraisal frameworks only. |
| Citations [7,9] UCI Heart Disease sources | Official UCI DOI 10.24432/C52P4X; Detrano et al DOI 10.1016/0002-9149(89)90524-9 | Four databases, missingness, outcome descriptions and intro paper verified. |
| Citation [8] AP/PR interpretation | Davis & Goadrich, ICML 2006, DOI 10.1145/1143844.1143874 | Supports context for precision–recall metrics; study numbers still from local saved artifacts. |

## Abstract numerical audit

- ZAS source size and label counts: **303 / 216 / 87** match saved repeat table and validated dataset.
- Phase6 AUCs and comparator AUCs: match repeat rows 1–3 in `REPEAT_LEVEL_PERFORMANCE.csv` (rounding only).
- AUC paired differences and interval: match Phase6 paired summary; independently reconstructed in Phase6.1. Interval explicitly labeled split-variation.
- Group/model selection counts: match `MODEL_SELECTION_FREQUENCY.csv` and the 15 outer selections.
- UCI record counts and Switzerland class counts: match centre profile and audited predictions.
- UCI stage AUCs: match Phase4.3 report; Stage2/Stage4 task distinction stated.
- All citations in the abstract are avoided: no abstract attribution is necessary; relevant methodological citations are in Introduction/Methods.
