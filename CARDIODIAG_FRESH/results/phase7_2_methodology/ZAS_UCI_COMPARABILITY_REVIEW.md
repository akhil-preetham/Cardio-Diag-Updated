# ZAS and UCI comparability review

## Keep the tracks separate

ZAS and UCI answer distinct within-dataset questions, have different schemas and provenance, and have no verified patient linkage. Their predictions and metrics must not be pooled. UCI leave-one-centre-out performance is not external validation of ZAS.

| Aspect | ZAS | UCI four-centre Heart Disease |
|---|---|---|
| Records | 303 rows; unique patient identity not verified | 920 records across Cleveland 303, Hungary 294, Switzerland 123 and VA Long Beach 200; unique patient identity not verified |
| Endpoint | Cath: Cad vs Normal; angiography-defined threshold per local source/data dictionary | `num` 0 vs 1–4; related angiographic disease classification, not assumed identical to the ZAS definition |
| Features | 56 local columns reported in project audit, 55 predictors plus Cath; different clinical variables including extensive symptom/history/lab fields | Conventional UCI attributes with different coding and availability; staged sets use age/sex, chest-pain category, resting measures/ECG, then exercise variables. `ca`, `thal` and target are excluded from stated prediction groups. |
| Missingness/provenance | ZAS local table is a distinct source; predictor timing and some schema documentation remain unresolved | Four source centres have different missingness and case spectrum; harmonized reconstruction/imputation is part of local pipeline; Switzerland is high-prevalence and has sparse negatives. |
| Validation | Stratified source-row CV for Phase 5; matched repeated source-row outer splits for Phase 6 | Leave-one-centre-out/internal-external evaluation within the four assembled centres. It does not add a new independent ZAS-compatible cohort. |

Sources: `results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`; `results/phase4_validation_audit/PHASE4_3_FINAL_REPORT.md`; `results/phase4_validation_audit/CENTRE_SPECTRUM_ANALYSIS.csv`; `results/phase5/data_registry/DATASET_COMPATIBILITY_MATRIX.csv`; `results/phase5/uci_track/UCI_TRACK_FINAL_DECISION.md`; `results/phase3/PHASE3_FINAL_DECISION.md`.

## Supported UCI claims

The saved UCI analysis supports describing pooled and held-out-centre performance of the staged procedures on those four assembled centres. Phase 4.3 independently reconstructed 3,680 saved staged predictions and their metrics. Stage 2 is an intended symptom-stage information set; Stage 4 includes exercise-test information and is a different, later task. Their scores should appear in separate stage rows with centre-specific results and class counts, not as a single universal model trajectory. The stage labels express the intended information set; they do not prove timestamp availability.

Switzerland has 115 events among 123 records, leaving only 8 negatives. Specificity, calibration and negative-case ranking for this centre are consequently imprecise and should be reported with denominators. Pooled metrics are conditional on the observed centre mix; spectrum differences can affect pooled ranking. See `results/phase4_validation_audit/PHASE4_3_FINAL_REPORT.md`, “Reliability and centre robustness”; `CENTRE_SPECTRUM_ANALYSIS.csv`; `CENTRE_CALIBRATION_REPORT.md`; and `PHASE4_3_FINAL_DECISION.md`.

Phase 3’s external validation decision states that an independent patient-level cohort, compatible predictor set, timing, outcome adjudication, governance and sample size were unavailable. Cleveland is a related endpoint candidate, not a strict independent ZAS validation set. See `results/phase3/PHASE3_FINAL_DECISION.md` and `results/phase5/data_registry/DATASET_COMPATIBILITY_MATRIX.csv`.

## Reporting recommendation

Label UCI “four-centre leave-one-centre-out evaluation” or “internal-external validation within the assembled UCI centres,” with an explicit statement that it is not external validation of ZAS. Report endpoint recoding, cohort counts, centre-specific missingness/outcome prevalence, stage definitions, and centre-level metric denominators. Keep record-level paired bootstrap intervals explicitly record-based. Do not pool ZAS/UCI or use UCI to claim transportability of the ZAS model.
