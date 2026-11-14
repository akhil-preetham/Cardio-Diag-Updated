# Phase 7.4 — Manuscript numerical audit

## Scope

Inspected the complete Phase 7.3 manuscript, six table pairs, four PNG/PDF figures, claim traceability, Phase 6.1 independent reconstruction, Phase 4.3 audited outputs, and saved Phase 5/6 results. Recomputed declared SHA-256 values read-only. No fitting, search, or benchmark was run. Source rows are not assumed to be unique patients.

## Claim-by-claim audit

| Location / claim | Source checked | Finding |
|---|---|---|
| Title and abstract: retrospective ZAS discrimination; separate UCI analysis | Manuscript; Phase 7.2 final decision | Correct bounded framing; no prospective, prediagnostic, or external-validation claim. |
| ZAS 303 rows, 216 Cad, 87 Normal, prevalence .7129 | Validated local ZAS CSV; Phase 6 repeat table | Correct counts/coding. These are source rows; patient-level independence is unverified. |
| 3×5 outer folds, 3 inner folds, six groups × five unique families | Phase 6 protocol, runner, frozen split manifest; Phase 6.1 audit | Correct for 15 completed partitions; reduction from planned five repeats after runtime limit is disclosed. |
| Repeat AUCs .9134/.8999/.9176 vs .7370/.7420/.7300 | Phase 6 `REPEAT_LEVEL_PERFORMANCE.csv`; Phase 6.1 reconstructed repeat metrics | Correct rounding from complete 303-row OOF sets, not fold-average AUC. |
| Paired differences +.1764/+.1579/+.1877; mean +.1740; interval +.1367–+.2113 | Phase 6 paired differences/summary; Phase 6.1 reconstruction | Correct. Interval is called split-variation, not patient/population uncertainty or significance evidence. |
| Group eligible_combined 15/15; L2 9/15, elastic-net 4/15, spline 2/15 | Model selection frequency, outer selections, Table 6 | Correct. Manuscript correctly avoids inferring individual feature necessity. |
| UCI 920 records; stage AUC .7179/.8225/.8215/.8450; Switzerland 115 positive, 8 negative | Phase 4.3 audited stage and centre files | Correct; Stage 4 is distinct later exercise-test task. No external validation language. |
| Official ZAS ≥50% outcome; 54 catalog features vs local 56 columns including outcome | UCI catalog, local schema, Phase 5.1 audit | Discrepancy correctly retained as unresolved. |
| Phase 5 AUC .928214, AP .966830, Brier .096424, log loss .314108; 36 rows | Phase 5 benchmark; Phase 5.1 evidence decision | Correctly called post-hoc exploratory and not used to form Phase 6 candidate pool. |
| Phase 6 secondary means: AP .9573/.8469, Brier .1076/.1746, log loss .3467/.5286; sensitivity/specificity .8318/.8314 vs .6898/.7088; calibration intercept/slope .173/.868 vs −.007/1.005 | Three repeat rows and Phase 6.1 reconstruction | Rounded means agree. Training-selected Youden thresholds and calibration limitations are disclosed. |
| Feature importance Typical Chest Pain .125 (SD .050), Age .042 (.017), DM .019 (.011); Age/Sex AUC .726–.731 | Phase 6 permutation and demographic sensitivity CSVs | Values agree; correctly descriptive/noncausal and timing-uncertain. |
| Table 1 counts/prevalences; Table 2 feature timing and exclusions | Cohort profiles; Phase 5.1 timing/leakage audits; Phase 6 protocol | Correct. No time-confirmed predictor; symptoms/ECG/labs uncertain; EF-TTE/RWMA/VHD potentially downstream; Cath/angiographic fields excluded. |
| Table 3 all fold-repeat metrics and calibration values | Phase 6 repeat results; Phase 6.1 reconstruction | Values match; calibration caveat retained. |
| Table 4 UCI pooled metrics and centre AUC ranges | Phase 4.3 independent metric reproduction | Values match; PR/AP has cohort context and UCI remains separate. |
| Table 5 Stage 3→4 changes and 400 paired record bootstrap draws | Phase 4.3 paired comparisons/runner | Values match; intervals are record-level conditional on four centres, not patient/centre sampling. |
| Table 6 selection frequencies | Phase 6 outer selections and frequency file | Counts and proportions agree. |
| Figure 1 workflow | Phase 6 and Phase 4.3 protocols | Correct separation of ZAS and UCI tracks. |
| Figure 2 repeated AUCs | Phase 6 repeat results | Labels match. Axis starts at .65, visually accentuating the difference; numeric labels/ticks remain visible. Recommend zero baseline or a stronger note at journal production. |
| Figure 3 UCI centre heatmap | Phase 4.3 held-centre metric file | Values and denominators match; numeric annotations visible; sparse Switzerland negatives disclosed. |
| Figure 4 timing diagram | Phase 5.1 audits | Correct documentation categories; no timestamps inferred from names. |
| Discussion/conclusion | Phase 7.1/7.2 decisions; robustness/calibration files | Correct retrospective source-row scope; +.174 not cast as patient-level uncertainty/significance; no clinical utility or external-validation claim. |

## Disposition

No numerical or scientific interpretation discrepancy requiring a manuscript rewrite was found. The Phase 7.3 manifest's 74 inputs/outputs, Phase 7.1's 25, and Phase 7.2's 41 hashes all matched. Figure 2's truncated scale is a presentation recommendation, not a metric error. The Phase 5 .928 result is appropriately exploratory/post-hoc.
