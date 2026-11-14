# Paper-style report content audit

## Scope and structure

The report uses the supplied PDF only as a writing/layout specimen. It is a six-page, US Letter, two-column research-paper format with a full-width title block, structured abstract, index terms, Roman-numbered sections, numbered tables and figures, declarations and numbered references. The sample PDF's author names, results, data and any instructions in it were not used as evidence or copied.

The current research question is **retrospective discrimination of angiography-defined CAD using recorded clinical features**. The manuscript does not claim prospective prediction, confirmed prediagnostic availability, clinical utility, patient-level independence, or external validation.

## Included evidence

- ZAS: 303 source rows; Cath labels 216 CAD and 87 Normal. Patient identity is unverified, so the evaluation unit remains source rows.
- Feature timing: no source-confirmed eligible predictor; 38 candidate fields are timing uncertain (Age/Sex plus 36 other fields). Cath and diagnostic/downstream fields are excluded.
- Phase 5 AUC 0.928: explicitly post hoc, selected from 36 benchmark rows, exploratory only, and not used to define the Phase 6 candidate pool.
- Phase 6: frozen 30-candidate pool, six feature groups and five unique model families, three inner folds and 15 paired outer partitions across three repeats. Every repeat-level metric uses its complete 303-row out-of-fold set.
- Selection-aware ROC-AUC by repeat: [0.9134, 0.8999, 0.9176]; matched V3-style comparator: [0.737, 0.742, 0.73]; mean paired difference 0.1740. The interval is split variation, not patient-level or population uncertainty.
- Secondary metric means and calibration summaries are reconciled against saved results. Calibration is clearly described as selection-dependent and training-size mismatched.
- UCI Heart Disease is a separate four-centre retrospective study track, not external validation of ZAS. Strict pooled logistic ROC-AUC is 0.6753; staged AUCs are [0.7179, 0.8225, 0.8215, 0.845].
- The independent Phase 6.1 audit verified saved labels, partitions and metrics. The clean environment audit is described as computational reproducibility, not scientific validity.

## Numerical verification

Every abstract number was checked against its source: six repeat-level ROC-AUC values, the paired mean and its three-repeat split-variation interval, and the rounded strict UCI pooled ROC-AUC. Fold-level ranges, model-selection counts, secondary metrics and calibration summaries were also checked. No pooled-fold AUC average was substituted for a complete-repeat AUC. The Phase 5 value is labelled exploratory throughout.

| Abstract claim | Printed value | Reconciled source |
|---|---:|---|
| Selection-aware repeat ROC-AUC | [0.9134, 0.8999, 0.9176] | `REPEAT_LEVEL_PERFORMANCE.csv`, complete 303-row OOF sets |
| V3-style repeat ROC-AUC | [0.737, 0.742, 0.73] | `REPEAT_LEVEL_PERFORMANCE.csv`, same repeats and rows |
| Mean paired ROC-AUC difference | 0.174 | Paired differences across three repeats |
| Split-variation interval | [0.137, 0.211] | Recomputed t interval over three repeat-level differences; not patient-level uncertainty |
| UCI strict pooled logistic ROC-AUC | 0.675 | `POOLED_OUT_OF_CENTRE_METRICS.csv` strict/logistic value rounded to three decimals |

## Remaining limits

Author and institutional declarations, patient identity, feature acquisition times, the local ZAS schema discrepancy and clinical data-use documentation remain unresolved. These are not inferred from the style-reference PDF or from populated reporting checklists.
