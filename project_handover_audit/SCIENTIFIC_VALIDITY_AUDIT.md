# Scientific validity audit

The defensible question remains retrospective discrimination of angiography-defined CAD using recorded clinical features.

Phase 6 source inspection finds candidate selection and tuning in GridSearchCV within outer training. Pipeline-local imputation, encoding, scaling and optional splines are refit within training folds. Six groups and five family labels yield 30 candidates. The first three repeats of frozen five-fold outer assignments are used; operational inner folds are generated anew with 61051 + 100*repeat + fold and n_splits=3, despite the older source split manifest containing four-fold inner assignments. Reusing the outer assignments does not mean reusing its old inner folds.

Selection precedes crossfit_selected_scores. Held-out calibration-fold labels have influenced group/family/hyperparameter selection. Scores are estimator-out-of-fold, not selection-independent. Sigmoid fitting and Youden threshold selection share these training scores. Inner scoring fits use about two-thirds of outer training, while final prediction uses the full outer-training refit. These limitations remain in the historical procedure. The metadata range(1,5) produces four labels for three operational folds; Phase 6.1 supplies a separate corrected display, not a new calibration experiment.

No outer-test-label path into selection/calibrator/threshold was identified in the inspected runner. Outer-test permutation importance is post-fit descriptive analysis. The independent focused tests confirm saved coverage, labels, candidate ranking metadata, fold/repeat metrics and hashes. This supports the recorded outer discrimination estimate of the implemented procedure; it does not independently reproduce model fitting from scratch.

Mean complete-repeat AUC is 0.910316 for selection-aware versus 0.736315 for the re-evaluated V3-style procedure; mean difference 0.174001. Comparator features/family are fixed, but C is inner-tuned and calibration is refitted. It is not an untouched original fitted model. Averaging fold AUCs would give different values and is not used for the primary repeat result.

Three overlapping repeats do not justify independent-replicate t inference. The reported interval is descriptive split variation; do not claim population significance, patient-level superiority or clinical benefit. Pooling repeat OOF scores also compares scores from different fitted fold models, so interpret this aggregation as the recorded procedure estimand, not one fixed-model ROC curve.

Timing, patient independence, schema provenance, compatible external validation and clinical utility remain unresolved. Phase 5's post-hoc winning row is exploratory. UCI is separate. Existing checklists do not establish TRIPOD+AI compliance or low PROBAST+AI risk.
