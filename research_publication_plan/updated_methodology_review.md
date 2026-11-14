# Updated CARDIODIAG: methodological review and meaningful next study

Prepared 14 November 2026. This review reads current code and saved artifacts; it does not run model training or modify historical outputs. Paths below are relative to `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH` unless stated otherwise.

## What is already supported

The strongest current contribution is a traceable retrospective reanalysis in which the complete feature-group/model-selection procedure is evaluated inside outer validation, compared on the same source-row partitions with a five-feature reference procedure. This is a methodological and reproducibility contribution, not a new classifier or verified clinical diagnostic system.

| Finding | Local evidence | Defensible interpretation |
|---|---|---|
| Selection-aware versus V3-style mean repeat ROC-AUC 0.910316 versus 0.736315; mean difference 0.174001 | `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv`; `results/phase6_1_audit/PHASE6_METRIC_RECONCILIATION.md` | Better discrimination of the recorded angiographic label on the evaluated source-row partitions |
| Six feature groups and five family labels are searched within outer training | `runner/phase6_zas_selection/run.py`, `GROUPS`, `MODEL_GRIDS`, `choose_candidate` | Within-run group/family/hyperparameter selection is evaluated; earlier researcher choices are not thereby independently validated |
| eligible_combined selected 15/15 times; L2 9, elastic-net 4, spline 2 | `results/phase6_zas_selection/MODEL_SELECTION_FREQUENCY.csv` | Stability of group-level choice across tested splits; no evidence every one of the 38 variables is necessary |
| Typical Chest Pain has largest marginal importance | `results/phase6_zas_selection/ROBUSTNESS_REPORT.md`; `FEATURE_PERMUTATION_IMPORTANCE.csv` | Reason to test symptom dependence; not a causal or timing claim |
| UCI four-centre/staged analysis exists separately | `results/phase4_improvement/FINAL_MODEL_SELECTION.md`; `DOMAIN_SHIFT_EXPERIMENTS.csv` | A complementary internal-external validation task; not external validation of ZAS |

## Issues to resolve before stronger claims

1. **Selection-independent calibration is not implemented.** `choose_candidate` uses all outer-training labels before `crossfit_selected_scores` generates estimator-out-of-fold scores with that winner fixed. A calibration-fold label influenced the winning configuration. Sigmoid and Youden threshold then share these scores. This does not expose outer-test labels, so saved outer discrimination remains a useful estimate of the implemented training procedure. It does limit assertions about independent calibration construction or transferable individual probabilities. Source: `runner/phase6_zas_selection/run.py:149`, `:177`, `:190`; `results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md`.
2. **Calibration score training size differs from final refit size.** Inner scoring models train on roughly two-thirds of outer training; the final selected model uses all outer training. This is a construction mismatch to evaluate rather than a reason to retroactively erase existing metrics.
3. **The four-versus-three calibration-fold metadata bug remains in historical code.** Lines 330–331 create four reporting entries with `range(1, 5)` although three folds are operational. The separate Phase 6.1 CSV corrects display metadata; it does not change the fitted procedure. New implementation should derive metadata from actual splits.
4. **Retrospective protocol adaptation remains a limitation.** Phase 6 candidate universe follows earlier Phase 5 benchmarking and reuses first three repeats of the V3 outer partition manifest. The code demonstrates no within-run outer-label path; it cannot prove the researcher's design choices were independent of all previous dataset results. New random seeds on the same 303 rows do not create a new untouched cohort. Call the study a retrospective methodological reanalysis, and do not equate frozen-before-this-run with a prospectively registered confirmatory study. Sources: `SELECTION_AWARE_PROTOCOL.md`; `results/phase5_1_audit/ZAS_0928_EVIDENCE_DECISION.md`; `load_splits`.
5. **Uncertainty is conditional/descriptive.** The three repeats overlap. A t interval across them does not establish patient/population superiority. Keep the differences and ranges descriptive; do not select a correction method based on which produces significance.
6. **Data schema and predictor timing are limiting evidence gaps.** No source-confirmed patient identity or measurement times establish prediagnostic availability. The local 56-column ZAS file differs from the catalogue's 54-feature description. More tuning cannot fix these facts.
7. **Feature typing is dtype-based.** `make_pipeline:104` treats all numeric columns except Sex as numeric. This may be appropriate for some binary indicators, but future diverse datasets need an explicit semantic data dictionary for binary, nominal, ordinal and continuous fields. Spline expansion should be restricted to justified continuous variables, rather than numeric-coded categories by accident.

## Important API finding: elastic-net is not a confirmed current bug

`make_pipeline` instantiates `LogisticRegression(solver='saga', l1_ratio=0.25, ...)` without an explicit penalty string. This is not sufficient evidence that the current model is L2.

The installed scikit-learn source at `.venv/Lib/site-packages/sklearn/linear_model/_logistic.py:1279` defaults `penalty='deprecated'` and `l1_ratio=0.0`. Its `fit` implementation at lines 1362–1376 maps an interior l1_ratio to `penalty='elasticnet'`. Therefore the existing interior-ratio grid implements elastic-net under this installed API. Older scikit-learn releases may behave differently. Pin and report the actual version; add a structural configuration assertion in a future isolated implementation rather than changing old results based on obsolete API assumptions. No predictive fit was run to establish this finding; it follows directly from installed source inspection.

## Recommended next study: reliability under feature availability and calibration construction

An incremental, defensible study question is:

> On explicitly defined retrospective CAD tasks, how do calibration construction and availability of symptom/laboratory/ECG information affect discrimination, probability accuracy and stability across source-row splits or held-out centres?

Keep the current question and historical outputs. The additions below should form a small prespecified correction/robustness package, not a growing search for a better headline AUC.

### A. Correct calibration with a selection-independent ensemble

One exact, feasible design worth piloting:

1. Preserve an outer evaluation split. Partition its outer-training rows into three calibration blocks.
2. For each block k, perform the *whole* feature-group/family/hyperparameter selection only on the other two blocks, using a separate inner CV within that development subset. The held calibration block's labels must not enter that selection.
3. Fit the chosen base pipeline on that development subset and generate scores on its held calibration block. Fit a prespecified regularized sigmoid on those calibration predictions/labels.
4. Keep this same fitted base pipeline and its calibrator together as one component. Do not refit its base model using calibration rows after learning the map.
5. Apply each component to the outer test rows and average the three calibrated component probabilities. This avoids applying a map learned from smaller training models to a larger refit model. It evaluates an ensemble procedure, not one selected full-data model.
6. Evaluate probability accuracy (Brier/log loss), discrimination, and calibration plots/slope on untouched outer predictions. Compare with a paired *uncalibrated ensemble constructed from the same base components*, plus the original historical procedure as a qualified reference. Calibration may improve or worsen results; report either outcome.

At 303 rows the calibration blocks have only about 80 observations inside an outer training set, so estimation remains noisy. Use low-complexity prespecified regularized sigmoid calibration; do not add unconstrained isotonic fitting or choose calibrators on the outer test set. Group-aware splitting becomes mandatory if repeat-patient identifiers are later recovered.

Make the primary comparison probabilistic and threshold-free. Either omit threshold metrics or use a fixed prespecified rule for descriptive reporting. A clinically justified sensitivity/cost threshold requires clinical input. If a data-selected operational threshold is required, allocate an additional independent training-only threshold selection layer; do not pretend an ensemble's in-sample training probabilities are independent threshold-validation evidence.

This is a design proposal, not completed evidence. It adds computational cost: retaining the current grid means 120 hyperparameter combinations across six groups, approximately 360 inner-CV fits plus 30 search refits per selection. Three component selections × 15 outer partitions is about 17,550 search/refit operations before comparator/metric work. Wall time cannot be inferred reliably from CPU count alone. Pilot one outer partition with a logged time/RAM budget before scheduling the full run. Any smaller candidate pool must be frozen and explained as an exploratory follow-up informed by historical results, not silently relabeled as the original protocol.

### B. Focused information-group ablation

Do not immediately add more model families. Existing selection predominantly favors regularized logistic models. Prespecify a few information sets and apply the *same model/tuning/calibration rules* to each:

- Historical 38-field combined set.
- Combined set excluding the symptom block (including Typical Chest Pain).
- Risk factors plus symptoms/examination, without laboratory and ECG variables.
- A compact baseline fixed before running the new comparison.

These comparisons isolate information availability more clearly than comparing a rich selected model with five baseline features and attributing all gain to algorithm novelty. Timing remains uncertain for all these sets. If choosing a compact subset, perform that choice inside development training only. Report any practical loss in discrimination/probability accuracy, feature count and runtime; a simpler procedure with similar performance can be a meaningful result even without a higher AUC.

### C. Missingness and centre robustness

The UCI track already uses median/mode imputation and numeric missingness indicators. It has not performed native-missing comparisons or multiple imputation as primary methods (`results/phase4_improvement/MISSING_DATA_METHOD_COMPARISON.csv`). Domain shift work is descriptive (`DOMAIN_SHIFT_EXPERIMENTS.csv`).

For a bounded follow-up, compare training-fold median/mode imputation with and without indicators, with all other procedure choices held constant. Report centre-wise missingness and performance, and optionally preregister simulated test-time deletion rates for specific clinically grouped fields. Test-time corruption is a stress test, not a new clinical cohort; MCAR masks do not establish robustness to actual MNAR processes. Do not choose deletion patterns or methods to maximize favorable results. Avoid many sparse subgroup claims with eight Switzerland negatives.

Centre-held-out results should show each centre and dataset-specific endpoint, not only pooled AUC. A high pooled AUC can reflect between-centre distribution differences; centre-specific discrimination and calibration answer additional questions. If local target-centre outcomes are used to recalibrate, explicitly separate an adaptation subset from a later target evaluation subset and rename the experiment; it is no longer untouched centre-held-out validation.

## Diverse datasets: what can and cannot be promised

Two legitimate strategies differ:

- **External validation of one fitted ZAS model:** requires compatible predictors, measurement definitions/units, timing, endpoint and independent records. UCI's `num` task is not automatically equivalent to ZAS Cath. Retraining on UCI does not externally validate fixed ZAS weights.
- **Replication of a learning procedure across datasets:** each dataset has its own feature dictionary, target, preprocessing policy and outer evaluation. Refit within each dataset or development centres, report separate results, and test robustness of the workflow rather than claiming one universal model. Mendeley, cardiovascular-70k, CAD angiography and heart-failure mortality cannot share one undifferentiated metric table or universal target.

The strongest long-term addition is a new compatible independently acquired angiographic cohort with documented patient identifiers and predictor times. If unavailable, a carefully bounded reproducibility/robustness paper is preferable to unsupported clinical reliability claims. A predictable and reliable answer for every diverse patient or dataset is not something this evidence can guarantee.

## Execution priority and acceptance criteria

| Priority | Work | Required output/decision |
|---|---|---|
| P0 | Source/schema/timing and authorship/governance reconciliation | Verified dictionary and provenance; retained explicit gaps where facts cannot be recovered |
| P0 | Finalize narrow objectives and primary endpoints | Dated protocol distinguishing retrospective hypothesis development from fresh validation |
| P1 | Add split/configuration/metadata checks in isolated new namespace | No overlap of selection/development, calibration and outer test labels; semantic typing; actual fold counts; convergence logging; environment pin |
| P1 | Single-partition compute pilot for calibration design | Measured time/RAM and full-run budget; no favorable-score stopping rule |
| P1 | Locked calibration/ablation follow-up | New saved predictions, full metric tables, unchanged historical hashes, explicit negative findings |
| P2 | Separate UCI missingness/centre replication | Site-specific results and no false ZAS external-validation claim |
| P2 | IEEE-style manuscript | Traceable evidence only; proposed experiments stay Methods/plan until executed; journal-specific format and author declarations confirmed |

Do not claim clinical calibration, deployment readiness, broad generalization, algorithmic novelty or statistical significance simply because these engineering steps are completed.
