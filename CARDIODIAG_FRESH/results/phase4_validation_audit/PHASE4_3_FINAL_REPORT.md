# CARDIODIAG Phase 4.3 final report

## Independently verified

The audit reconstructed 920 UCI four-centre records and independently checked 3,680 saved staged predictions. Prediction keys, centre membership, duplicate keys, target labels, all requested classification metrics, calibration intercept/slope, saved metric agreement, calibration curves, and manifest hashes were checked. Historical Phase 4.1/4.2 output hashes remain valid.

## Main performance evidence

                      stage  roc_auc   pr_auc    brier  log_loss  calibration_intercept  calibration_slope
          stage1_basic_risk 0.717900 0.729290 0.211205  0.611668               0.072896           0.947730
            stage2_symptoms 0.822525 0.824077 0.168070  0.510644              -0.148460           1.085358
stage3_resting_measurements 0.821505 0.819088 0.168381  0.513922               0.270583           0.955899
       stage4_exercise_test 0.845033 0.864469 0.158443  0.486086               0.076470           1.003182

Paired bootstrap comparisons used the same `(held_out_centre, source_row)` records. Stage 1 to Stage 2 improved ROC-AUC by approximately 0.105. Stage 2 to Stage 3 changed ROC-AUC by approximately -0.001 with an interval crossing zero. Stage 3 to Stage 4 improved ROC-AUC by approximately 0.024. These are incremental record-level comparisons, not clinical significance tests.

## Reliability and centre robustness

Calibration was calculated for every stage and held-out centre using fixed five- and ten-bin ECE analyses, with bin counts retained. Centre-specific results are unstable where prevalence and negative-case counts are sparse, particularly Switzerland. OOD diagnostics compare development-only standardized Euclidean, robust Mahalanobis and isolation-forest scores. Selective prediction reports fixed confidence thresholds across centres; neither diagnostic validates a clinical safety or abstention rule.

## Corrected and unresolved issues

The paired-comparison implementation explicitly includes centre in the row key; local source-row indices are not globally unique. Duplicate sensitivity reconstructs full-profile duplicate groups and distinguishes removal from flagged-only analysis. Remaining limitations include absent patient identifiers, centre sampling uncertainty, incomplete predictor timing documentation, record-level rather than cluster-level bootstrap intervals, and no independent external cohort.

## Decision

Stage 2 is the most defensible expanded early-presentation task. Stage 4 is a separate exercise-test task and cannot be substituted for initial pretest prediction. No stage is clinically selected. Independent external validation remains required, and no claim of clinical readiness, universal transportability, reliable abstention or patient-level safety is supported.

## Reproduction

```powershell
$env:PYTHONPATH='src;.'
& .\.venv-repro\Scripts\python.exe runner\phase4_validation_audit\run.py
& .\.venv-repro\Scripts\python.exe runner\phase4_validation_audit\make_figures.py
& .\.venv-repro\Scripts\python.exe -m pytest -q
& .\.venv-repro\Scripts\python.exe -m ruff check runner\phase4_validation_audit tests\test_phase4_validation_audit.py
```
