# Phase D — UCI multicentre protocol and decision

## Protocol and task separation

The project preserves four centre strata and internal-external leave-one-centre-out evaluation. Stages are not interchangeable: Stage 2 is the early-presentation symptom task; Stage 4 includes exercise-test information and is a later task. Phase 4.3 uses saved out-of-centre predictions from the same source rows across stages and key-matches comparisons by `(held_out_centre, source_row)`. The dataset has no verified patient ID; record-paired uncertainty is not patient-clustered or centre-level uncertainty.

The four-centre source table has Cleveland (303 rows/139 events), Hungary (294/106), Switzerland (123/115), and VA Long Beach (200/149). Switzerland has only eight negatives; discrimination/calibration estimates there are unstable and prevalence is extreme. The aggregate totals are 920 rows and 509 events.

## Centre-wise discrimination

| Centre | Stage 2 n/events | Stage 2 ROC-AUC | Stage 4 n/events | Stage 4 ROC-AUC |
|---|---:|---:|---:|---:|
| Cleveland | 303/139 | 0.8305 | 303/139 | 0.8638 |
| Hungary | 294/106 | 0.8380 | 294/106 | 0.8747 |
| Switzerland | 123/115 | 0.7478 | 123/115 | 0.7261 |
| VA Long Beach | 200/149 | 0.7102 | 200/149 | 0.7294 |

Stage 2 pooled out-of-centre: ROC-AUC 0.8225, PR-AUC 0.8241, Brier 0.1681, log loss 0.5106. Stage 4: 0.8450, 0.8645, 0.1584, 0.4861. Stage 3 to 4 paired record-level differences are reported in `PHASE_A_METRIC_RECONCILIATION.md`; this does not make exercise-test measures appropriate for an early-presentation decision.

## Reliability and robustness interpretation

Existing reports contain centre-wise calibration, missingness shifts, duplicate-profile checks, OOD diagnostics, selective-prediction summaries, and threshold sweeps. These are descriptive. OOD distance is not disease probability; fixed-threshold selective prediction is not validated abstention; decision curves do not establish clinical utility without justified decision thresholds. The duplicate-sensitivity artifact is excluded from supporting robustness claims until corrected with the centre-qualified key. Switzerland-specific estimates are especially fragile.

## Decision

Stage 2 is the most defensible expanded early-presentation *research task* in this dataset; Stage 4 is a separate later exercise-test task. Neither is clinically validated. Do not describe UCI as external validation of ZAS: the populations, available variables, and endpoint cohorts are not a proven independent matching cohort, and the UCI track does not contain all five frozen V3 predictors.
