# Supplementary material

## S1. Frozen ZAS baseline

The frozen V3 configuration in `results/primary_research_v3/EXPERIMENT_CONFIGURATION.json` uses Age, Sex, Current Smoker, EX-Smoker and FH. Five outer and four inner folds reuse the stored split assignments. Calibration uses fold-specific nested selection and sigmoid calibration. The executive report gives the highest descriptive calibrated ROC-AUC as 0.735 (SD 0.060). This is internal validation; no patient identifier is available.

## S2. Phase 5 ZAS metric reconstruction

For `eligible_combined` Logistic and Ridge, each has 606 outer predictions. The two repeats are averaged by source row to 303 rows, with 216 positive outcomes and no label conflicts. Recomputed metrics: ROC-AUC 0.9282141337, PR-AUC 0.9668302657, Brier 0.0964242607 and log loss 0.3141080938. Every pairwise Logistic/Ridge prediction is identical. The aliases use the same L2 LogisticRegression constructor and C grid. This candidate is exploratory because feature-group/model choice was not nested and most inputs lack timing evidence.

## S3. UCI staged results

| Stage | n | Events | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 920 | 509 | 0.717900 | 0.729290 | 0.211205 | 0.611668 |
| Stage 2 | 920 | 509 | 0.822525 | 0.824077 | 0.168070 | 0.510644 |
| Stage 3 | 920 | 509 | 0.821505 | 0.819088 | 0.168381 | 0.513922 |
| Stage 4 | 920 | 509 | 0.845033 | 0.864469 | 0.158443 | 0.486086 |

Paired Stage 3-to-Stage 4: ROC-AUC difference +0.023528 (95% record-bootstrap interval 0.006084 to 0.044258); PR-AUC +0.045381 (0.019600 to 0.070623); Brier -0.009939 (-0.018608 to -0.001509); log loss -0.027836 (-0.051392 to -0.005624). Comparisons pair by `(held_out_centre, source_row)`.

## S4. UCI centre-level results

| Centre | n/events | Stage 2 AUC | Stage 4 AUC |
|---|---:|---:|---:|
| Cleveland | 303/139 | 0.830541 | 0.863792 |
| Hungary | 294/106 | 0.837992 | 0.874699 |
| Switzerland | 123/115 | 0.747826 | 0.726087 |
| VA Long Beach | 200/149 | 0.710225 | 0.729438 |

## S5. Duplicate audit limitation

Two exact within-centre profile groups are documented in `results/phase4_validation_audit/DUPLICATE_GROUPS.csv`. The saved duplicate-sensitivity implementation drops rows by a source-row integer without centre identity. Since row indices repeat across centres, it incorrectly removes additional unrelated rows, while its `excluded_n` is always zero. The artifact is retained as historical evidence but is excluded from the claims. The pooled base results and paired comparisons use centre/source-row keys and remain separately supported by the Phase 4.3 audit.

## S6. Existing reliability and explanation outputs

Calibration, OOD, selective-prediction, decision-curve and feature-importance files are listed in the Phase 5 and Phase 4 manifests. Interpretation is limited as described in the manuscript: record-level uncertainty, unstable Switzerland estimates, OOD scores not being probabilities, selective prediction not validated for clinical abstention, and feature importance not being causal.

## S7. Reproduction commands

From the project root, set `PYTHONPATH=src;.` and run the specific saved-output audit and QA commands listed in `results/final_project/REPRODUCIBILITY_AND_QA_REPORT.md`. Model-training commands are intentionally not included for the completion audit because no additional training was justified by the unresolved feature-timing evidence.
