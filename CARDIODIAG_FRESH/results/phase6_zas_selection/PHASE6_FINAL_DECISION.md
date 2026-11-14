# Phase 6 final decision

## Classification

**1. Improvement supported by selection-aware validation.**

The selection procedure exceeded the paired V3 comparator on ROC-AUC in all three repeat-level estimates, with the descriptive split-variation t interval above zero. This supports a within-cohort retrospective discrimination improvement under the frozen candidate-selection procedure; it does not establish patient-level, prospective, external, or clinical benefit.

## Paired results

| Estimate | Selection-aware procedure | Frozen V3 comparator |
|---|---:|---:|
| Repeat-mean ROC-AUC | 0.910 (range 0.900–0.918) | 0.736 (range 0.730–0.742) |
| Mean paired ROC-AUC difference | +0.174 | — |
| Split-variation 95% t interval for difference | +0.137 to +0.211 | — |

There were 303 source rows (216 CAD labels; prevalence 0.713), three repeats of five paired outer folds, and 15 outer selection fits. No patient identifier is verified. The intervals describe only split variation and do not quantify patient-level or external-population uncertainty. The original Phase 5 ROC-AUC 0.928 was not used to define or select the candidate pool.

## Study-type conclusion

The evidence supports a **retrospective discrimination paper** describing the within-dataset selection-aware comparison and its limits. It does **not** support a prediagnostic prediction paper: no Phase 5 predictor has source-confirmed acquisition timing, and the highest marginal feature importance is chest-pain classification. No clinical utility or external-validation claim is supported. UCI remains a separate track; no UCI files were modified.

## Limitations and non-claims

- The estimand is performance across source-row splits; unique patient independence is unverified.
- Only three outer repeats were completed after the full 25-fold plan exceeded 35 CPU minutes without producing artifacts. The reduced analysis uses the first three exact V3 outer repeats and a deterministic 3-fold inner design.
- The selected feature group uses timing-uncertain variables; the demographic-only sensitivity does not establish a timing-confirmed comparator.
- PR-AUC is interpreted relative to the observed prevalence, and calibration/threshold metrics show fold variability.
- ROC-AUC improvement is not clinical utility and does not establish transportability or causal feature effects.

The historical Phase 1–5 outputs and Phase 5.1 audit artifacts were not overwritten. The Phase 6 manifest records the new files and source hashes.
