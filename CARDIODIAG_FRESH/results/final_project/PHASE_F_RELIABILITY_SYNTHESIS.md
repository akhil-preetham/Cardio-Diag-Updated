# Phase F — Reliability and robustness synthesis

## ZAS

V3 includes calibration, uncertainty and robustness artifacts under `results/primary_research_v3/` and an independent audit under `results/primary_research_v3_independent_audit/`. Its reported calibrated ROC-AUC is 0.735 ± 0.060 across five outer folds; the source-row reuse, absent patient ID and internal-only validation limit uncertainty claims. The Phase 5 exploratory ZAS combined group shows Brier 0.0964 and log loss 0.3141, but those are post-selection, repeated-out-of-fold aggregate metrics for one selected group/model label and not an unbiased final-model estimate. No confidence interval accounts for selecting from 36 rows or refitting the full development process.

## UCI

Phase 4/4.3/5 provide discrimination, PR-AUC, Brier, log loss, calibration intercept/slope, fixed-bin calibration summaries, paired stage comparisons, OOD scores, selective-prediction summaries and threshold grids. Stage 2 and 4 site performance varies; Switzerland has 115/123 positives and only eight negatives, producing unstable specificity/calibration estimates. The stage comparison bootstrap resamples records and is not a centre-level uncertainty interval. Duplicate sensitivity in the old artifact is invalid for the reason in the metric reconciliation report.

## Boundaries

- Brier/log loss and calibration estimates summarize saved predictions; they do not imply calibrated deployment probabilities.
- Threshold metrics use 0.5 or fixed exploratory grids, not an independently justified operating threshold.
- OOD outputs measure feature-space atypicality, not disease probability.
- Selective prediction is descriptive and does not validate a safe clinical abstention mechanism.
- Decision-curve artifacts are not used to claim utility because no threshold range is clinically justified for deployment.
- Subgroup metrics are small, descriptive summaries, not a fairness certification.
- Patient-cluster and centre-level uncertainty are not established without patient identifiers and more independent centres.
