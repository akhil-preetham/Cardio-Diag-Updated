# Phase 4.1/4.2 final report

The existing Phase 4 analysis was independently reproduced: 144 metric checks were generated and prediction completeness was verified. The weak strict pooled result is explained by severe prevalence/missingness shift and low negative counts in Switzerland, not by an identified arithmetic or obvious split-leakage defect.

Staged IECV was executed for four clinical stages using regularized logistic regression. Stage performance is in `STAGEWISE_PERFORMANCE_RESULTS.csv`; centre-wise reliability is in `RELIABILITY_METRICS.csv`. Advanced Firth, GAM, Bayesian and conformal analyses were not executed because their implementations or exchangeability/calibration requirements were not defensible in this environment.

No independent external validation occurred. The output is a research prediction system with substantial transportability limitations, not a clinical deployment model.
