# Explainability, reliability, fairness, and clinical utility audit

## Explainability

LIME and XAIH-DRC artifacts exist under `results/lime_experiments/` and `results/xaih_drc_experiments/`. LIME is a local, model-dependent explanation and must not be described as causal. The repository itself notes that independent model/input alignment was not fully established. Every explanation should be linked to a persisted model hash, row index, transformed feature schema, and generation seed.

## Calibration and uncertainty

`src/cardiodiag/calibration/metrics.py` computes ten-bin ECE/MCE and outputs exist in `results/ensemble_postprocessing/`. These are useful diagnostics, but calibration estimates from one internal holdout are uncertain. No independent calibration cohort or formal uncertainty interval for calibration is established.

## Fairness

`runner/fairness_ensemble.py` and subgroup artifacts provide descriptive Sex/gender metrics with minimum subgroup size 20. They do not establish fairness certification: subgroup confidence intervals, multiplicity control, prespecified clinical groups, and adequate sample sizes are not demonstrated for all claims.

## Clinical utility

`src/cardiodiag/clinical_utility/decision_curve.py` computes net benefit over thresholds. Decision curves depend on calibrated probabilities, prevalence, threshold interpretation, and harm/benefit assumptions. Existing curves are internal diagnostics and do not support clinical effectiveness or deployment claims.

## Required evidence

Persist model/input provenance, add calibration uncertainty, prespecify subgroup analyses, report subgroup counts and intervals, and externally validate probabilities before making reliability or clinical-utility claims.
