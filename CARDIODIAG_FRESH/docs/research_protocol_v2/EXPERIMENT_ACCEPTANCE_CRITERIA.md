# Experiment Acceptance Criteria

An experiment may be accepted for scientific reporting only if:

1. The raw dataset and historical artifacts remain unchanged.
2. The target and strict feature list match the frozen protocol.
3. Exact outer and inner assignments, seeds, source-row labels, and configuration are saved.
4. Split validation proves no train/test or inner/out-of-outer-training overlap.
5. All preprocessing, resampling, selection, and calibration are fold-local.
6. Every expected model-fold prediction is present exactly once.
7. Failures and warnings are recorded rather than hidden.
8. Metrics independently recompute from saved predictions.
9. Native scores and calibrated probabilities are labeled distinctly.
10. Calibration metrics use held-out predictions and ECE limitations are reported.
11. Uncertainty is labeled as conditional resampling or validation variation, not population inference.
12. Statistical comparisons account for paired/dependent predictions and multiplicity.
13. Explainability is clearly predictive, not causal.
14. Manuscript numbers match artifacts and contain no unsupported clinical or external-validation claim.
15. Input/output hashes and environment metadata are recorded.

Failure of any criterion requires correction or explicit deferral before primary claims.
