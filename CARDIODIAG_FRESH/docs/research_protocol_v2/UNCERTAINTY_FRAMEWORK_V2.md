# Uncertainty Framework v2

## Reportable from Version 1

Report outer-fold and repeat variation descriptively. A row-aggregated bootstrap of
repeated-CV predictions may be reported as conditional resampling uncertainty, provided
it is labeled as such. It does not represent an independently sampled population.

## Sampling uncertainty

The resampling unit is the patient. The current derivative has no patient identifier, so
the available bootstrap can only use source rows and must not claim patient-level
independence beyond the documented 303-patient source description.

## Algorithmic uncertainty

Bootstrap resampling of saved predictions does not refit preprocessing, hyperparameters,
models, or calibration. It therefore does not capture full algorithmic uncertainty. A
future experiment must refit the complete pipeline within each bootstrap sample if that
uncertainty is required.

## Predictive uncertainty

Individual predictive intervals are not estimated from Version 1. They require a
specified probabilistic or conformal method with calibration and coverage validation.

## Validation variability

Repeated CV reuses the same participants, so repeat/fold variation is dependent and is
not variation across independent cohorts. No Version 1 interval may be called a
conventional population confidence interval.
