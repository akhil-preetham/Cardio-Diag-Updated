# Uncertainty and Bootstrap Audit

- **Verified correct:** uncertainty resamples the 303 unique source row indices after aggregating repeated-CV predictions per row.
- **Correct with limitations:** this avoids counting the five repeated predictions for the same row as five bootstrap observations.
- **Requires correction in interpretation:** the intervals quantify resampling uncertainty conditional on the observed cohort and averaged repeated-CV predictions. They do not quantify full model-fitting, split, hyperparameter, or calibration uncertainty and are not population-level confidence intervals.
- **Unresolved:** a patient-level bootstrap cannot be independently implemented because no patient identifier is available in the modeling derivative.
- **Verified correct:** bootstrap samples lacking both outcome classes are skipped; 500 successful-replicate targets are recorded.
