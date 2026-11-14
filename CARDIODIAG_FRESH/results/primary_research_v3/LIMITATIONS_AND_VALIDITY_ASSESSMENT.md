# V3 Limitations and Validity Assessment

- The final estimator is single and refitted on all outer-training data; calibration scores come from fold-specific selected models.
- Score-scale differences between fold-specific calibration models and the final refitted estimator may affect calibration.
- Repeated CV reuses records and no patient identifier is available.
- Bootstrap summaries are conditional source-row resampling and do not capture refitting or population uncertainty.
- Ten-bin ECE is supplementary and unstable for small outer test folds.
- Permutation importance is predictive, not causal.
