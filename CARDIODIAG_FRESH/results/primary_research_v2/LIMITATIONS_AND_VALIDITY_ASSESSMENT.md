# Limitations and Validity Assessment

- This is repeated internal validation on 303 records, not independent external validation.
- Source-row labels are not patient identifiers; patient-level grouping cannot be proven from the derivative.
- Calibration metrics and ten-bin ECE are unstable at outer-fold sample sizes.
- Conditional bootstrap summaries do not capture refitting or population sampling uncertainty.
- Predictive permutation importance is not causal evidence.
- Strict-feature timing is conservative but not completely documented.
