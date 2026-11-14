# Executive Methodology Review v2

The Version 1 experiment is retained as a valid preliminary internal-validation baseline with explicit limitations. No new model training was launched.

## Decisions

- Preserve the five-feature strict primary analysis.
- Prohibit treating SVM expit-transformed scores as native probabilities.
- Use nested sigmoid calibration as the primary future probability protocol.
- Treat current bootstrap intervals and paired comparisons as descriptive.
- Persist exact inner folds and source-row reproducibility labels in future runs.
- Do not claim population confidence intervals, clinical utility, or external validation.

The full policy tables and exact split-manifest module are versioned in this directory and `runner/methodology_corrections_v2/`.
