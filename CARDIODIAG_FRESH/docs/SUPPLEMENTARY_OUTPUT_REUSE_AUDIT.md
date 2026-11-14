# Supplementary output reuse audit

The original extracted notebooks contain saved outputs, and several contain metric-like text, plots, or exported artifact paths. Those outputs were inventoried in `results/supplementary_output_reuse_audit.csv`.

They were not promoted to experimental results because the saved outputs lack a verified controlled-run manifest, complete input hashes, aligned observation identifiers across models, and a proven historical environment. Reusing them as if newly reproduced would conflate author-saved artifacts with this project's execution.

## Feasible reuse

- Existing LIME HTML/PDF/CSV artifacts can be preserved as provenance references.
- Existing calibration text can be compared qualitatively after exact model and sample alignment is verified.
- Existing subgroup spreadsheets can be inspected manually, but they do not establish the corrected-validation subgroup protocol.

## Not defensibly reusable without additional evidence

- OVEC/OSEC published-performance claims.
- Paired bootstrap, DeLong, or Wilcoxon comparisons.
- LIME explanations as held-out explanations.
- LoRAS/XAIH-DRC implementation fidelity.
- Clinical utility or decision-curve claims.

No supplementary notebook was rerun solely to obtain a superficially complete table.
