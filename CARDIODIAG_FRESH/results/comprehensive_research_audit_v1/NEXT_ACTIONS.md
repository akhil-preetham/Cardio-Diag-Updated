# Next actions

## First task

Implement an authoritative experiment registry and validation check that reconciles all reported metrics and paired-statistics artifacts.

## Required files

- New registry under `results/comprehensive_research_audit_v1/` or a separately versioned registry directory.
- Existing `results/experiment_metrics.json`.
- Existing corrected-validation reports.
- Existing `results/ensemble_experiments/**/manifest.json`.
- Existing `results/ensemble_postprocessing/paired_model_comparisons.csv`.
- `docs/EXPERIMENT_RESULTS.md`.
- `docs/FINAL_EXECUTION_VERIFICATION.md`.

## Acceptance criteria

- Every reported metric has exactly one authoritative source path.
- Every paired comparison records dataset, track, method, prediction path, and ordered row IDs.
- Contradictory “performed/not performed” claims are eliminated.
- No existing source or result is overwritten; corrections are versioned.
- A focused test fails when paired predictions have mismatched row IDs or unresolved provenance.

Do not begin new optimization or Problem 2 until this registry passes validation.
