# CARDIODIAG - Reproduction and validation

This workspace separates author-faithful reproduction, compatibility-adapted work, and corrected scientific validation.

## Completed experiments

- Author-faithful Five States base notebook: executed in `.venv-repro`, accuracy 0.8922 and ROC-AUC 0.9555, with warnings documented.
- Corrected Five States validation: 10 repeated stratified 75:25 holdouts with fold-local preprocessing and SMOTE.
- Corrected Mendeley validation: same protocol, with `patientid` excluded.
- Controlled OVEC/OSEC ensemble runs for Five States and Mendeley across
  author-faithful, compatibility-adapted, and corrected tracks.
- Training-only LoRAS comparison, XAIH-DRC execution, persisted LIME
  explanations, paired comparisons, calibration, and decision-curve outputs.

## Results

See [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md), [`results/experiment_metrics.json`](results/experiment_metrics.json), and [`results/dataset_comparison.csv`](results/dataset_comparison.csv).

## Structure

- `reference/reproduction_runner/audit/`: 18-notebook inventory and evidence audit.
- `reference/reproduction_runner/author_faithful/`: copied notebook runs and logs.
- `reference/reproduction_runner/results/`: corrected-validation JSON and predictions.
- `runner/`: guarded execution and validation scripts.
- `results/ensemble_experiments/`: persisted ensemble models, predictions,
  search records, metrics, and figures.
- `results/lime_experiments/`, `results/xaih_drc_experiments/`: explainability
  artifacts and feature-selection manifests.
- `docs/`: methodology, results, limitations, manuscript mapping, and audit records.
- `paper/manuscript.tex`: newly authored manuscript source using only verified project results.

## Safety and limitations

The author runner uses an exact prepared-copy allowlist, source/dataset hash
checks, isolated outputs, and explicit approval gates. Notebook execution gates
remain disabled. The corrected repeated holdouts overlap and are not
independent external validation. See
[`docs/FINAL_EXECUTION_VERIFICATION.md`](docs/FINAL_EXECUTION_VERIFICATION.md)
for execution evidence and remaining blockers.
