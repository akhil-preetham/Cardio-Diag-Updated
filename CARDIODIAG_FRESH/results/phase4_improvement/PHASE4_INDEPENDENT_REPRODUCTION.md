# Independent Phase 4 reproduction

The saved Phase 4 predictions were independently read and reconstructed against the four-centre source table. Expected prediction count is 5520; observed count is 5520. Metric recomputation is saved in `METRIC_RECOMPUTATION_AUDIT.csv`. The independent checks confirm prediction completeness and metric agreement within numerical tolerance.

The audit can verify saved predictions and metric arithmetic, but it cannot prove preprocessing provenance from predictions alone. Source-code inspection confirms that the Phase 4 pipeline places imputation, scaling and model selection inside the development-centre fit.
