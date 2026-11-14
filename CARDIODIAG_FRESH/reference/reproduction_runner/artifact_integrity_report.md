# Artifact integrity report

## Required artifacts

All ten required deliverables are readable and substantively populated. The first three are stored in the preserved base-reproduction package:

- `work/author_faithful/base_reproduction/historical_environment_feasibility.md`
- `work/author_faithful/base_reproduction/three_track_execution_design.md`
- `work/author_faithful/base_reproduction/category_b_resolution_matrix.csv`

The remaining seven are under `reference/reproduction_runner/`, including the experiment template under `config/`.

## Matrix checks

- Source execution plan: 18 rows.
- Readiness matrix: 18 rows.
- Unique readiness paths: 18.
- Duplicate readiness paths: none.
- Readiness path set equals the source execution-plan path set: yes.
- Every readiness path exists under `reference/extracted_author_archives`: yes.
- Category B matrix: 7 data rows with a valid quoted CSV header.
- Experiment template: parses as JSON and contains required safety fields.
- Template execution flag: `false`.

No corrected readiness matrix was necessary.

## Cross-report consistency

The reports consistently state that execution is static-only, historical compatibility is not fully proven, and no reproduction result has been established. They distinguish author-faithful, compatibility-adapted, and corrected-validation tracks.

One pre-existing transcription inconsistency was found: `base_notebook_environment.txt` contains an incorrect working-copy hash substring. The separately named `base_notebook_environment_corrected.txt` records the authoritative hash without overwriting the prior record.

## Path and readiness interpretation

The readiness matrix identifies the base notebook as the first candidate but does not mark it executable automatically. All other notebooks remain blocked pending dependencies, external-file provenance, Colab-path handling, specialized-method review, or statistical/clinical assumptions.
