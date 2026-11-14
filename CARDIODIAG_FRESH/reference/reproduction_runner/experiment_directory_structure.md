# Experiment directory structure

```text
reference/reproduction_runner/
  audit/
  config/
    experiment_config.template.json
  work/
    author_faithful/
      base_reproduction/
      runs/<run_id>/
    compatibility_adapted/
      base_reproduction/
      runs/<run_id>/
    corrected_validation/
      code/
      runs/<run_id>/
  results/
    author_faithful/<run_id>/
    compatibility_adapted/<run_id>/
    corrected_validation/<run_id>/
  logs/<run_id>/
```

Original extracted notebooks and `reference/author_release` remain outside writable run directories. Each run stores source hashes, dataset hashes, environment metadata, stdout, stderr, warnings, traceback, cell status, metrics, predictions, and a manifest.
