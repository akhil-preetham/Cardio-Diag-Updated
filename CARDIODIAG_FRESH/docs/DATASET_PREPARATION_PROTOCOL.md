# Dataset preparation protocol

Official archives are downloaded into `data/raw/` and never edited. ZIP contents are extracted for inspection. Validated CSV derivatives are deterministic copies with no automatic row deletion. Quality reports record missingness, duplicates, identifiers, ranges, quantiles, constants, and dataset-specific implausible-value counts. Disease-presence targets are not merged with mortality targets. Any harmonization must use an explicit clinical mapping and remain derived.

The script is `runner/data/prepare_dataset_collection.py` and uses `.venv-repro`.
