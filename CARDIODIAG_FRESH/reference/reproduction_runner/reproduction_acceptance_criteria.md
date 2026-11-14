# Reproduction acceptance criteria

## Source integrity

Pass requires the original notebook hash to equal `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`, the prepared copy hash to equal `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`, and the author repository to remain clean. Any changed source cell must have a manifest entry.

## Environment reproducibility

Pass requires a recorded Python version, operating system, package versions, dependency check, dataset hashes, and external-file manifest. A package set that has not been verified against the source is labelled unverified, not historically compatible.

## Notebook execution

Successful execution requires every approved cell to complete, no unrecorded exception, outputs written only to the dedicated run directory, and a complete machine-readable manifest. A controlled failure is still a valid execution observation but is not a successful reproduction.

## Metric reproduction

A reported metric is reproduced only when the same dataset identity, target definition, split protocol, estimator, preprocessing, random seed, metric definition, and evaluation population are established. Compare metric value, rounding, sample count, and uncertainty where available. Accuracy alone is insufficient.

## Partial and failed reproduction

Partial reproduction means an independently verified prefix completed but the notebook later failed or diverged. Failed reproduction means the required source or environment cannot produce the reported result. Both require the exact failing cell and traceback.

## Corrected validation

Corrected validation is accepted only as a separate result: preprocessing and SMOTE are training-fold scoped, model selection cannot use the final test set, evaluation is stratified and independent, and metrics include appropriate discrimination, calibration, threshold, and uncertainty measures.

## External validation

External validation requires a distinct dataset, documented target and population compatibility, no fitting on the external test data, and uncertainty estimates. It must not be conflated with internal resampling or author-faithful results.
