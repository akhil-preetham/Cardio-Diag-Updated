# Reproducing and extending the release

## Fast verification

Install Python 3.11 and `requirements-release.txt`, then run from the repository root:

```sh
python tools/portable_release.py --root . --check-runtime
python -m pytest -q
```

The portable verifier resolves recorded Windows paths **only into the current checkout**, checks the primary manifests and frozen protocol, and recalculates source/target metrics. It does not silently use an original workstation directory that happens to exist. Omitted-artifact allowances are explicit, and the standard release invocation allows none.

`.gitattributes` disables Git line-ending conversion because existing hashes identify exact bytes. Do not run automatic formatting or EOL rewriting over frozen experiment runners or artifacts and then continue claiming their old hashes apply.

## Training from scratch

The complete ZAS nested run is computationally heavier than verification. It retains six groups, five families, all tuning grids, three calibration components and 15 outer partitions. The completed workstation run used about 19 minutes of single-thread checkpoint computation; other machines vary.

The existing frozen namespaces are evidence. To train an independent new version, copy the executed runner into a **new module/output namespace**, set its new `OUT`, provide its protocol before fitting, and retain separate manifests. Do not modify frozen `transport.py`/`zas.py` in place and overwrite outputs. Fresh transport training uses only source data; target labels are reserved for evaluation after probabilities are frozen. New seeds on old data are not an independent clinical cohort.

The pinned release requirements are the supported path for saved model loading and the historical elastic-net API. The full original 128-distribution environment is recorded separately, including bootstrap/tools/optional packages. Installing the entire Windows snapshot on Linux is not the default setup. Additional historical notebook/model tracks may require optional libraries and separately acquired datasets; their source is preserved for research history.

## Report compilation

The final report was compiled with Tectonic 0.17.0 and IEEEtran, not a Markdown-to-PDF substitute. Editable `.tex`, `.md` and vector figure assets are under `CARDIODIAG_FRESH/paper/research_extension_v1/`. Install Tectonic separately; compiler binaries are not stored in Git. To compile the existing source without regenerating data:

```sh
cd CARDIODIAG_FRESH/paper/research_extension_v1
tectonic --keep-logs CARDIODIAG_RESEARCH_REPORT.tex
```

The figure/table generator uses the saved results. Its historical runtime/report-input records include workstation paths; relocation-aware primary verification is the supported fresh-clone audit. A recompiled report can have different PDF metadata/font/toolchain bytes and needs its own visual check and new output hash. Do not replace a historical PDF while keeping its old hash.

## Test scope and limitations

The default release suite covers the current primary evidence, component disjointness, metric aggregation, endpoint reconstruction, model-bundle parity, source-trained imputation, input validation and relocated-checkout verification. Legacy tests remain available explicitly, but require their original optional dependencies/data and some workstation environment records. The historical full-suite result is documented; it is not represented as the fresh-clone suite result.

Patient IDs, acquisition timing, target-code equivalence and the exact >=50% source versus >50% target boundary remain incomplete clinical evidence. Reproducible software and good internal scores do not resolve these facts. New code changes should be judged by a defined experiment and validation plan, not by chasing a higher AUC or declaring that no improvement is ever possible.
