# CARDIODIAG — updated research implementation

A reproducible retrospective coronary artery disease study extending the original [CardioDiag work by Khan, Raza and Nadeem](https://doi.org/10.7717/peerj-cs.4059). This repository contains the project source, frozen research evidence, release verification tools and the final eight-page IEEE-style report.

**Authors:** Erukudinnela Akhil Preetham; Chunduri Hari Krishna; G Vennela.

**Institution:** Department of Computer Science and Engineering, Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT), Vijayawada, Andhra Pradesh, India.

**Project guide:** K. Naga Gopi.

## Report and findings

Read the [eight-page report](CARDIODIAG_FRESH/paper/research_extension_v1/CARDIODIAG_RESEARCH_REPORT.pdf), [editable LaTeX](CARDIODIAG_FRESH/paper/research_extension_v1/CARDIODIAG_RESEARCH_REPORT.tex), or [Markdown](CARDIODIAG_FRESH/paper/research_extension_v1/CARDIODIAG_RESEARCH_REPORT.md).

| Completed experiment | Main finding |
|---|---|
| Equal-information comparison | Mean ZAS AUC 0.917447 for 38-feature L2 logistic versus 0.910316 for historical automated selection |
| Information ablation | Removing ten symptom/examination fields reduced mean AUC to 0.846566; risk plus symptoms alone gave 0.903558 |
| Selection-independent calibration | Matched calibrated/raw ensemble Brier 0.103731/0.105460; improvement in two repeats and deterioration in one |
| New shared-core cohort evaluation | Source mean AUC 0.805884; AngioCAD AUC 0.712471 with a conditional record-bootstrap interval of 0.647437–0.773000 |
| Missing-input stress | Mean source AUC declined to 0.687544 under 50% random cell deletion |

These results **do not establish an automated-selection advantage, universally reliable probabilities, clinical deployment readiness or prospective prediction**. The new AngioCAD model uses a different five-field core, not the historical V3 or 38-field model. Endpoint, coding, timing, population and patient-independence limitations are reported explicitly. Mixed and negative findings are preserved.

## Quick start

Use **Python 3.11**. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-release.txt
python tools/portable_release.py --root . --check-runtime
python -m pytest -q
```

The default tests cover the published primary study and portable release. They read saved results, test data/partition integrity and use small synthetic checks; **they do not rerun the full nested benchmark**. The original workstation suite passed 116 tests with one optional historical LoRAS dependency skipped. Historical code/tests are retained, but some older tracks need separately acquired datasets or optional packages and are outside the default release suite.

The isolated publication environment passed **39 tests with one optional historical LoRAS test skipped**, and verified all **163 primary hash entries** with no missing-file allowances. All 153 published Python modules passed syntax parsing. See [release checks](RELEASE_CHECKS.md).

The frozen runtime matters: the saved model and historical elastic-net configuration were produced with scikit-learn 1.9.1. The portable tools check core package versions before model loading. The release requirements use an exact transitive constraint file. The older unpinned requirements are retained as historical files, not the recommended release environment.

## Research scoring

The model interface accepts exactly `Age, Sex, DM, HTN, Current Smoker`. Sex uses `Male/Female`; binary fields use `0/1`. Missing cells remain missing until the source-trained imputer handles them. The example contains synthetic rows, not patient records:

```sh
python tools/portable_predict.py --root . --input CARDIODIAG_FRESH/paper/research_extension_v1/example_input.csv --output research_scores.csv
```

The tool validates the schema, checks the trusted local model bundle/runtime, refuses existing output files and reports missingness/source-range flags. Its probabilities are **research outputs, not a diagnosis or patient-specific reliability guarantee**. Do not load arbitrary external joblib/pickle files.

## Reproducing the full experiment

The completed design uses 15 paired outer partitions, 45 corrected calibration components and the original 30 group-family candidates. The observed run took approximately 19 minutes of checkpoint computation on a single thread, plus verification and reporting. All fixed comparisons run even if performance decreases.

Historical protocols/manifests are byte-preserved, including the original workstation paths. Use the portable verifier above after cloning: the old workstation-only verifier is retained for provenance and is not the clone entry point. A completed transport namespace intentionally refuses overwrite. For new training, create a new output namespace and freeze a new protocol rather than editing saved results or pretending an altered run was historical. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository contents

- `CARDIODIAG_FRESH/src`, `runner`, `tests`, `experiments`, `config`: project implementation and research workflows.
- `CARDIODIAG_FRESH/notebooks`, `reference`: retained notebook/source tracks; published notebooks have output cells and execution state cleared.
- `CARDIODIAG_FRESH/results`: historical evidence and the completed `research_extension_v1` study.
- `CARDIODIAG_FRESH/paper/research_extension_v1`: final report, editable sources, figures and numeric traceability.
- `research_publication_plan/data_screening/angiocad`: small official CC BY 4.0 clinical/label workbooks, provenance and screening script.
- `tools`: relocation-safe verification and guarded research scoring.

Virtual environments, caches, compiler binaries, multi-gigabyte imaging archives and unrelated raw-data downloads are omitted. Source code is retained even for optional historical tracks. See [DATA_SOURCES.md](DATA_SOURCES.md), [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) and `SOURCE_PUBLICATION_MAP.csv` for provenance and exclusions.

## Remaining boundaries

No finite review can establish that research admits no further improvement. The release criteria are a tested, reproducible implementation with traceable results and honest limitations. Additional compatible patient-level cohorts and documented predictor timing would strengthen the evidence. Institutional ethics determination, corresponding author, contributions, funding/conflicts and venue-specific requirements still require author confirmation before a journal submission. Publication acceptance and clinical validation are not claimed.
