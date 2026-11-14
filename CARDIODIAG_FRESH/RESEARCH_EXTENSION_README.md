# CARDIODIAG completed computational extension

This extension adds information-matched controls, selection-independent calibration, a new public-clinical-cohort transport analysis and a missing-input stress test. It preserves historical results and does not turn the project into a clinically validated diagnostic tool.

## Read the report

The final 6-8 page report and editable sources are under `paper/research_extension_v1/`. Its tables and figures are generated from saved results. The compiled IEEEtran PDF is a research report; authorship, institutional declarations and a target venue's submission requirements require human completion.

## Experiments and artifacts

| Location | Content |
|---|---|
| `runner/research_extension_v1/zas.py` | Full 30-candidate corrected calibration study; same-feature logistic control; fixed information-group ablations |
| `results/research_extension_v1/zas/` | Frozen inputs/protocol, immutable component checkpoints, predictions, metrics, selection scores and traces |
| `runner/research_extension_v1/transport.py` | Source-only shared-core development, frozen AngioCAD scoring, three endpoint assignments, record bootstrap and MCAR stress |
| `results/research_extension_v1/transport/` | Models, prediction freeze, source/target predictions, bootstrap metrics, membership and missingness records |
| `results/research_extension_v1/provenance/` | Fresh official ZAS source comparison and independent raw-data checks |
| `results/research_extension_v1/INDEPENDENT_REVIEW.md` | Independent implementation/data/result review |
| `runner/research_extension_v1/verify.py` | Read-only final manifest and primary-metric verification; no model fitting |
| `runner/research_extension_v1/predict_research.py` | Strict-schema research scoring using only the trusted hash-checked local five-core model bundle |

## Verify without retraining

Run in PowerShell from this directory with the existing Python 3.11 environment:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m runner.research_extension_v1.verify
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

Bug-focused lint for the new runners/tests is recorded separately from legacy formatting/style findings. Do not change frozen executed files merely to make an old manifest appear valid after a style edit.

## Reproduce model training

The ZAS full candidate study takes approximately 20 minutes on the observed machine with a single compute thread; timing varies. Commands use source hashes and immutable outputs. The ZAS runner resumes matching checkpoints and refuses inconsistent replacements; the completed transport runner refuses an overwrite. For a fresh independent reproduction, copy the code/data to a new isolated workspace and use an empty new output namespace. Do not delete historical outputs to make room.

```powershell
.\.venv\Scripts\python.exe -m runner.research_extension_v1.zas
.\.venv\Scripts\python.exe -m runner.research_extension_v1.transport --help
```

Read the transport CLI help and frozen protocol before a fresh run; the no-overwrite guard is intentional. No AngioCAD outcome was used to tune, impute, calibrate or update source models.

## Research scoring example

The example CSV contains two synthetic schema examples, not patient records. Supply exactly Age, Sex, DM, HTN and Current Smoker. Sex must be Male/Female; binary fields use 0/1. Empty cells remain missing until source-trained imputation. The output contains probabilities and input-quality flags, without clinical diagnosis or patient-specific reliability claims.

```powershell
.\.venv\Scripts\python.exe -m runner.research_extension_v1.predict_research --input paper\research_extension_v1\example_input.csv --output results\research_extension_v1\my_new_research_scores.csv
```

The tool rejects existing output files and arbitrary model paths. Keep the completed local `SOURCE_MODELS.joblib` and its `PREDICTION_FREEZE.json` together. Hash checking identifies byte changes; it is not independent proof of model authenticity or clinical safety.

## Scope and honest failure boundaries

The transported shared core uses DM/HTN instead of the original V3 former-smoker/family-history fields. AngioCAD's >50% segment definition differs from ZAS's catalogue >=50% rule; ten target cases are boundary-ambiguous, and non-obstructive cases are not necessarily healthy controls. Timing, clinical coding equivalence and unique-patient independence remain incompletely verified. Results therefore describe compatibility-constrained retrospective transport, not external validation of historical 38-feature weights.

The source-only shared model's discrimination and probability accuracy deteriorated on the target cohort. Simulated missing values also reduced source discrimination. These findings are part of the completed study. They must not be removed to advertise reliable predictions for arbitrary datasets.

Original CardioDiag architectures belong to the original authors. Existing original-data, UCI, ZAS and newly acquired AngioCAD tracks remain distinct. Author order, ethics determination, funding/conflicts, distribution permissions and publication acceptance cannot be established by completing code or generating a PDF.
