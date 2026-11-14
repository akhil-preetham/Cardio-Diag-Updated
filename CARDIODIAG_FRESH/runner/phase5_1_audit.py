"""Read-only Phase 5.1 audit of the ZAS eligible_combined result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase5_1_audit"
ZAS_RUNNER = ROOT / "runner" / "phase5" / "zas_track" / "run.py"
MODEL_RUNNER = ROOT / "runner" / "phase5" / "common" / "models.py"
METRICS_RUNNER = ROOT / "runner" / "phase5" / "common" / "metrics.py"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"

FEATURES = [
    ("Age", "Patient age in years", "Demographic measurement", "A", "Age is necessarily known before the Cath outcome, although the exact recording time is not documented."),
    ("Sex", "Recorded sex", "Demographic measurement", "A", "Sex is available before the Cath outcome; source coding is imperfectly documented."),
    ("DM", "Diabetes mellitus indicator", "Medical history", "B", "Clinically prediagnostic, but the source does not establish whether history was recorded before the diagnostic work-up."),
    ("HTN", "Hypertension indicator", "Medical history", "B", "Clinically prediagnostic, but acquisition timing is undocumented."),
    ("Current Smoker", "Current smoking indicator", "Risk history", "B", "Clinically prediagnostic, but interview timing and ascertainment are undocumented."),
    ("EX-Smoker", "Former smoking indicator", "Risk history", "B", "Clinically prediagnostic, but interview timing and ascertainment are undocumented."),
    ("FH", "Family-history indicator", "Risk history", "B", "Clinically prediagnostic, but source timing and definition are undocumented."),
    ("Typical Chest Pain", "Typical chest-pain indicator", "Symptom", "B", "Could be available at presentation; the source does not document whether it was recorded before diagnostic decisions."),
    ("Dyspnea", "Dyspnea indicator", "Symptom", "B", "Could be available at presentation; timing is undocumented."),
    ("Function Class", "Functional class", "Symptom/functional status", "B", "May reflect established heart-failure severity or clinical assessment; prediction-time availability is uncertain."),
    ("Atypical", "Atypical chest-pain indicator", "Symptom", "B", "Could be available at presentation; timing is undocumented."),
    ("Nonanginal", "Nonanginal symptom indicator", "Symptom", "B", "Could be available at presentation; timing is undocumented."),
    ("Exertional CP", "Exertional chest-pain indicator", "Symptom", "B", "Could be available at presentation; timing is undocumented."),
    ("LowTH Ang", "Low-threshold angina indicator", "Symptom", "B", "Angina classification may be part of diagnostic assessment; timing is undocumented."),
    ("BP", "Blood pressure", "Vital sign", "B", "Likely obtainable at presentation, but the source does not document measurement timing or whether treatment occurred first."),
    ("PR", "Pulse rate", "Vital sign", "B", "Likely obtainable at presentation, but measurement timing is undocumented."),
    ("Edema", "Peripheral edema indicator", "Examination", "B", "Could be a presentation examination finding; timing and ascertainment are undocumented."),
    ("Q Wave", "Q-wave ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but ECG timing relative to diagnostic work-up is undocumented."),
    ("St Elevation", "ST-elevation ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("St Depression", "ST-depression ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("Tinversion", "T-wave inversion ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("LVH", "Left-ventricular-hypertrophy ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("Poor R Progression", "Poor R-wave progression ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("BBB", "Bundle-branch-block ECG finding", "Resting ECG", "B", "Potentially available before angiographic confirmation, but timing is undocumented."),
    ("FBS", "Fasting blood sugar", "Laboratory", "B", "May be baseline laboratory information, but fasting status, units and timing are undocumented."),
    ("CR", "Creatinine", "Laboratory", "B", "May be baseline laboratory information, but timing and clinical context are undocumented."),
    ("TG", "Triglycerides", "Laboratory", "B", "May be baseline laboratory information, but fasting status and timing are undocumented."),
    ("LDL", "Low-density lipoprotein", "Laboratory", "B", "May be baseline laboratory information, but timing and calculation/measurement details are undocumented."),
    ("HDL", "High-density lipoprotein", "Laboratory", "B", "May be baseline laboratory information, but fasting status and timing are undocumented."),
    ("BUN", "Blood urea nitrogen", "Laboratory", "B", "May be baseline laboratory information, but timing and clinical context are undocumented."),
    ("ESR", "Erythrocyte sedimentation rate", "Laboratory", "B", "May be available before angiography, but ordering and timing are undocumented."),
    ("HB", "Hemoglobin", "Laboratory", "B", "May be baseline laboratory information, but timing is undocumented."),
    ("K", "Potassium", "Laboratory", "B", "May be baseline laboratory information, but timing and treatment context are undocumented."),
    ("Na", "Sodium", "Laboratory", "B", "May be baseline laboratory information, but timing and treatment context are undocumented."),
    ("WBC", "White blood cell count", "Laboratory", "B", "May be baseline laboratory information, but timing is undocumented."),
    ("Lymph", "Lymphocyte measure", "Laboratory", "B", "May be baseline laboratory information, but timing and unit definition are undocumented."),
    ("Neut", "Neutrophil measure", "Laboratory", "B", "May be baseline laboratory information, but timing and unit definition are undocumented."),
    ("PLT", "Platelet count", "Laboratory", "B", "May be baseline laboratory information, but timing is undocumented."),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.DataFrame(FEATURES, columns=["feature", "source_definition", "clinical_group", "classification", "timing_evidence"])
    feature_frame.to_csv(OUT / "ZAS_FEATURE_TIMING_TABLE.csv", index=False)
    benchmark = pd.read_csv(ROOT / "results" / "phase5" / "zas_track" / "ZAS_MODEL_BENCHMARK.csv")
    target_rows = benchmark[(benchmark.feature_group == "eligible_combined") & benchmark.model.isin(["logistic", "ridge"])]
    selections = pd.read_csv(ROOT / "results" / "phase5" / "zas_track" / "ZAS_HYPERPARAMETER_SELECTION.csv")
    (OUT / "ZAS_FEATURE_TIMING_AUDIT.md").write_text(
        f"""# ZAS eligible_combined feature-timing audit

## Result traced

The reported logistic/ridge row is reproduced in `results/phase5/zas_track/ZAS_MODEL_BENCHMARK.csv` with {len(target_rows)} matching rows. The exact feature list is hard-coded at `runner/phase5/zas_track/run.py:20-27`; it contains {len(FEATURES)} columns and is listed in `ZAS_FEATURE_TIMING_TABLE.csv`. No feature list is inferred from column names.

The included variables are demographics, medical history, symptoms/examination, resting ECG and laboratory values. The code does **not** include EF-TTE, Region RWMA, VHD, Cath, angiographic measurements or vessel-count fields. Those excluded fields are specifically identified by the prior clinical feature audit as downstream or target-derived risks.

## Classification

- **A:** Age and Sex are clearly available before the diagnostic outcome.
- **B:** All other included variables remain timing-uncertain. They are clinically plausible presentation/history/ECG/laboratory variables, but the local source documentation does not establish that they were recorded before diagnostic decisions or before angiographic work-up.
- **C:** No included variable is conclusively shown by the available documentation to be post-diagnostic, but Function Class and some ECG/laboratory fields could become C if source acquisition records show diagnostic-work-up timing.
- **D:** No direct target-derived field is included in `eligible_combined`; Cath and downstream echo/angiographic fields are excluded.

The complete feature-by-feature evidence is in `ZAS_FEATURE_TIMING_TABLE.csv`. Prior repository evidence explicitly classifies EF-TTE and Region RWMA as diagnostic/downstream and Cath as the target; it does not provide sufficient timing evidence to upgrade the included B features to A.
""",
        encoding="utf-8",
    )
    (OUT / "ZAS_LEAKAGE_AUDIT.md").write_text(
        f"""# ZAS leakage and validation audit

## Implementation evidence

1. The target is mapped with `frame["Cath"].map({{"Cad": 1, "Normal": 0}})` at `runner/phase5/zas_track/run.py:31-35`.
2. The six feature groups and six model families are declared before the loops at `run.py:20-28`. `eligible_combined` is not assembled dynamically from test outcomes.
3. The outer design is `RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=5107)` at `run.py:38`. Each outer training partition is separated from its test partition at `run.py:41-43`.
4. `GridSearchCV` performs three-fold inner selection at `run.py:44-46`. The estimator is a pipeline; imputation, scaling, spline transformation and one-hot encoding are therefore fitted inside each inner training fold by `runner/phase5/common/models.py:11-40`.
5. No automated feature selection occurs. The feature list is fixed before each model fit. Consequently, feature-set selection across the six groups is **not** nested as a separate model-selection layer.
6. Predictions are written at `run.py:50-52`; repeated predictions are averaged per `(group, model, row_index)` at `run.py:53`. This correctly returns one aggregated prediction per the 303 source rows for each group/model, but repeated folds are not independent patients.
7. Metrics are computed from the aggregated predictions at `run.py:54-58` using `runner/phase5/common/metrics.py:19-40`. Calibration intercept and slope are descriptive evaluation statistics, not tuning operations.

## Integrity conclusions

- Fold-local preprocessing: **supported by code**.
- Inner hyperparameter selection: **nested within each outer training partition**.
- Outer test leakage during fitting: **not identified**.
- Fold-local feature selection: **not applicable; features were fixed**.
- Patient-level aggregation: **one row per source patient after repeat averaging**; no patient identifier exists beyond the source row index.
- Independent evaluation unit: **303 source rows**, not 606 repeated predictions.
- Direct target leakage in `eligible_combined`: **not identified**.

## Material limitation

The output compares six feature groups and six model families, producing 36 benchmark rows and {len(selections)} outer selection records. The row with ROC-AUC 0.928 was identified after viewing this benchmark table. Inner nesting protects hyperparameter selection within a candidate, but it does not protect the subsequent choice of the best feature group/model from selection optimism. The result is therefore not an unbiased estimate of a prespecified final model.

The output also records `optimism_correction` as not implemented. Repeated out-of-fold averaging reduces dependence from repeated predictions, but it does not remove candidate-selection optimism or resolve feature-timing uncertainty.
""",
        encoding="utf-8",
    )
    target_text = target_rows.to_string(index=False)
    (OUT / "ZAS_0928_EVIDENCE_DECISION.md").write_text(
        f"""# Decision on the ZAS ROC-AUC 0.928 result

## Classification: **EXPLORATORY ONLY**

The result is not classified as invalid arithmetic or proven direct leakage. The implementation excludes the obvious target-derived and downstream fields, and the candidate-level pipeline uses fold-local preprocessing and inner hyperparameter selection. However, it cannot support a defensible prediagnostic CAD claim because:

1. {len(FEATURES) - 2} of {len(FEATURES)} included variables are only clinically plausible, not source-documented as available before diagnostic confirmation.
2. The feature group/model row was selected from 36 benchmark candidates after observing performance; feature-group choice was not nested.
3. The cohort has no patient identifier and the source timing, units and ascertainment definitions are incomplete.
4. The reported confidence is based on repeated-CV prediction aggregation, not an untouched independent cohort.

## Traced benchmark rows

```text
{target_text}
```

## Required corrective action

Do not present 0.928/0.967/0.096 as the primary prediagnostic model result. Freeze a prediction-time data dictionary, obtain source evidence for each B feature, exclude unresolved downstream/timing variables, prespecify one feature group and model-selection protocol, and evaluate it in a genuinely independent cohort. Until then, retain the result as an exploratory hypothesis-generating analysis alongside the frozen five-predictor V3 reference.

## Reproducibility evidence

Source hashes used by this audit:

```json
{json.dumps({str(path.relative_to(ROOT)): sha256(path) for path in [DATA, ZAS_RUNNER, MODEL_RUNNER, METRICS_RUNNER]}, indent=2)}
```
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
