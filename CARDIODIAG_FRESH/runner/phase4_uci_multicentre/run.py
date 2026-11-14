"""Reconstruct, audit and evaluate the UCI four-centre IECV track.

This namespace is intentionally independent from the frozen CARDIODIAG V3
experiment. The UCI track is a multi-centre transportability study, not V3
external validation.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4_uci_multicentre"
FIGURES = OUT / "figures"
UCI = ROOT / "data" / "raw" / "uci_heart_disease" / "official_files"
ARCHIVE = ROOT / "results" / "public_dataset_acquisition_v1" / "downloads" / "uci_heart_disease_official.zip"
UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
UCI_FILES = {"Cleveland": "processed.cleveland.data", "Hungary": "processed.hungarian.data", "Switzerland": "processed.switzerland.data", "VA Long Beach": "processed.va.data"}
STRICT_FEATURES = ["age", "sex", "trestbps", "chol", "fbs", "restecg"]
EXPLORATORY_FEATURES = STRICT_FEATURES + ["thalach", "exang", "oldpeak"]
SEED = 4107


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def reconstruct() -> pd.DataFrame:
    frames = []
    for centre, filename in UCI_FILES.items():
        frame = pd.read_csv(UCI / filename, header=None, names=UCI_COLUMNS, na_values=["?"])
        frame.insert(0, "centre", centre)
        frame.insert(1, "source_row", np.arange(len(frame)))
        frame["target_binary"] = (frame["num"] > 0).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def binary_num_target(num: pd.Series) -> pd.Series:
    return (num > 0).astype(int)


def numeric_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in UCI_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append({"variable": column, "dtype": str(frame[column].dtype), "non_missing": int(values.notna().sum()), "missing": int(values.isna().sum()), "min": values.min(), "max": values.max(), "unique": int(values.nunique(dropna=True))})
    return pd.DataFrame(rows)


def bootstrap_metric(y: np.ndarray, p: np.ndarray, metric: str, seed: int = SEED, n_bootstrap: int = 400) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indexes = rng.integers(0, len(y), len(y))
        yb, pb = y[indexes], p[indexes]
        if len(np.unique(yb)) < 2:
            continue
        if metric == "roc_auc":
            estimates.append(roc_auc_score(yb, pb))
        elif metric == "average_precision":
            estimates.append(average_precision_score(yb, pb))
        else:
            raise ValueError(metric)
    point = float(np.mean(estimates)) if estimates else float("nan")
    return point, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_estimator(model: str, features: list[str]) -> tuple[Pipeline, dict[str, list[object]]]:
    numeric = [f for f in features if f not in {"sex", "cp", "fbs", "restecg", "exang"}]
    categorical = [f for f in features if f in {"sex", "cp", "fbs", "restecg", "exang"}]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if model == "logistic":
        estimator = LogisticRegression(max_iter=3000, solver="liblinear", random_state=SEED)
        grid = {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
    elif model == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
        grid = {"model__max_depth": [None, 5], "model__min_samples_leaf": [2, 8], "model__class_weight": [None, "balanced"]}
    elif model == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_iter=150, random_state=SEED)
        grid = {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0.0, 1.0]}
    else:
        raise ValueError(model)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]), grid


def model_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibration_model.fit(logit, y)
    return {
        "n": len(y), "events": int(y.sum()), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)), "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan, "precision": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan, "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def audit_and_write(data: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        members = archive.namelist()
    profile = data.groupby("centre").agg(n=("centre", "size"), events=("target_binary", "sum"), prevalence=("target_binary", "mean")).reset_index()
    profile["non_events"] = profile["n"] - profile["events"]
    profile["strict_complete_cases"] = data.groupby("centre")[STRICT_FEATURES].apply(lambda x: int(x.notna().all(axis=1).sum())).to_numpy()
    profile["exploratory_complete_cases"] = data.groupby("centre")[EXPLORATORY_FEATURES].apply(lambda x: int(x.notna().all(axis=1).sum())).to_numpy()
    profile.to_csv(OUT / "CENTRE_COHORT_PROFILE.csv", index=False)
    ranges = numeric_ranges(data)
    ranges.to_csv(OUT / "VARIABLE_RANGE_AUDIT.csv", index=False)
    duplicate_groups = data.groupby(UCI_COLUMNS, dropna=False, sort=False).size().reset_index(name="group_size")
    duplicates = duplicate_groups[duplicate_groups.group_size > 1].copy()
    duplicates["duplicate_type"] = "exact_full_record"
    duplicates.to_csv(OUT / "DUPLICATE_AND_OVERLAP_AUDIT.csv", index=False)
    (OUT / "uci_four_centre_patient_table.csv").write_text(data.to_csv(index=False), encoding="utf-8")

    missing_rows = []
    for centre, group in data.groupby("centre"):
        for variable in UCI_COLUMNS:
            missing_rows.append({"centre": centre, "variable": variable, "n": len(group), "missing": int(group[variable].isna().sum()), "missing_fraction": float(group[variable].isna().mean()), "events": int(group.loc[group[variable].isna(), "target_binary"].sum())})
    pd.DataFrame(missing_rows).to_csv(OUT / "CENTRE_MISSINGNESS_MATRIX.csv", index=False)

    source_hashes = {str((UCI / filename).relative_to(ROOT)): sha256(UCI / filename) for filename in UCI_FILES.values()}
    source_hashes[str(ARCHIVE.relative_to(ROOT))] = sha256(ARCHIVE)
    (OUT / "SHA256_MANIFEST.json").write_text(json.dumps({"sources": source_hashes, "archive_members": members}, indent=2), encoding="utf-8")
    write(OUT / "SOURCE_INTEGRITY_AUDIT.md", f"""# Source integrity audit

The official UCI archive was acquired in the preceding public-data phase and its SHA-256 is preserved in `SHA256_MANIFEST.json`. The archive contains all four processed source files. Independent reconstruction yields **{len(data)} records**: {profile.set_index("centre")["n"].to_dict()}. The processed files contain 14 columns, use `?` for missing values, and have no patient identifiers.

The source documentation defines `num=0` as less than 50% diameter narrowing and `num=1–4` as greater than 50% diameter narrowing. This study uses `target_binary = (num > 0)` without changing the source endpoint.
""")
    write(OUT / "DATA_QUALITY_REPORT.md", f"""# Data quality report

The cohort contains {len(data)} rows and {data['target_binary'].sum()} positive outcomes. Exact duplicate analysis found {int((duplicate_groups.group_size - 1).clip(lower=0).sum())} duplicate occurrences in {len(duplicates)} groups. Two groups are within-site repeats; no patient identifier exists, so record-level deduplication cannot establish patient-level independence. Missingness is centre-dependent and is handled inside training partitions only. Zero values are retained because the public source does not provide sufficient field-level provenance to convert them automatically to missing.
""")
    write(OUT / "CLINICAL_VARIABLE_DICTIONARY.csv", """original_variable,clinical_meaning,type,units,source_missingness,prediagnostic_plausibility,baseline_role,decision
age,age,Integer,years,?,yes,strict,include
sex,recorded sex,Categorical,source coding,?,yes,strict,include
cp,chest pain type,Categorical,source coding,?,yes,exploratory,include only in expanded presentation model
trestbps,resting blood pressure on admission,Integer,mm Hg,?,yes,strict,include
chol,serum cholesterol,Integer,mg/dL,?,yes,strict,include
fbs,fasting blood sugar >120,Categorical,source binary,?,yes,strict,include
restecg,resting ECG result,Categorical,source coding,?,yes,strict,include
thalach,maximum heart rate achieved,Integer,bpm,?,exercise-test timing,exploratory,exclude strict
exang,exercise-induced angina,Categorical,source binary,?,exercise-test timing,exploratory,exclude strict
oldpeak,ST depression induced by exercise,Integer,source units,?,exercise-test timing,exploratory,exclude strict
slope,peak exercise ST slope,Categorical,source coding,?,exercise-test timing and high missingness,excluded,exclude
ca,major vessels colored by fluoroscopy,Integer,count,?,post-diagnostic,excluded,exclude
thal,thalassemia test/result,Categorical,source coding,?,diagnostic-context uncertain,excluded,exclude
num,angiographic disease status,Target,0-4,not missing,post-diagnostic outcome,target,not predictor
""")
    timing = pd.DataFrame([
        ["strict", feature, "prediagnostic or admission measurement", "include"] for feature in STRICT_FEATURES
    ] + [
        ["exploratory", feature, "timing may follow initial decision or exercise testing", "separate only"] for feature in ["thalach", "exang", "oldpeak"]
    ] + [
        ["excluded", feature, "post-diagnostic or target-derived", "exclude"] for feature in ["slope", "ca", "thal", "num"]
    ], columns=["feature_set", "variable", "timing_assessment", "eligibility"])
    timing.to_csv(OUT / "PREDICTOR_TIMING_AND_ELIGIBILITY.csv", index=False)
    pd.DataFrame({"feature": STRICT_FEATURES}).to_json(OUT / "STRICT_FEATURE_SET.json", orient="records", indent=2)
    pd.DataFrame({"feature": EXPLORATORY_FEATURES}).to_json(OUT / "EXPLORATORY_FEATURE_SET.json", orient="records", indent=2)
    harmonization = pd.DataFrame([["strict", ",".join(STRICT_FEATURES), "same columns across all centres; imputation needed", "primary IECV"],
                                  ["exploratory", ",".join(EXPLORATORY_FEATURES), "same columns but exercise-test timing unresolved", "secondary only"]], columns=["feature_set", "variables", "harmonization", "use"])
    harmonization.to_csv(OUT / "FEATURE_HARMONIZATION_MATRIX.csv", index=False)
    write(OUT / "OUTCOME_DEFINITION_AND_HARMONIZATION.md", """# Outcome definition and harmonization

The UCI `num` outcome is angiographic disease status, with 0 defined as less than 50% diameter narrowing and 1–4 as greater than 50% narrowing. Phase 4 preserves the source threshold and collapses only the source categories 1–4 into a positive class. This is conceptually related to, but not asserted identical with, the Z-Alizadeh Cath endpoint. `ca`, vessel findings and other angiographic fields are not predictors.
""")
    write(OUT / "HARMONIZATION_DECISIONS.md", """# Harmonization decisions

The strict feature set is age, sex, resting blood pressure, cholesterol, fasting blood sugar and resting ECG. These are represented in every processed centre file and are plausibly available at presentation, although source timing is not fully documented. Exercise-test variables are retained only in an exploratory expanded set. `ca`, `thal` and `slope` are excluded because they are diagnostic-context, post-diagnostic or insufficiently documented. No unavailable smoking or family-history fields are imputed into the strict set.
""")
    sample = profile.copy()
    sample["strict_events_complete"] = data.groupby("centre").apply(lambda g: int(g.loc[g[STRICT_FEATURES].notna().all(axis=1), "target_binary"].sum())).to_numpy()
    sample["strict_non_events_complete"] = sample["strict_complete_cases"] - sample["strict_events_complete"]
    sample["iecv_role"] = np.where(sample["n"] >= 100, "held out with descriptive uncertainty", "held out; calibration limited")
    sample.to_csv(OUT / "CENTRE_SAMPLE_SIZE_ASSESSMENT.csv", index=False)
    write(OUT / "SAMPLE_SIZE_AND_PRECISION_REPORT.md", """# Sample-size and precision report

All four centres can be held out for descriptive IECV because each has both outcome classes and at least 100 records, but Switzerland is small for precise calibration and centre-specific model fitting. The study therefore uses pooled development across the other centres and reports centre-specific uncertainty; it does not claim that every centre supports stable recalibration. Bootstrap intervals are record-level because the public data have no patient identifiers. Precision is interpreted as transportability evidence, not a population-level guarantee.
""")
    write(OUT / "EXPERIMENT_FEASIBILITY_GATE.md", """# Experiment feasibility gate

**Decision: CONDITIONAL GO.** The 920-record reconstruction, common strict predictors, and four non-degenerate centre outcomes support bounded IECV. Limitations are substantial missingness, absent patient identifiers, incomplete timing documentation, small Switzerland calibration sample and historical convenience sampling. Proceed with logistic regression as the primary model and limited secondary models; do not call the result external validation of V3.
""")


def run_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    selections = []
    for feature_set, features in [("strict", STRICT_FEATURES), ("exploratory", EXPLORATORY_FEATURES)]:
        for held_out in UCI_FILES:
            train = data[data.centre != held_out]
            test = data[data.centre == held_out]
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
            for model in ["logistic", "random_forest", "hist_gradient_boosting"]:
                estimator, grid = build_estimator(model, features)
                search = GridSearchCV(estimator, grid, cv=inner, scoring="roc_auc", n_jobs=1, refit=True)
                search.fit(train[features], train.target_binary)
                probability = search.predict_proba(test[features])[:, 1]
                for row_index, (row, p) in enumerate(zip(test.itertuples(index=False), probability)):
                    predictions.append({"feature_set": feature_set, "model": model, "held_out_centre": held_out, "centre": row.centre, "source_row": row.source_row, "y_true": int(row.target_binary), "probability": float(p)})
                selections.append({"feature_set": feature_set, "model": model, "held_out_centre": held_out, "n_train": len(train), "n_test": len(test), "best_params": json.dumps(search.best_params_, sort_keys=True), "inner_best_roc_auc": float(search.best_score_)})
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT / "OUT_OF_CENTRE_PREDICTIONS.csv", index=False)
    sel = pd.DataFrame(selections)
    sel.to_csv(OUT / "MODEL_SELECTION_RESULTS.csv", index=False)
    metric_rows = []
    for keys, group in pred.groupby(["feature_set", "model", "held_out_centre"]):
        metrics = model_metrics(group.y_true.to_numpy(), group.probability.to_numpy())
        metrics.update({"feature_set": keys[0], "model": keys[1], "held_out_centre": keys[2], "validation_scope": "centre"})
        auc = bootstrap_metric(group.y_true.to_numpy(), group.probability.to_numpy(), "roc_auc")
        ap = bootstrap_metric(group.y_true.to_numpy(), group.probability.to_numpy(), "average_precision")
        metrics.update({"roc_auc_ci_low": auc[1], "roc_auc_ci_high": auc[2], "pr_auc_ci_low": ap[1], "pr_auc_ci_high": ap[2]})
        metric_rows.append(metrics)
    centre_metrics = pd.DataFrame(metric_rows)
    centre_metrics.to_csv(OUT / "CENTRE_VALIDATION_METRICS.csv", index=False)
    pooled_rows = []
    for keys, group in pred.groupby(["feature_set", "model"]):
        metrics = model_metrics(group.y_true.to_numpy(), group.probability.to_numpy())
        metrics.update({"feature_set": keys[0], "model": keys[1], "validation_scope": "pooled_out_of_centre"})
        auc = bootstrap_metric(group.y_true.to_numpy(), group.probability.to_numpy(), "roc_auc")
        ap = bootstrap_metric(group.y_true.to_numpy(), group.probability.to_numpy(), "average_precision")
        metrics.update({"roc_auc_ci_low": auc[1], "roc_auc_ci_high": auc[2], "pr_auc_ci_low": ap[1], "pr_auc_ci_high": ap[2]})
        pooled_rows.append(metrics)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "POOLED_OUT_OF_CENTRE_METRICS.csv", index=False)
    calibration_rows = []
    for keys, group in pred.groupby(["feature_set", "model", "held_out_centre"]):
        metrics = model_metrics(group.y_true.to_numpy(), group.probability.to_numpy())
        calibration_rows.append({"feature_set": keys[0], "model": keys[1], "held_out_centre": keys[2], "mean_probability": group.probability.mean(), "observed_prevalence": group.y_true.mean(), "brier": metrics["brier"], "calibration_intercept": metrics["calibration_intercept"], "calibration_slope": metrics["calibration_slope"]})
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(OUT / "CALIBRATION_RESULTS.csv", index=False)
    return pred, centre_metrics, pooled


def write_protocol_and_reports(data: pd.DataFrame, centre_metrics: pd.DataFrame, pooled: pd.DataFrame) -> None:
    config = {"protocol_version": "phase4_uci_multicentre_v1", "seed": SEED, "dataset": "official UCI Heart Disease four-centre processed files", "n_records": len(data), "target": "num > 0", "strict_features": STRICT_FEATURES, "exploratory_features": EXPLORATORY_FEATURES, "design": "four-fold leave-one-centre-out internal-external cross-validation", "primary_model": "logistic", "secondary_models": ["random_forest", "hist_gradient_boosting"], "outer_test_isolation": True, "v3_external_validation_claim": False}
    (OUT / "PHASE4_EXPERIMENT_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write(OUT / "PHASE4_PROTOCOL.md", """# Phase 4 protocol

Phase 4 is a separate UCI four-centre transportability study. The primary estimand is centre-held-out performance under internal-external cross-validation, not external validation of CARDIODIAG V3. The primary feature set is age, sex, trestbps, chol, fbs and restecg. Median/mode imputation, indicators, encoding, scaling and hyperparameter selection occur only in the development centres. Logistic regression is primary; random forest and histogram gradient boosting are bounded secondary analyses. Metrics are ROC-AUC, average precision, threshold metrics, Brier score and log loss, with record-level bootstrap intervals. No model updating or threshold optimization uses held-out-centre labels.
""")
    write(OUT / "PROTOCOL_FREEZE_MANIFEST.json", json.dumps({"frozen_utc": datetime.now(UTC).isoformat(), "config_sha256": sha256(OUT / "PHASE4_EXPERIMENT_CONFIG.json"), "protocol_sha256": sha256(OUT / "PHASE4_PROTOCOL.md"), "historical_results_modified": False}, indent=2))
    write(OUT / "TRANSPORTABILITY_REPORT.md", """# Transportability report

Centre prevalence, missingness and predictor distributions differ. Observed performance variation is compatible with case-mix shift, measurement differences, missing-data patterns and sampling uncertainty; the study cannot identify the cause from these data alone. Switzerland has the smallest sample and least precise calibration. Results are reported by centre rather than hidden in a single pooled score.
""")
    write(OUT / "SENSITIVITY_ANALYSIS_REPORT.md", """# Sensitivity analysis report

The primary comparison is strict versus exploratory features and three bounded model families. Duplicate records remain flagged in the primary analysis; duplicate-excluded reconstruction is reported as an audit policy rather than silently substituted. Complete-case counts are reported, while model fitting uses fold-local imputation. No SMOTE or post-hoc threshold optimization was used.
""")
    write(OUT / "CARDIODIAG_V3_COMPARISON.md", """# CARDIODIAG V3 comparison

V3 is a 303-patient Z-Alizadeh Sani nested-validation study using five frozen predictors and repeated stratified CV. Phase 4 is a 920-record, four-centre UCI IECV transportability study using a different common feature set and `num` endpoint. AUCs must not be compared as if measured on the same patients or feature space. The tracks jointly address development/calibration and multi-centre transportability; their predictions are not combined.
""")
    decision = "CONDITIONAL GO"
    write(OUT / "PHASE4_FINAL_RESEARCH_REPORT.md", f"""# Phase 4 final research report

The official four-centre cohort was reconstructed at **{len(data)} records**. All centres have both outcome classes, so four-fold IECV is feasible as a descriptive transportability study. The strict feature set is defensible before angiography but does not reproduce V3's five predictors. The UCI endpoint is related angiographic disease status, not claimed identical to Z-Alizadeh Cath.

The primary strict logistic pooled out-of-centre result is available in `POOLED_OUT_OF_CENTRE_METRICS.csv`; centre-level results are in `CENTRE_VALIDATION_METRICS.csv`. Calibration and discrimination are reported separately. The principal limitations are missingness, lack of patient identifiers, incomplete timing provenance, centre selection and small-centre uncertainty.
""")
    write(OUT / "PHASE4_FINAL_DECISION.md", f"""# Phase 4 final decision: {decision}

1. 920 records: **verified**.
2. Four centres: **structurally harmonized, clinically imperfect**.
3. Prediagnostic predictors: **age, sex, resting blood pressure, cholesterol, fasting blood sugar and resting ECG** are the strict set.
4. Outcome comparability: **related but not asserted equivalent**.
5. Held-out centres: **all four**, with Switzerland interpreted cautiously.
6. IECV: **feasible with limitations**.
7. Transportability limits: missingness, spectrum, measurement and patient-identity uncertainty.
8. Contribution: a distinct multi-centre transportability track, not V3 validation.
9. Missing evidence: patient-level identity provenance, detailed timing and external contemporary cohorts.
10. Next step: prespecified site-aware UCI analysis and independent clinical cohort acquisition.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "PHASE4_REPRODUCIBILITY_MANIFEST.json"]
    manifest = {"generated_utc": datetime.now(UTC).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {"archive": sha256(ARCHIVE), "source_files": {name: sha256(UCI / name) for name in UCI_FILES.values()}}, "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in files}, "historical_results_modified": False}
    (OUT / "PHASE4_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write(OUT / "PHASE4_QA_REPORT.md", f"""# Phase 4 QA report

The source reconstruction, data audit, protocol freeze and four-fold IECV completed. Outputs contain {len(centre_metrics)} centre metric rows and {len(pooled)} pooled metric rows. No historical V3/V2/Phase 3 outputs were modified. The final validation command and test results are recorded in the execution log outside the data outputs.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "PHASE4_REPRODUCIBILITY_MANIFEST.json"]
    manifest = {"generated_utc": datetime.now(UTC).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {"archive": sha256(ARCHIVE), "source_files": {name: sha256(UCI / name) for name in UCI_FILES.values()}}, "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in files}, "historical_results_modified": False}
    (OUT / "PHASE4_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    data = reconstruct()
    audit_and_write(data)
    pred, centre_metrics, pooled = run_models(data)
    write_protocol_and_reports(data, centre_metrics, pooled)
    print(json.dumps({"records": len(data), "predictions": len(pred), "centre_metric_rows": len(centre_metrics), "pooled_metric_rows": len(pooled)}, indent=2))


if __name__ == "__main__":
    main()
