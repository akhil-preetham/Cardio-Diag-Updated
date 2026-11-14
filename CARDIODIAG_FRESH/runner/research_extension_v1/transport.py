"""Frozen, source-only five-core ZAS-to-AngioCAD transport sensitivity study.

This is an exploratory endpoint-mismatched transport analysis, not validation of
the historical V3 or 38-feature model. No target labels enter any training step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT.parent / "research_publication_plan/data_screening/angiocad"
OUT = ROOT / "results/research_extension_v1/transport"
SOURCE = ROOT / "data/interim/validated/z_alizadeh_sani.csv"
SPLITS = ROOT / "results/methodology_corrections_v2/reproducibility/split_manifest_v2.json"
CORE = ["Age", "Sex", "DM", "HTN", "Current Smoker"]
BINARY = CORE[1:]
ARMS = ["raw_full_source_fit", "source_calibrated_ensemble"]
SEED = 826041
VESSELS = ["LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig", "Prox LCX", "Mid LCX", "Dist LCX", "OM", "Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"]
SEVERE = {"51-75", "76-90", "91-99", "100"}
ALLOWED_STENOSIS = SEVERE | {"NL", "1-25", "26-50"}
METRICS = ["roc_auc", "average_precision", "brier_score", "log_loss", "sensitivity_at_0_5", "specificity_at_0_5", "joint_calibration_intercept", "joint_calibration_slope"]
MCAR_RATES = [0.0, 0.1, 0.25, 0.5]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_workbook(path: Path) -> pd.DataFrame:
    """Read first worksheet without executing macros or formulas."""
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                value_node = cell.find("m:v", NS)
                value = value_node.text if value_node is not None else ""
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", NS).itertext())
                key = "".join(c for c in cell.get("r", "") if c.isalpha())
                values[key] = value
            rows.append(values)
    header = rows[0]
    return pd.DataFrame([{header.get(k, k).strip(): v for k, v in row.items()} for row in rows[1:] if row.get("A", "").strip()])


def normalize_core(frame: pd.DataFrame, *, source: bool) -> pd.DataFrame:
    """Fixed direct mappings; no label-based recoding, selection or imputation."""
    x = frame[CORE].copy().replace("", np.nan)
    x["Age"] = pd.to_numeric(x["Age"], errors="raise")
    if not x["Age"].dropna().between(18, 110).all():
        raise ValueError("Age outside frozen adult plausibility bounds 18..110")
    sexes = {"Male": 1.0, "Female": 0.0}
    if source:
        sexes["Fmale"] = 0.0  # literal local ZAS misspelling, not a learned code
    unknown = set(x["Sex"].dropna()) - set(sexes)
    if unknown:
        raise ValueError(f"Unsupported sex labels: {sorted(unknown)}")
    x["Sex"] = x["Sex"].map(sexes)
    for col in ["DM", "HTN", "Current Smoker"]:
        x[col] = pd.to_numeric(x[col], errors="raise")
        if not x[col].dropna().isin([0, 1]).all():
            raise ValueError(f"Unexpected {col} code")
    return x.astype(float)


def make_search(seed: int) -> GridSearchCV:
    preprocessor = ColumnTransformer([
        ("age", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["Age"]),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(solver="lbfgs", C=1, max_iter=3000, random_state=SEED))])
    return GridSearchCV(pipeline, {"model__C": [0.1, 1.0, 10.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="roc_auc", refit=True, n_jobs=1, error_score="raise")


def make_calibrated(seed: int) -> CalibratedClassifierCV:
    # Each complete search is cloned/fitted only on its calibration-training
    # subset. Each sigmoid is paired with that exact fitted search predictor;
    # no final full-data model is swapped into a small-fit calibrator.
    return CalibratedClassifierCV(estimator=make_search(seed + 1000), method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=seed), ensemble=True, n_jobs=1)


def endpoints(label_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not set(VESSELS).issubset(label_rows.columns):
        raise ValueError("Missing required artery labels")
    values = label_rows[VESSELS].fillna("").to_numpy(dtype=str)
    unknown = set(values.ravel()) - ALLOWED_STENOSIS
    if unknown:
        raise ValueError(f"Unknown/missing artery labels: {unknown}")
    y = np.isin(values, list(SEVERE)).any(axis=1).astype(int)
    ambiguous = (y == 0) & (values == "26-50").any(axis=1)
    return y, ambiguous


def joint_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Joint evaluation regression, not fitting an applied probability map."""
    z = np.column_stack([np.ones(len(y)), logit(np.clip(p, 1e-6, 1 - 1e-6))])
    def value(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z @ beta
        return float(np.mean(np.logaddexp(0, eta) - y * eta)), z.T @ (expit(eta) - y) / len(y)
    fitted = minimize(value, np.array([0.0, 1.0]), jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
    if not fitted.success and np.linalg.norm(fitted.jac, ord=np.inf) > 1e-6:
        return float("nan"), float("nan")
    return float(fitted.x[0]), float(fitted.x[1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    intercept, slope = joint_calibration(y, p)
    positive = y == 1
    negative = y == 0
    return {"n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])),
            "sensitivity_at_0_5": float((p[positive] >= .5).mean()), "specificity_at_0_5": float((p[negative] < .5).mean()),
            "joint_calibration_intercept": intercept, "joint_calibration_slope": slope}


def report_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable short names for reports, preserving explicit canonical columns."""
    aliases = {"arm": "model", "average_precision": "ap", "brier_score": "brier", "sensitivity_at_0_5": "sensitivity", "specificity_at_0_5": "specificity", "joint_calibration_intercept": "calibration_intercept", "joint_calibration_slope": "calibration_slope"}
    result = frame.copy()
    for original, alias in aliases.items():
        if original in result:
            result[alias] = result[original]
    return result


def bootstrap(y: np.ndarray, p: np.ndarray, *, draws: int, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    samples = []
    skipped = 0
    for _ in range(draws):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) != 2:
            skipped += 1
            continue
        samples.append(metrics(y[index], p[index]))
    results = []
    for name in METRICS:
        values = np.asarray([row[name] for row in samples], dtype=float)
        finite = values[np.isfinite(values)]
        results.append({"metric": name, "bootstrap_025": float(np.quantile(finite, .025)), "bootstrap_975": float(np.quantile(finite, .975)), "usable_draws": len(finite)})
    return results, {"requested_draws": draws, "single_class_draws_skipped": skipped, "metric_nonconvergence": {name: sum(not np.isfinite(row[name]) for row in samples) for name in METRICS}}


def frozen_config() -> dict:
    return {"version": "transport_v1", "source": "ZAS validated local CSV; Cath Cad=1/Normal=0",
            "target": "AngioCAD feature-label ID intersection; no image data",
            "features": CORE, "not_historical_v3": True,
            "mapping": {"Age": "numeric years; adult18..110 plausibility; no clipping; source median fitted only in training",
                        "Sex": "source Male=1,Fmale=0,Female=0; target Male=1,Female=0",
                        "DM": "direct same-name numeric0/1; convention retained without outcome-guided inversion",
                        "HTN": "direct same-name numeric0/1; convention retained without outcome-guided inversion",
                        "Current Smoker": "direct0/1; missing remains missing until source-training mode imputer"},
            "semantic_limit": "Binary direction follows conventional provider same-name0/1 fields; accessible AngioCAD metadata lacks explicit 0=no/1=yes codebook. Clinical ascertainment/timing equivalence unverified; exploratory compatibility-constrained transport only.",
            "source_documentation": ["https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani", "https://zenodo.org/records/15826856", "https://doi.org/10.1016/j.cmpb.2026.109368"],
            "outer": "first3repeats x5folds original frozen ZAS assignments; reused/exploratory",
            "search": {"family": "unbalanced L2 logistic", "C": [.1, 1, 10], "inner_folds": 3, "selection_metric": "roc_auc", "n_jobs": 1},
            "arms": {ARMS[0]: "full-source training GridSearchCV-refit raw uncalibrated probability", ARMS[1]: "three source-trained search+sigmoid pairs; each search only calibration-training subset; average calibrated probabilities"},
            "source_calibration": "ensemble=True; each full search nested within calibration-training folds; no target calibration",
            "arm_comparison_limit": "arms differ in training size, averaging and calibration; not a calibration-only causal comparison",
            "endpoints": ["base_gt50", "exclude_ambiguous26_50", "upper_assignment26_50_positive"],
            "endpoint_limits": "not identical ZAS>=50; upper assignment is sensitivity bound not true >=50; nonobstructive not healthy",
            "threshold": .5, "target_tuning": False, "target_calibration": False,
            "bootstrap_draws": 2000, "bootstrap_seed": SEED+9000,
            "uncertainty": "percentile record-bootstrap conditional on fixed source models and target case cohort; no model-refit/centre/patient-independence inference",
            "seed": SEED, "excluded": "all lab fields, Diagnosis, artery/angiographic fields, Gensini Score, LVEF/echo and medications",
            "source_mcar_stress": {"rates": MCAR_RATES, "arm": ARMS[0], "mask_unit": "independent per-test-cell MCAR", "seed": "826041+10000+100*repeat+fold; same Uniform(0,1) draw per cell nested across mask rates", "fit": "none; source-training imputers already fitted", "interpretation": "simulated random missingness only; not MNAR or real-site generalization proof"},
            "data_selection": "ID match only; no missing-feature complete-case exclusion; target missing values use source fitted imputers"}


def inputs() -> dict[str, str]:
    paths = [SOURCE, SPLITS, SCREEN/"AngioCAD_Features.xlsx", SCREEN/"AngioCAD_Labels.xlsx", SCREEN/"zenodo_record_15826856.json", Path(__file__)]
    return {str(path.resolve()): sha(path) for path in paths}


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "FROZEN_PROTOCOL.json"
    config = frozen_config()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current["config"] != config or current["input_hashes"] != inputs():
            raise RuntimeError("Frozen protocol/input mismatch; use new version for scientific changes")
    else:
        write_json(path, {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config, "input_hashes": inputs()})
    (OUT / "FROZEN_PROTOCOL.sha256").write_text(sha(path)+"  FROZEN_PROTOCOL.json\n", encoding="ascii")
    print("Frozen protocol SHA256", sha(path))


def outer_partitions(y: np.ndarray) -> list[dict]:
    original = json.loads(SPLITS.read_text(encoding="utf-8"))
    records = {}
    for row in original["outer_assignments"]:
        if row["repeat"] <= 3:
            records.setdefault((row["repeat"], row["fold"]), {"train": [], "test": []})[row["role"]].append(row["row_index"])
    result = []
    for (repeat, fold), roles in sorted(records.items()):
        train, test = sorted(roles["train"]), sorted(roles["test"])
        if set(train) & set(test) or set(train) | set(test) != set(range(len(y))):
            raise ValueError("Outer split invalid")
        result.append({"repeat": repeat, "fold": fold, "train": train, "test": test})
    if len(result) != 15:
        raise ValueError("Expected15outerpartitions")
    return result


def data_quality(source_x: pd.DataFrame, target_x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for name in CORE:
        s, t = source_x[name], target_x[name]
        rows.append({"feature": name, "source_n": len(s), "target_n": len(t), "source_missing": int(s.isna().sum()), "target_missing": int(t.isna().sum()), "source_min": float(s.min()), "source_max": float(s.max()), "target_min": float(t.min()), "target_max": float(t.max()), "source_mean_nonmissing": float(s.mean()), "target_mean_nonmissing": float(t.mean()), "target_outside_source_range": int(((t < s.min()) | (t > s.max())).sum())})
    def fingerprints(frame: pd.DataFrame) -> pd.Series:
        return frame.apply(lambda r: "|".join("NA" if pd.isna(v) else format(float(v), ".12g") for v in r), axis=1)
    source_f, target_f = fingerprints(source_x), fingerprints(target_x)
    overlap = {"feature_set": CORE, "source_duplicate_core_rows": int(source_f.duplicated().sum()), "target_duplicate_core_rows": int(target_f.duplicated().sum()), "shared_core_fingerprints": len(set(source_f)&set(target_f)), "source_rows_with_target_core_match": int(source_f.isin(set(target_f)).sum()), "target_rows_with_source_core_match": int(target_f.isin(set(source_f)).sum()), "interpretation": "Five common coarse fields can collide by chance. These are not verified duplicate persons, full-record duplicates or evidence of source-cohort overlap; source patient independence remains unverified."}
    return pd.DataFrame(rows), overlap


def run() -> None:
    freeze()
    if (OUT / "TARGET_METRICS.csv").exists():
        raise FileExistsError("Transport outputs already complete; refusing to overwrite; verify saved artifacts or use a new version")
    source = pd.read_csv(SOURCE)
    x = normalize_core(source, source=True)
    y = source.Cath.map({"Cad": 1, "Normal": 0})
    if y.isna().any() or len(y) != 303:
        raise ValueError("Unexpected source labels")
    y = y.to_numpy(dtype=int)
    features = load_workbook(SCREEN / "AngioCAD_Features.xlsx")
    labels = load_workbook(SCREEN / "AngioCAD_Labels.xlsx")
    if features.ID.duplicated().any() or labels.ID.duplicated().any():
        raise ValueError("ID duplication")
    joined = features[["ID"]+CORE].merge(labels[["ID"]+VESSELS], on="ID", validate="one_to_one", how="inner")
    target_x = normalize_core(joined, source=False)
    if len(joined) != 377:
        raise ValueError("Unexpected target cohort size")
    quality, overlap = data_quality(x, target_x)
    quality.to_csv(OUT / "CASE_MIX_AND_MISSINGNESS.csv", index=False)
    write_json(OUT / "CORE_FINGERPRINT_AUDIT.json", overlap)
    exclusions = [{"record_scope": "label_only", "target_case_id": key, "reason": "no_clinical_feature_ID"} for key in sorted(set(labels.ID)-set(features.ID))]
    exclusions += [{"record_scope": "feature_only", "target_case_id": key, "reason": "no_label_ID"} for key in sorted(set(features.ID)-set(labels.ID))]
    pd.DataFrame(exclusions, columns=["record_scope", "target_case_id", "reason"]).to_csv(OUT / "EXCLUSIONS.csv", index=False)
    source_predictions, source_folds, selections, memberships = [], [], [], []
    stress_predictions, stress_masks = [], []
    for split in outer_partitions(y):
        repeat, fold, train, test = split["repeat"], split["fold"], split["train"], split["test"]
        seed = SEED+100*repeat+fold
        for arm in ARMS:
            estimator = make_search(seed) if arm == ARMS[0] else make_calibrated(seed)
            estimator.fit(x.iloc[train], y[train])
            p = estimator.predict_proba(x.iloc[test])[:, 1]
            source_folds.append({"repeat": repeat, "fold": fold, "arm": arm, **metrics(y[test], p)})
            if arm == ARMS[0]:
                selections.append({"scope": "source_outer", "repeat": repeat, "fold": fold, "arm": arm, "component": 0, "C": estimator.best_params_["model__C"]})
                uniform = np.random.default_rng(SEED+10000+100*repeat+fold).random((len(test), len(CORE)))
                for rate in MCAR_RATES:
                    mask = uniform < rate
                    stressed = x.iloc[test].mask(mask)
                    stress_p = estimator.predict_proba(stressed)[:, 1]
                    stress_predictions.extend({"repeat": repeat, "fold": fold, "mask_rate": rate, "source_row_index": int(index), "y_true": int(y[index]), "probability": float(prob)} for index, prob in zip(test, stress_p))
                    for local_index, index in enumerate(test):
                        stress_masks.append({"repeat": repeat, "fold": fold, "mask_rate": rate, "source_row_index": int(index), **{f"mask_{name}": bool(mask[local_index, col]) for col, name in enumerate(CORE)}})
            else:
                cv = StratifiedKFold(3, shuffle=True, random_state=seed)
                for component, (fit, calibration) in enumerate(cv.split(x.iloc[train], y[train])):
                    if set(fit)&set(calibration):
                        raise RuntimeError("Calibration overlap")
                    selections.append({"scope": "source_outer", "repeat": repeat, "fold": fold, "arm": arm, "component": component, "C": estimator.calibrated_classifiers_[component].estimator.best_params_["model__C"]})
                    for role, local_ids in [("component_fit_and_inner_selection", fit), ("sigmoid_fit_only", calibration)]:
                        memberships.extend({"scope": "source_outer", "repeat": repeat, "fold": fold, "component": component, "source_row_index": train[int(i)], "role": role} for i in local_ids)
            source_predictions.extend({"repeat": repeat, "fold": fold, "arm": arm, "source_row_index": int(index), "y_true": int(y[index]), "probability": float(prob)} for index, prob in zip(test, p))
        print(f"Source outer partition {repeat}/{fold} complete", flush=True)
    source_pred = pd.DataFrame(source_predictions)
    source_pred.to_csv(OUT / "SOURCE_OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(source_folds).to_csv(OUT / "SOURCE_FOLD_METRICS.csv", index=False)
    source_repeats = []
    for (repeat, arm), g in source_pred.groupby(["repeat", "arm"]):
        if len(g) != 303 or g.source_row_index.nunique() != 303:
            raise RuntimeError("OOF source coverage error")
        source_repeats.append({"repeat": int(repeat), "arm": arm, **metrics(g.y_true.to_numpy(), g.probability.to_numpy())})
    report_aliases(pd.DataFrame(source_repeats)).to_csv(OUT / "SOURCE_REPEAT_METRICS.csv", index=False)
    stress = pd.DataFrame(stress_predictions)
    stress.to_csv(OUT / "SOURCE_MCAR_STRESS_PREDICTIONS.csv", index=False)
    pd.DataFrame(stress_masks).to_csv(OUT / "SOURCE_MCAR_MASKS.csv", index=False)
    stress_repeats = [{"repeat": int(repeat), "mask_rate": float(rate), "arm": ARMS[0], **metrics(g.y_true.to_numpy(), g.probability.to_numpy())} for (repeat, rate), g in stress.groupby(["repeat", "mask_rate"])]
    report_aliases(pd.DataFrame(stress_repeats)).to_csv(OUT / "SOURCE_MCAR_STRESS_REPEAT_METRICS.csv", index=False)
    # Target outcome derivation deliberately occurs after all source OOF work.
    # Full-source models are fixed and saved before target outcomes are scored.
    fitted = {}
    target_predictions = joined[["ID"]].rename(columns={"ID": "target_case_id"}).copy()
    for arm in ARMS:
        model = make_search(SEED) if arm == ARMS[0] else make_calibrated(SEED)
        model.fit(x, y)
        fitted[arm] = model
        target_predictions[arm] = model.predict_proba(target_x)[:, 1]
        if arm == ARMS[0]:
            selections.append({"scope": "full_source", "repeat": 0, "fold": 0, "arm": arm, "component": 0, "C": model.best_params_["model__C"]})
        else:
            for component, (fit, calibration) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y)):
                selections.append({"scope": "full_source", "repeat": 0, "fold": 0, "arm": arm, "component": component, "C": model.calibrated_classifiers_[component].estimator.best_params_["model__C"]})
                for role, ids in [("component_fit_and_inner_selection", fit), ("sigmoid_fit_only", calibration)]:
                    memberships.extend({"scope": "full_source", "repeat": 0, "fold": 0, "component": component, "source_row_index": int(i), "role": role} for i in ids)
    joblib.dump(fitted, OUT / "SOURCE_MODELS.joblib")
    pd.DataFrame(selections).to_csv(OUT / "SOURCE_SELECTIONS.csv", index=False)
    pd.DataFrame(memberships).to_csv(OUT / "SOURCE_CALIBRATION_MEMBERSHIP.csv", index=False)
    target_predictions.to_csv(OUT / "TARGET_PROBABILITIES_FROZEN_BEFORE_SCORING.csv", index=False)
    write_json(OUT / "PREDICTION_FREEZE.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "protocol_sha256": sha(OUT/"FROZEN_PROTOCOL.json"), "models_sha256": sha(OUT/"SOURCE_MODELS.joblib"), "probabilities_sha256": sha(OUT/"TARGET_PROBABILITIES_FROZEN_BEFORE_SCORING.csv"), "source_rows": len(x), "target_label_fit_count": 0})
    target_y, ambiguous = endpoints(joined)
    target_predictions["y_base_gt50"] = target_y
    target_predictions["boundary_ambiguous26_50"] = ambiguous
    target_predictions["case_id"] = target_predictions.target_case_id
    target_predictions["y_gt50"] = target_y
    target_predictions["boundary"] = ambiguous
    target_predictions["prob_raw"] = target_predictions[ARMS[0]]
    target_predictions["prob_calibrated"] = target_predictions[ARMS[1]]
    target_predictions.to_csv(OUT / "TARGET_PREDICTIONS.csv", index=False)
    variants = {"base_gt50": (np.ones(len(target_y), dtype=bool), target_y), "exclude_ambiguous26_50": (~ambiguous, target_y), "upper_assignment26_50_positive": (np.ones(len(target_y), dtype=bool), np.where(ambiguous, 1, target_y))}
    rows, interval_rows, bootstrap_info = [], [], []
    for variant, (keep, endpoint_y) in variants.items():
        for arm in ARMS:
            p = target_predictions.loc[keep, arm].to_numpy()
            evaluated_y = endpoint_y[keep]
            rows.append({"endpoint": variant, "arm": arm, **metrics(evaluated_y, p)})
            ci, info = bootstrap(evaluated_y, p, draws=2000, seed=SEED+9000)
            interval_rows.extend({"endpoint": variant, "arm": arm, **item} for item in ci)
            bootstrap_info.append({"endpoint": variant, "arm": arm, **info})
            print(f"Target fixed-model evaluation {variant}/{arm} complete", flush=True)
    report_aliases(pd.DataFrame(rows)).to_csv(OUT / "TARGET_METRICS.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(OUT / "TARGET_RECORD_BOOTSTRAP_INTERVALS.csv", index=False)
    write_json(OUT / "BOOTSTRAP_DIAGNOSTICS.json", bootstrap_info)
    (OUT / "INTERPRETATION.md").write_text("""# Shared-core transport interpretation and limitations

This is a new five-core model using Age, Sex, DM, HTN and Current Smoker. It is
neither historical V3 (which uses FH/EX-Smoker) nor the historical 38-feature model.
No AngioCAD label selected features, C, preprocessing, calibration or thresholds.
No target model updating was done. All source fitting precedes target scoring;
PREDICTION_FREEZE.json hashes the trained models and target probabilities.

The raw arm refits on all source rows. The sigmoid ensemble retains three matched
source search/calibrator pairs, each search independent of its calibration labels.
They differ in training sizes, model averaging and calibration; a difference in
performance cannot be attributed to calibration alone. Raw probabilities are not
explicitly calibrated. Neither arm is a clinically validated risk estimate.

Core field codes are mapped directly from observed named source/target columns;
binary presence direction follows conventional0/1, with no label-guided inversion.
The accessible target source has no explicit binary coding or timing dictionary.
The resulting analysis is exploratory and conditional on this semantic assumption.
Source timing and patient independence remain unverified. Target case keys do not
prove distinct patients. Shared coarse-core fingerprints are not duplicate people.

Target >50% endpoint differs from source catalogue >=50%; ten matched nonpositive
cases have 26-50 segments. Exclusion and upper assignment are sensitivity scenarios,
not validated true >=50 labels. Nonobstructive controls can have other coronary
conditions. The acute/stable referral spectrum differs from ZAS. Thirty-six label
cases lack clinical records; exclusion may produce selection. No prediagnostic,
population screening, any-dataset, clinical benefit or external validation of the
original fitted models is claimed.

Target intervals use2,000 record-resampling percentile bootstrap draws conditional
on fixed source fits and the available target cohort. They omit source fitting and
centre-sampling uncertainty; no patient-independence assumption is verified. The
three overlapping source repeat summaries describe split variation, not three
independent studies. Reused development data and known historical performance make
this a methodological sensitivity analysis rather than a newly untouched trial.

Joint calibration intercept/slope regressions are evaluation diagnostics, jointly
estimated on logit probabilities. They are not calibration-in-the-large with an
offset and are never applied to update target predictions. The0.5 threshold is
fixed without clinical justification. Model artifacts are research-only; load only
trusted local files in the recorded environment, never arbitrary external joblib.
""", encoding="utf-8")
    historical = {path: digest for path, digest in json.loads((OUT/"FROZEN_PROTOCOL.json").read_text())["input_hashes"].items() if path != str(Path(__file__).resolve())}
    if not all(sha(Path(path)) == digest for path, digest in historical.items()):
        raise RuntimeError("An input changed during analysis")
    write_json(OUT / "TRANSPORT_MANIFEST.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "protocol_sha256": sha(OUT/"FROZEN_PROTOCOL.json"), "environment": {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn_version, "numpy": np.__version__, "pandas": pd.__version__}, "target_label_fits": 0, "model_training_data": "source ZAS only", "input_hashes": historical, "outputs": {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "TRANSPORT_MANIFEST.json"}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if args.freeze:
            freeze()
        elif args.run:
            run()
        else:
            parser.error("Choose--freeze or--run")


if __name__ == "__main__":
    main()
