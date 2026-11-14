"""Selection-aware nested validation for retrospective ZAS discrimination.

The complete candidate-selection procedure is repeated inside each outer fold.
No outer-test labels enter preprocessing, fitting, candidate choice, calibration,
or threshold selection. Source rows, not verified patients, are the split unit.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/interim/validated/z_alizadeh_sani.csv"
SPLIT_PATH = ROOT / "results/methodology_corrections_v2/reproducibility/split_manifest_v2.json"
OUT = ROOT / "results/phase6_zas_selection"
SEED = 61051
TARGET = "Cath"
V3_FEATURES = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]

# Frozen Phase 5 candidate universe. Logistic and Ridge are one implementation;
# only the unique logistic family is retained. No candidate is added mid-run.
GROUPS: dict[str, list[str]] = {
    "v3_five_predictor": V3_FEATURES,
    "basic_risk": ["Age", "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH"],
    "symptoms_examination": ["Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "BP", "PR", "Edema"],
    "resting_ecg": ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"],
    "laboratory": ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
    "eligible_combined": ["Age", "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "BP", "PR", "Edema", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
}
MODEL_GRIDS: dict[str, tuple[str, dict[str, list[object]]]] = {
    "logistic_l2": ("logistic", {"model__C": [0.1, 1.0, 10.0]}),
    "elastic_net": ("elastic_net", {"model__C": [0.1, 1.0, 10.0], "model__l1_ratio": [0.25, 0.75]}),
    "spline_logistic": ("spline_logistic", {"model__C": [0.1, 1.0, 10.0]}),
    "random_forest": ("random_forest", {"model__max_depth": [None, 5], "model__min_samples_leaf": [2, 8]}),
    "hist_gradient_boosting": ("hist_gradient_boosting", {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}),
}
FAMILY_ORDER = list(MODEL_GRIDS)
V3_MODEL_ID = "v3_logistic_balanced"
V3_GRID = {"model__C": [0.1, 1.0, 10.0]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits(path: Path, y: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("n_rows") != 303 or manifest.get("outer_folds") != 5 or manifest.get("inner_folds") != 4:
        raise ValueError("Unexpected frozen V3/Stage 6 split manifest dimensions")
    outer: dict[tuple[int, int], dict[str, set[int]]] = {}
    inner: dict[tuple[int, int, int], dict[str, set[int]]] = {}
    for row in manifest["outer_assignments"]:
        outer.setdefault((row["repeat"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    for row in manifest["inner_assignments"]:
        inner.setdefault((row["repeat"], row["outer_fold"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    if len(outer) != 25 or len(inner) != 100:
        raise ValueError("Frozen split manifest must contain 25 outer and 100 inner folds")
    outer_records: list[dict] = []
    inner_splits: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (repeat, fold), roles in sorted(outer.items()):
        if repeat > 3:
            continue
        train, test = roles["train"], roles["test"]
        if train & test or train | test != set(range(303)):
            raise ValueError(f"Invalid outer partition {repeat}/{fold}")
        ordered_train = sorted(train)
        x_positions = np.arange(len(ordered_train))
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + repeat * 100 + fold)
        folds = [(fit_pos, valid_pos) for fit_pos, valid_pos in splitter.split(x_positions, y[ordered_train])]
        outer_records.append({"repeat": repeat, "fold": fold, "train": ordered_train, "test": sorted(test), "seed": manifest["outer_seeds"][repeat - 1]})
        inner_splits[(repeat, fold)] = folds
    return outer_records, inner_splits


def make_pipeline(frame: pd.DataFrame, features: list[str], family: str, *, v3: bool = False) -> Pipeline:
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(frame[col]) and col != "Sex"]
    categorical = [col for col in features if col not in numeric]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", add_indicator=not v3))]
    if family == "spline_logistic":
        numeric_steps.append(("spline", SplineTransformer(n_knots=3, degree=2)))
    numeric_steps.append(("scale", StandardScaler()))
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not v3)
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", categorical_encoder)]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if v3:
        classifier = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=1)
    elif family in {"logistic", "spline_logistic"}:
        classifier = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED)
    elif family == "elastic_net":
        classifier = LogisticRegression(solver="saga", l1_ratio=0.25, max_iter=3000, random_state=SEED)
    elif family == "random_forest":
        classifier = RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=1)
    elif family == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
    else:
        raise ValueError(family)
    return Pipeline([("preprocess", transformer), ("model", classifier)])


def candidate_specs(frame: pd.DataFrame) -> list[dict]:
    specs = []
    for group, features in GROUPS.items():
        for family in FAMILY_ORDER:
            kind, grid = MODEL_GRIDS[family]
            specs.append({"group": group, "family": family, "features": features, "pipeline": make_pipeline(frame, features, kind), "grid": grid})
    return specs


def search_candidate(spec: dict, x: pd.DataFrame, y: np.ndarray, cv: list[tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    search = GridSearchCV(spec["pipeline"], spec["grid"], scoring="roc_auc", cv=cv, n_jobs=1, refit=True, error_score="raise", return_train_score=False)
    search.fit(x, y)
    return search


def choose_candidate(specs: list[dict], x: pd.DataFrame, y: np.ndarray, cv: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, GridSearchCV, pd.DataFrame]:
    records = []
    best_spec = None
    best_search = None
    best_score = -np.inf
    for spec in specs:
        search = search_candidate(spec, x[spec["features"]], y, cv)
        score = float(search.best_score_)
        records.append({"group": spec["group"], "model_family": spec["family"], "inner_roc_auc": score, "best_params": json.dumps(search.best_params_, sort_keys=True)})
        # Stable tie resolution follows the protocol's declared group/family order.
        if score > best_score + 1e-12:
            best_score, best_spec, best_search = score, spec, search
    assert best_spec is not None and best_search is not None
    return best_spec, best_search, pd.DataFrame(records)


def v3_spec(frame: pd.DataFrame) -> dict:
    return {"group": "v3_five_predictor", "family": V3_MODEL_ID, "features": V3_FEATURES, "pipeline": make_pipeline(frame, V3_FEATURES, "logistic", v3=True), "grid": V3_GRID}


def choose_v3(x: pd.DataFrame, y: np.ndarray, cv: list[tuple[np.ndarray, np.ndarray]], spec: dict) -> tuple[dict, GridSearchCV]:
    return spec, search_candidate(spec, x[spec["features"]], y, cv)


def raw_probability(estimator: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return estimator.predict_proba(x)[:, 1]


def crossfit_selected_scores(spec: dict, search: GridSearchCV, x_train: pd.DataFrame, y_train: np.ndarray, local_splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Cross-fit the already inner-selected pipeline using outer-training rows only."""
    scores = np.full(len(y_train), np.nan)
    for fit_pos, valid_pos in local_splits:
        estimator = make_pipeline(x_train, spec["features"], "logistic" if spec["family"] == V3_MODEL_ID else MODEL_GRIDS.get(spec["family"], ("logistic",))[0], v3=spec["family"] == V3_MODEL_ID)
        estimator.set_params(**search.best_params_)
        estimator.fit(x_train.iloc[fit_pos][spec["features"]], y_train[fit_pos])
        scores[valid_pos] = raw_probability(estimator, x_train.iloc[valid_pos][spec["features"]])
    if np.isnan(scores).any():
        raise RuntimeError("Inner cross-fit did not cover every outer-training row")
    return scores


def fit_sigmoid(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrator.fit(scores.reshape(-1, 1), y)
    return calibrator


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    youden = tpr[finite] - fpr[finite]
    best = np.flatnonzero(np.isclose(youden, np.max(youden)))
    # Highest threshold on exact ties, fixed before outer evaluation.
    return float(np.max(thresholds[finite][best]))


def calibration_stats(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(logit(clipped).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metric_record(y: np.ndarray, p: np.ndarray, threshold: float, *, repeat: int, fold: int, model_id: str, selected_group: str, selected_family: str) -> dict:
    pred = p >= threshold
    positive = y == 1
    negative = ~positive
    intercept, slope = calibration_stats(y, p)
    return {
        "repeat": repeat, "fold": fold, "model_id": model_id, "selected_group": selected_group,
        "selected_model_family": selected_family, "n": len(y), "events": int(y.sum()),
        "prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "pr_auc_prevalence_reference": float(y.mean()), "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "threshold_selected_inner": float(threshold),
        "sensitivity": float(pred[positive].mean()) if positive.any() else math.nan,
        "specificity": float((~pred[negative]).mean()) if negative.any() else math.nan,
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def summarize_predictions(predictions: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for (repeat, fold, model_id), group in predictions.groupby(["repeat", "fold", "model_id"], sort=True):
        meta = selections[(selections.repeat == repeat) & (selections.fold == fold) & (selections.model_id == model_id)].iloc[0]
        fold_rows.append(metric_record(group.y_true.to_numpy(dtype=int), group.calibrated_probability.to_numpy(dtype=float), float(meta.threshold_selected_inner), repeat=int(repeat), fold=int(fold), model_id=model_id, selected_group=meta.selected_group, selected_family=meta.selected_model_family))
    folds = pd.DataFrame(fold_rows)
    # Within each repeat every source row appears once. These are split-repeat
    # estimates; repeated rows are never pooled as independent patients.
    repeat_rows = []
    for (repeat, model_id), group in predictions.groupby(["repeat", "model_id"], sort=True):
        meta = selections[(selections.repeat == repeat) & (selections.model_id == model_id)]
        # Each row was predicted once in this repeat, with fold-specific threshold.
        row_threshold = group[["row_index", "threshold_selected_inner"]].drop_duplicates("row_index")
        joined = group.merge(row_threshold, on="row_index", suffixes=("", "_x"))
        threshold = joined.threshold_selected_inner.to_numpy(dtype=float)
        y = joined.y_true.to_numpy(dtype=int)
        p = joined.calibrated_probability.to_numpy(dtype=float)
        intercept, slope = calibration_stats(y, p)
        pred = p >= threshold
        repeat_rows.append({"repeat": int(repeat), "model_id": model_id, "n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)), "pr_auc_average_precision": float(average_precision_score(y, p)), "pr_auc_prevalence_reference": float(y.mean()), "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), "sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()), "calibration_intercept": intercept, "calibration_slope": slope, "mean_selected_threshold": float(meta.threshold_selected_inner.mean())})
    repeats = pd.DataFrame(repeat_rows)
    overall_rows = []
    for model_id, group in repeats.groupby("model_id"):
        row = {"model_id": model_id, "repeat_count": len(group)}
        for metric in ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity", "calibration_intercept", "calibration_slope"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean_across_repeats"] = float(values.mean())
            row[f"{metric}_sd_across_repeats"] = float(values.std(ddof=1))
            row[f"{metric}_min_repeat"] = float(values.min())
            row[f"{metric}_max_repeat"] = float(values.max())
            # A t interval summarizes split-repeat variation, not patient-level
            # sampling uncertainty; explicitly labeled that way in the report.
            half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan
            row[f"{metric}_split_variation_95ci_low"] = float(values.mean() - half) if math.isfinite(half) else math.nan
            row[f"{metric}_split_variation_95ci_high"] = float(values.mean() + half) if math.isfinite(half) else math.nan
        row["prevalence"] = float(group.prevalence.mean())
        row["pr_auc_prevalence_reference"] = float(group.pr_auc_prevalence_reference.mean())
        overall_rows.append(row)
    return folds, repeats, pd.DataFrame(overall_rows)


def run() -> None:
    if OUT.exists() and any(path.name not in {"SELECTION_AWARE_PROTOCOL.md", ".work"} for path in OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Phase 6 output namespace: {OUT}")
    frame = pd.read_csv(DATA_PATH)
    if frame.shape[0] != 303 or TARGET not in frame:
        raise ValueError("Unexpected local ZAS dataset shape/target")
    expected = set().union(*(set(features) for features in GROUPS.values()))
    if not expected.issubset(frame.columns):
        raise ValueError(f"Candidate features missing from source: {sorted(expected - set(frame.columns))}")
    if any(feature in expected for feature in ["EF-TTE", "Region RWMA", "VHD", TARGET]):
        raise ValueError("A diagnostic/downstream or target field entered the candidate pool")
    y = frame[TARGET].map({"Cad": 1, "Normal": 0})
    if y.isna().any():
        raise ValueError("Unexpected Cath values")
    y = y.to_numpy(dtype=int)
    outer, inner_map = load_splits(SPLIT_PATH, y)
    specs = candidate_specs(frame)
    baseline_spec = v3_spec(frame)
    predictions: list[dict] = []
    selections: list[dict] = []
    frequencies: list[dict] = []
    calibration_rows: list[dict] = []
    inner_oof_records: list[dict] = []
    permutation_rows: list[dict] = []
    work_dir = OUT / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, int]] = set()
    for checkpoint_path in sorted(work_dir.glob("outer_*.pkl")):
        state = pickle.loads(checkpoint_path.read_bytes())
        completed.add((state["repeat"], state["fold"]))
        predictions.extend(state["predictions"])
        selections.extend(state["selections"])
        frequencies.extend(state["frequencies"])
        calibration_rows.extend(state["calibration_rows"])
        inner_oof_records.extend(state["inner_oof_records"])
        permutation_rows.extend(state["permutation_rows"])
    for split in outer:
        repeat, fold = split["repeat"], split["fold"]
        if (repeat, fold) in completed:
            continue
        train_rows, test_rows = split["train"], split["test"]
        cv = inner_map[(repeat, fold)]
        x_train_all = frame.iloc[train_rows]
        y_train = y[train_rows]
        x_train_local = x_train_all.reset_index(drop=True)
        candidate, candidate_search, candidate_scores = choose_candidate(specs, x_train_local, y_train, cv)
        v3, v3_search = choose_v3(x_train_local, y_train, cv, baseline_spec)
        for item in candidate_scores.to_dict("records"):
            inner_oof_records.append({"repeat": repeat, "fold": fold, **item})
        candidate_score = crossfit_selected_scores(candidate, candidate_search, x_train_local, y_train, cv)
        v3_score = crossfit_selected_scores(v3, v3_search, x_train_local, y_train, cv)
        candidate_y_local = y_train
        calibration_selection = [{"calibration_fold": index, "group": candidate["group"], "model_family": candidate["family"], "params": json.dumps(candidate_search.best_params_, sort_keys=True)} for index in range(1, 5)]
        v3_calibration_selection = [{"calibration_fold": index, "group": v3["group"], "model_family": v3["family"], "params": json.dumps(v3_search.best_params_, sort_keys=True)} for index in range(1, 5)]
        candidate_calibrator = fit_sigmoid(candidate_score, candidate_y_local)
        v3_calibrator = fit_sigmoid(v3_score, candidate_y_local)
        candidate_inner_p = candidate_calibrator.predict_proba(candidate_score.reshape(-1, 1))[:, 1]
        v3_inner_p = v3_calibrator.predict_proba(v3_score.reshape(-1, 1))[:, 1]
        candidate_threshold = choose_threshold(candidate_y_local, candidate_inner_p)
        v3_threshold = choose_threshold(candidate_y_local, v3_inner_p)
        fitted_candidate = candidate_search.best_estimator_
        fitted_v3 = v3_search.best_estimator_
        for model_id, spec, estimator, calibrator, threshold, inner_prob, inner_raw, selection_rows in [
            ("selection_aware_procedure", candidate, fitted_candidate, candidate_calibrator, candidate_threshold, candidate_inner_p, candidate_score, calibration_selection),
            ("frozen_v3_logistic_balanced", v3, fitted_v3, v3_calibrator, v3_threshold, v3_inner_p, v3_score, v3_calibration_selection),
        ]:
            x_test = frame.iloc[test_rows][spec["features"]]
            raw_test = raw_probability(estimator, x_test)
            probability = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
            chosen_group = spec["group"]
            chosen_family = spec["family"]
            selections.append({"repeat": repeat, "fold": fold, "model_id": model_id, "selected_group": chosen_group, "selected_model_family": chosen_family, "selected_params": json.dumps((candidate_search if model_id == "selection_aware_procedure" else v3_search).best_params_, sort_keys=True), "inner_best_roc_auc": float((candidate_search if model_id == "selection_aware_procedure" else v3_search).best_score_), "threshold_selected_inner": threshold, "calibration_intercept_training": float(calibrator.intercept_[0]), "calibration_slope_training": float(calibrator.coef_[0, 0]), "calibration_crossfit_rows": len(inner_raw), "calibration_selection_by_fold": json.dumps(selection_rows, sort_keys=True)})
            frequencies.append({"repeat": repeat, "fold": fold, "model_id": model_id, "selected_group": chosen_group, "selected_model_family": chosen_family, "selected_parameters": json.dumps((candidate_search if model_id == "selection_aware_procedure" else v3_search).best_params_, sort_keys=True)})
            for row_index, truth, raw, prob in zip(test_rows, y[test_rows], raw_test, probability):
                predictions.append({"repeat": repeat, "fold": fold, "row_index": int(row_index), "source_row_id": f"source_row:{row_index}", "y_true": int(truth), "model_id": model_id, "selected_group": chosen_group, "selected_model_family": chosen_family, "raw_probability": float(raw), "calibrated_probability": float(prob), "threshold_selected_inner": float(threshold), "prediction_at_inner_threshold": int(prob >= threshold), "split_seed": split["seed"]})
            calibration_rows.append({"repeat": repeat, "fold": fold, "model_id": model_id, "n_outer_test": len(test_rows), "events_outer_test": int(y[test_rows].sum()), "prevalence_outer_test": float(y[test_rows].mean()), "calibration_intercept_outer_test": float(metric_record(y[test_rows], probability, threshold, repeat=repeat, fold=fold, model_id=model_id, selected_group=chosen_group, selected_family=chosen_family)["calibration_intercept"]), "calibration_slope_outer_test": float(metric_record(y[test_rows], probability, threshold, repeat=repeat, fold=fold, model_id=model_id, selected_group=chosen_group, selected_family=chosen_family)["calibration_slope"]), "calibration_intercept_training": float(calibrator.intercept_[0]), "calibration_slope_training": float(calibrator.coef_[0, 0])})
        # Outer-test permutation importance is descriptive only and cannot
        # affect model selection. It measures marginal dependence of the frozen
        # selected outer model on each raw source feature.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(fitted_candidate, frame.iloc[test_rows][candidate["features"]], y[test_rows], scoring="roc_auc", n_repeats=5, random_state=SEED + repeat * 100 + fold, n_jobs=1)
        for feature, mean, std in zip(candidate["features"], perm.importances_mean, perm.importances_std):
            permutation_rows.append({"repeat": repeat, "fold": fold, "selected_group": candidate["group"], "selected_model_family": candidate["family"], "feature": feature, "permutation_auc_decrease_mean": float(mean), "permutation_auc_decrease_sd": float(std)})
        fold_state = {
            "repeat": repeat, "fold": fold,
            "predictions": [row for row in predictions if row["repeat"] == repeat and row["fold"] == fold],
            "selections": [row for row in selections if row["repeat"] == repeat and row["fold"] == fold],
            "frequencies": [row for row in frequencies if row["repeat"] == repeat and row["fold"] == fold],
            "calibration_rows": [row for row in calibration_rows if row["repeat"] == repeat and row["fold"] == fold],
            "inner_oof_records": [row for row in inner_oof_records if row["repeat"] == repeat and row["fold"] == fold],
            "permutation_rows": [row for row in permutation_rows if row["repeat"] == repeat and row["fold"] == fold],
        }
        (work_dir / f"outer_r{repeat}_f{fold}.pkl").write_bytes(pickle.dumps(fold_state, protocol=pickle.HIGHEST_PROTOCOL))
    pred_frame = pd.DataFrame(predictions)
    selection_frame = pd.DataFrame(selections)
    expected_prediction_rows = 303 * len({split["repeat"] for split in outer}) * 2
    if len(pred_frame) != expected_prediction_rows or pred_frame.duplicated(["repeat", "fold", "row_index", "model_id"]).any():
        raise RuntimeError("Outer prediction coverage or uniqueness check failed")
    folds, repeats, overall = summarize_predictions(pred_frame, selection_frame)
    comparison_rows = []
    for repeat, group in repeats.groupby("repeat"):
        left = group.set_index("model_id").loc["frozen_v3_logistic_balanced"]
        right = group.set_index("model_id").loc["selection_aware_procedure"]
        comparison_rows.append({"repeat": int(repeat), **{f"{metric}_difference_selection_minus_v3": float(right[metric] - left[metric]) for metric in ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity"]}})
    comparison = pd.DataFrame(comparison_rows)
    difference_summary = {"repeat_count": len(comparison)}
    for column in comparison.columns.drop("repeat"):
        values = comparison[column].to_numpy(dtype=float)
        half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan
        difference_summary[f"{column}_mean"] = float(values.mean())
        difference_summary[f"{column}_sd"] = float(values.std(ddof=1))
        difference_summary[f"{column}_min"] = float(values.min())
        difference_summary[f"{column}_max"] = float(values.max())
        difference_summary[f"{column}_split_variation_95ci_low"] = float(values.mean() - half)
        difference_summary[f"{column}_split_variation_95ci_high"] = float(values.mean() + half)
    overall = overall.merge(pd.DataFrame([{"model_id": "paired_difference_selection_minus_v3", **difference_summary}]), on="model_id", how="outer")
    group_frequency = selection_frame[selection_frame.model_id == "selection_aware_procedure"].groupby(["selected_group", "selected_model_family"]).size().rename("selection_count").reset_index()
    group_frequency["outer_fold_count"] = len(outer)
    group_frequency["selection_frequency"] = group_frequency.selection_count / len(outer)
    feature_frequency = []
    for feature in sorted(expected):
        groups_with_feature = [name for name, features in GROUPS.items() if feature in features]
        selected_groups = selection_frame[(selection_frame.model_id == "selection_aware_procedure") & selection_frame.selected_group.isin(groups_with_feature)]
        feature_frequency.append({"feature": feature, "category": "timing_uncertain_retrospective", "selected_group_inclusion_count": len(selected_groups), "outer_fold_count": len(outer), "selection_group_inclusion_frequency": len(selected_groups) / len(outer)})
    pd.DataFrame(feature_frequency).to_csv(OUT / "FEATURE_SELECTION_FREQUENCY.csv", index=False)
    # Demographic-only analysis is a prespecified sensitivity, not a selected
    # candidate and not labelled confirmed-pre-diagnostic.
    sensitivity_rows = []
    demographic = ["Age", "Sex"]
    for split in outer:
        train_rows, test_rows = split["train"], split["test"]
        cv = inner_map[(split["repeat"], split["fold"])]
        x_train = frame.iloc[train_rows].reset_index(drop=True)
        y_train = y[train_rows]
        spec = {"group": "demographic_only_sensitivity", "family": "logistic_l2", "features": demographic, "pipeline": make_pipeline(frame, demographic, "logistic"), "grid": V3_GRID}
        search = search_candidate(spec, x_train[demographic], y_train, cv)
        raw_oof = np.full(len(train_rows), np.nan)
        for _, (fit_pos, valid_pos) in enumerate(cv):
            est = make_pipeline(frame, demographic, "logistic").set_params(**search.best_params_)
            est.fit(x_train.iloc[fit_pos][demographic], y_train[fit_pos])
            raw_oof[valid_pos] = raw_probability(est, x_train.iloc[valid_pos][demographic])
        calibrator = fit_sigmoid(raw_oof, y_train)
        p_inner = calibrator.predict_proba(raw_oof.reshape(-1, 1))[:, 1]
        threshold = choose_threshold(y_train, p_inner)
        p_test = calibrator.predict_proba(raw_probability(search.best_estimator_, frame.iloc[test_rows][demographic]).reshape(-1, 1))[:, 1]
        sensitivity_rows.append(metric_record(y[test_rows], p_test, threshold, repeat=split["repeat"], fold=split["fold"], model_id="demographic_only_sensitivity", selected_group=spec["group"], selected_family=spec["family"]))
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "DEMOGRAPHIC_ONLY_SENSITIVITY.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(OUT / "OUTER_FOLD_PREDICTIONS.csv", index=False)
    group_frequency.to_csv(OUT / "MODEL_SELECTION_FREQUENCY.csv", index=False)
    folds.to_csv(OUT / "PERFORMANCE_COMPARISON.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "CALIBRATION_RESULTS.csv", index=False)
    repeats.to_csv(OUT / "REPEAT_LEVEL_PERFORMANCE.csv", index=False)
    overall.to_csv(OUT / "OVERALL_PERFORMANCE_SUMMARY.csv", index=False)
    comparison.to_csv(OUT / "PAIRED_REPEAT_DIFFERENCES.csv", index=False)
    pd.DataFrame(inner_oof_records).to_csv(OUT / "INNER_CANDIDATE_SELECTION_SCORES.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(OUT / "FEATURE_PERMUTATION_IMPORTANCE.csv", index=False)
    selection_frame.to_csv(OUT / "OUTER_MODEL_SELECTIONS.csv", index=False)
    shutil.rmtree(work_dir)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    run()
