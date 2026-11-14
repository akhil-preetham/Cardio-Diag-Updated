"""Retrospective ZAS extension; immutable checkpoints, selection-independent calibration.

No historical output is changed. Run with project .venv, PYTHONPATH=src;.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from runner.phase6_zas_selection import run as legacy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_extension_v1/zas"
PROTOCOL = OUT / "PROTOCOL.md"
SEED = 861051
MODEL_IDS = (
    "l2_full38_historical_calibration",
    "l2_no_symptoms28_historical_calibration",
    "l2_risk_symptoms17_historical_calibration",
    "l2_v3_ordinary_historical_calibration",
    "selection_independent_calibrated_ensemble",
    "same_components_uncalibrated_ensemble",
)
METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "sensitivity", "specificity", "calibration_intercept", "calibration_slope")
INPUTS = {
    "dataset": legacy.DATA_PATH,
    "outer_split_source": legacy.SPLIT_PATH,
    "historical_runner": Path(legacy.__file__),
    "historical_predictions": legacy.OUT / "OUTER_FOLD_PREDICTIONS.csv",
    "historical_manifest": legacy.OUT / "PHASE6_MANIFEST.json",
    "extension_runner": Path(__file__),
    "protocol": PROTOCOL,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_immutable(path: Path, value: object) -> None:
    """Publish a JSON/text artifact once; require equality on repeat execution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(data)


def input_hashes() -> dict[str, str]:
    return {key: sha256(path) for key, path in INPUTS.items()}


def freeze() -> dict:
    hashes = input_hashes()
    freeze_path = OUT / "FROZEN_INPUTS.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen["hashes"] != hashes:
            raise RuntimeError("Frozen source/protocol/input changed; use a new versioned namespace")
        return frozen
    frozen = {
        "created_utc_before_modeling": utc_now(),
        "hashes": hashes,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
        "python": sys.version,
        "platform": platform.platform(),
        "threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "script_sha256": hashes["extension_runner"],
    }
    write_immutable(freeze_path, frozen)
    return frozen


def feature_sets() -> dict[str, list[str]]:
    full = legacy.GROUPS["eligible_combined"]
    symptoms = set(legacy.GROUPS["symptoms_examination"])
    return {
        MODEL_IDS[0]: list(full),
        MODEL_IDS[1]: [col for col in full if col not in symptoms],
        MODEL_IDS[2]: legacy.GROUPS["basic_risk"] + legacy.GROUPS["symptoms_examination"],
        MODEL_IDS[3]: list(legacy.V3_FEATURES),
    }


def select_configuration(x_development: pd.DataFrame, y_development: np.ndarray, inner_cv: list, *, features: list[str] | None = None) -> tuple[dict, object, list[dict], list[dict]]:
    """Only development features/labels enter candidate/hyperparameter selection."""
    specs = legacy.candidate_specs(x_development) if features is None else [{
        "group": "fixed_information_set", "family": "logistic_l2", "features": features,
        "pipeline": legacy.make_pipeline(x_development, features, "logistic"),
        "grid": legacy.MODEL_GRIDS["logistic_l2"][1],
    }]
    best_score = -np.inf
    best_spec, best_estimator = None, None
    candidate_rows, grid_rows = [], []
    for spec in specs:
        search = GridSearchCV(spec["pipeline"], spec["grid"], scoring="roc_auc", cv=inner_cv, n_jobs=1, refit=True, error_score="raise", return_train_score=False)
        search.fit(x_development[spec["features"]], y_development)
        candidate_rows.append({"group": spec["group"], "family": spec["family"], "best_score": float(search.best_score_), "best_params": search.best_params_})
        for index, params in enumerate(search.cv_results_["params"]):
            grid_rows.append({"group": spec["group"], "family": spec["family"], "params": params,
                              "mean_inner_auc": float(search.cv_results_["mean_test_score"][index]),
                              "inner_fold_aucs": [float(search.cv_results_[f"split{k}_test_score"][index]) for k in range(len(inner_cv))]})
        if search.best_score_ > best_score + 1e-12:
            best_score, best_spec, best_estimator = float(search.best_score_), spec, search.best_estimator_
            best_params = search.best_params_
    if best_spec is None or best_estimator is None:
        raise RuntimeError("No selected candidate")
    selection = {"group": best_spec["group"], "family": best_spec["family"], "features": best_spec["features"], "params": best_params, "inner_auc": best_score}
    return selection, best_estimator, candidate_rows, grid_rows


def sigmoid(scores: np.ndarray, y_calibration: np.ndarray) -> LogisticRegression:
    # Same convention/regularization as Phase 6, now held independent of selection.
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(np.asarray(scores).reshape(-1, 1), y_calibration)


def train_corrected_component(x_training: pd.DataFrame, y_training: np.ndarray, x_outer_unlabeled: pd.DataFrame, development_pos: np.ndarray, calibration_pos: np.ndarray, inner_cv: list) -> dict:
    """No outer labels accepted; calibration labels never enter base selection."""
    if set(development_pos) & set(calibration_pos) or set(development_pos) | set(calibration_pos) != set(range(len(y_training))):
        raise ValueError("Development/calibration must partition outer training")
    xd = x_training.iloc[development_pos].reset_index(drop=True)
    selection, estimator, candidates, grids = select_configuration(xd, y_training[development_pos], inner_cv)
    calibration_scores = estimator.predict_proba(x_training.iloc[calibration_pos][selection["features"]])[:, 1]
    calibrator = sigmoid(calibration_scores, y_training[calibration_pos])
    raw = estimator.predict_proba(x_outer_unlabeled[selection["features"]])[:, 1]
    calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return {"selection": selection, "candidates": candidates, "grid_scores": grids,
            "calibration_raw_scores": calibration_scores.tolist(), "calibration_y": y_training[calibration_pos].tolist(),
            "calibrator_intercept": float(calibrator.intercept_[0]), "calibrator_coefficient": float(calibrator.coef_[0, 0]),
            "raw_probability": raw.tolist(), "calibrated_probability": calibrated.tolist()}


def train_historical_control(x_training: pd.DataFrame, y_training: np.ndarray, x_outer_unlabeled: pd.DataFrame, features: list[str], inner_cv: list) -> dict:
    selection, estimator, candidates, grids = select_configuration(x_training, y_training, inner_cv, features=features)
    oof = np.full(len(y_training), np.nan)
    for fit_pos, valid_pos in inner_cv:
        component = legacy.make_pipeline(x_training, features, "logistic").set_params(**selection["params"])
        component.fit(x_training.iloc[fit_pos][features], y_training[fit_pos])
        oof[valid_pos] = component.predict_proba(x_training.iloc[valid_pos][features])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete historical crossfit")
    calibrator = sigmoid(oof, y_training)
    threshold = legacy.choose_threshold(y_training, calibrator.predict_proba(oof.reshape(-1, 1))[:, 1])
    raw = estimator.predict_proba(x_outer_unlabeled[features])[:, 1]
    p = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return {"selection": selection, "candidates": candidates, "grid_scores": grids,
            "threshold": threshold, "raw_probability": raw.tolist(), "calibrated_probability": p.tolist(),
            "calibrator_intercept": float(calibrator.intercept_[0]), "calibrator_coefficient": float(calibrator.coef_[0, 0]),
            "inner_oof_scores": oof.tolist()}


def split_trace(split: dict, cv: list, y_training: np.ndarray) -> tuple[list[dict], list[dict]]:
    """All positions converted to stable source-row IDs for audit."""
    training_rows = np.asarray(split["train"], dtype=int)
    historical = [{"fold": k + 1, "fit_rows": training_rows[a].tolist(), "valid_rows": training_rows[b].tolist()} for k, (a, b) in enumerate(cv)]
    components = []
    for k, (dev, cal) in enumerate(cv, 1):
        inner_seed = SEED + 1000 * split["repeat"] + 100 * split["fold"] + k
        inside = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed).split(np.arange(len(dev)), y_training[dev]))
        components.append({"component": k, "inner_seed": inner_seed, "development_pos": dev.tolist(), "calibration_pos": cal.tolist(),
                           "development_rows": training_rows[dev].tolist(), "calibration_rows": training_rows[cal].tolist(),
                           "inner_cv": [{"fold": j + 1, "fit_rows": training_rows[dev[a]].tolist(), "valid_rows": training_rows[dev[b]].tolist()} for j, (a, b) in enumerate(inside)]})
    return historical, components


def warning_summary(caught: list) -> list[dict]:
    counts = Counter((type(item.message).__name__, str(item.message)) for item in caught)
    return [{"category": key[0], "message": key[1], "count": count} for key, count in sorted(counts.items())]


def execute_checkpoint(path: Path, hashes: dict, compute, trace: dict) -> dict:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["input_hashes"] != hashes or value["trace"] != trace:
            raise RuntimeError(f"Incompatible checkpoint: {path}")
        return value
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = compute()
    state.update({"input_hashes": hashes, "trace": trace, "completed_utc": utc_now(), "wall_seconds": time.perf_counter() - started, "warnings": warning_summary(caught)})
    write_immutable(path, state)
    print(json.dumps({"checkpoint": path.name, "wall_seconds": round(state["wall_seconds"], 2), "warnings": sum(item["count"] for item in state["warnings"])}), flush=True)
    return state


def evaluate(y: np.ndarray, p: np.ndarray, threshold: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    classified = p >= np.asarray(threshold)
    # Joint intercept/slope regression; NOT slope-fixed calibration-in-the-large.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
    return {"n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "sensitivity": float(classified[y == 1].mean()), "specificity": float((~classified[y == 0]).mean()),
            "calibration_intercept": float(diagnostic.intercept_[0]), "calibration_slope": float(diagnostic.coef_[0, 0])}


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds, repeats = [], []
    for columns, target in ((["repeat", "fold", "model_id"], folds), (["repeat", "model_id"], repeats)):
        for keys, group in predictions.groupby(columns, sort=True):
            record = dict(zip(columns, keys))
            record.update(evaluate(group.y_true.to_numpy(dtype=int), group.probability.to_numpy(dtype=float), group.threshold.to_numpy(dtype=float)))
            target.append(record)
    repeat_frame = pd.DataFrame(repeats)
    summaries = []
    for model, group in repeat_frame.groupby("model_id", sort=True):
        row = {"model_id": model, "repeats": len(group)}
        for metric in METRICS:
            row.update({f"{metric}_mean": float(group[metric].mean()), f"{metric}_sd": float(group[metric].std(ddof=1)),
                        f"{metric}_min": float(group[metric].min()), f"{metric}_max": float(group[metric].max())})
        summaries.append(row)
    return pd.DataFrame(folds), repeat_frame, pd.DataFrame(summaries)


def paired_comparisons(repeats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = [
        ("historical_selection", MODEL_IDS[0], "historical_selection_minus_equal38"),
        (MODEL_IDS[4], MODEL_IDS[5], "corrected_calibrated_minus_identical_raw_ensemble"),
        (MODEL_IDS[0], MODEL_IDS[1], "full38_minus_no_symptoms28"),
        (MODEL_IDS[0], MODEL_IDS[2], "full38_minus_risk_symptoms17"),
        (MODEL_IDS[0], MODEL_IDS[3], "full38_minus_ordinary_v3"),
        (MODEL_IDS[4], "historical_selection", "corrected_ensemble_minus_historical_selection_NOT_calibration_only"),
    ]
    records, summary = [], []
    for left, right, name in contrasts:
        l = repeats[repeats.model_id == left].set_index("repeat")
        r = repeats[repeats.model_id == right].set_index("repeat")
        delta = l[list(METRICS)] - r[list(METRICS)]
        for repeat, row in delta.iterrows():
            records.append({"contrast": name, "repeat": int(repeat), **{m: float(row[m]) for m in METRICS}})
        for m in METRICS:
            summary.append({"contrast": name, "metric": m, "mean_difference": float(delta[m].mean()), "sd_split_variation": float(delta[m].std(ddof=1)),
                            "minimum_difference": float(delta[m].min()), "maximum_difference": float(delta[m].max()), "repeat_count": len(delta),
                            "uncertainty": "descriptive overlapping source-row repeats; no population CI or significance test"})
    return pd.DataFrame(records), pd.DataFrame(summary)


def write_csv_immutable(path: Path, frame: pd.DataFrame) -> None:
    write_immutable(path, frame.to_csv(index=False, lineterminator="\n"))


def finalize(prediction_rows: list, selections: list, candidates: list, grids: list, traces: list, checkpoints: list, frozen: dict) -> None:
    pred = pd.DataFrame(prediction_rows)
    historical = pd.read_csv(INPUTS["historical_predictions"])
    historical["model_id"] = historical.model_id.map({"selection_aware_procedure": "historical_selection", "frozen_v3_logistic_balanced": "historical_v3_balanced"})
    historical = historical.rename(columns={"calibrated_probability": "probability", "threshold_selected_inner": "threshold"})
    common = ["repeat", "fold", "row_index", "y_true", "model_id", "probability", "raw_probability", "threshold"]
    pred = pd.concat([pred[common], historical[common]], ignore_index=True).sort_values(["repeat", "fold", "model_id", "row_index"]).reset_index(drop=True)
    if len(pred) != 303 * 3 * 8 or pred.duplicated(["repeat", "model_id", "row_index"]).any():
        raise RuntimeError("Prediction coverage or uniqueness violation")
    folds, repeats, summary = summarize(pred)
    differences, difference_summary = paired_comparisons(repeats)
    folds["ap"] = folds.average_precision
    repeats["ap"] = repeats.average_precision
    for name, frame in (("PREDICTIONS.csv", pred), ("OOF_PREDICTIONS.csv", pred), ("FOLD_METRICS.csv", folds), ("REPEAT_METRICS.csv", repeats), ("SUMMARY_METRICS.csv", summary), ("SUMMARY.csv", summary), ("PAIRED_REPEAT_DIFFERENCES.csv", differences), ("PAIRED_DESCRIPTIVE_SUMMARY.csv", difference_summary), ("SELECTIONS.csv", pd.DataFrame(selections)), ("CANDIDATE_SCORES.csv", pd.DataFrame(candidates)), ("GRID_SCORES.csv", pd.DataFrame(grids))):
        write_csv_immutable(OUT / name, frame)
    write_immutable(OUT / "SPLIT_TRACE.json", traces)
    warning_rows = [{"checkpoint": cp["checkpoint_name"], **item} for cp in checkpoints for item in cp["warnings"]]
    write_immutable(OUT / "WARNINGS.json", warning_rows)
    runtimes = [{"checkpoint": cp["checkpoint_name"], "wall_seconds": cp["wall_seconds"], "completed_utc": cp["completed_utc"]} for cp in checkpoints]
    write_immutable(OUT / "RUNTIMES.json", runtimes)
    if input_hashes() != frozen["hashes"]:
        raise RuntimeError("Input changed during computation")
    outputs = {str(path.relative_to(ROOT)): sha256(path) for path in OUT.rglob("*") if path.is_file() and path.name != "MANIFEST.json" and not path.name.endswith(".log")}
    manifest = {"completed_utc": utc_now(), "input_hashes": frozen["hashes"], "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()}, "versions": frozen["versions"], "outputs": outputs,
                "new_models": list(MODEL_IDS), "retrospectively_motivated": True, "outer_partitions": 15, "source_rows": 303, "new_prediction_rows": 303 * 3 * 6,
                "full_prediction_rows_with_historical": len(pred), "corrected_component_searches": 45, "corrected_candidate_searches": 45 * 30,
                "checkpoint_seconds_sum": sum(cp["wall_seconds"] for cp in checkpoints), "historical_outputs_modified": False}
    if (OUT / "MANIFEST.json").exists():
        prior = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
        if prior["input_hashes"] != manifest["input_hashes"] or prior["outputs"] != manifest["outputs"]:
            raise RuntimeError("Existing final manifest differs")
    else:
        write_immutable(OUT / "MANIFEST.json", manifest)
    print(summary[["model_id", "roc_auc_mean", "brier_mean", "log_loss_mean"]].to_string(index=False), flush=True)


def run(pilot_only: bool = False) -> None:
    frozen = freeze()
    frame = pd.read_csv(legacy.DATA_PATH)
    y = frame.Cath.map({"Cad": 1, "Normal": 0}).to_numpy(dtype=int)
    outer, inner_map = legacy.load_splits(legacy.SPLIT_PATH, y)
    predictions, selections, candidates, grids, traces, checkpoints = [], [], [], [], [], []
    for split_index, split in enumerate(outer):
        repeat, fold = split["repeat"], split["fold"]
        cv = inner_map[repeat, fold]
        train, test = np.asarray(split["train"]), np.asarray(split["test"])
        xt = frame.iloc[train].drop(columns=["Cath"]).reset_index(drop=True)
        # Training APIs receive no outer-test targets or target column.
        xo = frame.iloc[test].drop(columns=["Cath"]).reset_index(drop=True)
        yt = y[train]
        historical_trace, components = split_trace(split, cv, yt)
        trace = {"repeat": repeat, "fold": fold, "outer_train": train.tolist(), "outer_test": test.tolist(), "historical_inner": historical_trace, "components": components}
        traces.append(trace)
        for model, features in feature_sets().items():
            cp_trace = {"repeat": repeat, "fold": fold, "model": model, "outer_train": train.tolist(), "outer_test": test.tolist(), "historical_inner": historical_trace}
            path = OUT / "checkpoints" / f"r{repeat}_f{fold}_{model}.json"
            cp = execute_checkpoint(path, frozen["hashes"], partial(train_historical_control, xt, yt, xo, features, cv), cp_trace)
            cp["checkpoint_name"] = path.name
            checkpoints.append(cp)
            meta = {"repeat": repeat, "fold": fold, "model_id": model, "component": 0}
            selections.append({**meta, **cp["selection"], "threshold": cp["threshold"]})
            candidates.extend([{**meta, **row} for row in cp["candidates"]])
            grids.extend([{**meta, **row} for row in cp["grid_scores"]])
            for i, row in enumerate(test):
                predictions.append({**meta, "row_index": int(row), "y_true": int(y[row]), "probability": cp["calibrated_probability"][i], "raw_probability": cp["raw_probability"][i], "threshold": cp["threshold"]})
        component_states = []
        for component in components:
            k = component["component"]
            dev, cal = np.asarray(component["development_pos"]), np.asarray(component["calibration_pos"])
            inside = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=component["inner_seed"]).split(np.arange(len(dev)), yt[dev]))
            cp_trace = {"repeat": repeat, "fold": fold, "outer_test": test.tolist(), **component}
            path = OUT / "checkpoints" / f"r{repeat}_f{fold}_corrected_component{k}.json"
            cp = execute_checkpoint(path, frozen["hashes"], partial(train_corrected_component, xt, yt, xo, dev, cal, inside), cp_trace)
            cp["checkpoint_name"] = path.name
            checkpoints.append(cp)
            component_states.append(cp)
            meta = {"repeat": repeat, "fold": fold, "model_id": MODEL_IDS[4], "component": k}
            selections.append({**meta, **cp["selection"], "threshold": 0.5})
            candidates.extend([{**meta, **row} for row in cp["candidates"]])
            grids.extend([{**meta, **row} for row in cp["grid_scores"]])
        for model, value_key in ((MODEL_IDS[4], "calibrated_probability"), (MODEL_IDS[5], "raw_probability")):
            p = np.mean([cp[value_key] for cp in component_states], axis=0)
            raw = np.mean([cp["raw_probability"] for cp in component_states], axis=0)
            for i, row in enumerate(test):
                predictions.append({"repeat": repeat, "fold": fold, "model_id": model, "row_index": int(row), "y_true": int(y[row]), "probability": float(p[i]), "raw_probability": float(raw[i]), "threshold": 0.5})
        print(f"COMPLETE OUTER {split_index + 1}/15 repeat={repeat} fold={fold}", flush=True)
        if split_index == 0:
            pilot = {"first_outer_partition": "repeat1/fold1", "checkpoint_seconds": sum(cp["wall_seconds"] for cp in checkpoints),
                     "projected_15_partition_seconds_linear": sum(cp["wall_seconds"] for cp in checkpoints) * 15,
                     "limits": "Projection only; same candidate pool regardless of pilot scores; 1 BLAS/OMP thread, n_jobs=1", "versions": frozen["versions"]}
            write_immutable(OUT / "PILOT_RUNTIME.json", pilot)
            if pilot_only:
                print("Pilot complete; resume without --pilot-only for full locked design.", flush=True)
                return
    finalize(predictions, selections, candidates, grids, traces, checkpoints, frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(pilot_only=args.pilot_only)
