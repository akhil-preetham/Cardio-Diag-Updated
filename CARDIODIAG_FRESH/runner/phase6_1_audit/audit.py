"""Read-only Phase 6.1 reconciliation; no fitting or candidate search."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from runner.phase6_zas_selection.run import FAMILY_ORDER, GROUPS

ROOT = Path(__file__).resolve().parents[2]
PHASE6 = ROOT / "results/phase6_zas_selection"
DATA = ROOT / "data/interim/validated/z_alizadeh_sani.csv"
SPLITS = ROOT / "results/methodology_corrections_v2/reproducibility/split_manifest_v2.json"
PHASE6_RUNNER = ROOT / "runner/phase6_zas_selection/run.py"
PHASE6_PROTOCOL = PHASE6 / "SELECTION_AWARE_PROTOCOL.md"
OUT = ROOT / "results/phase6_1_audit"
SEED = 61051
MODELS = {"selection_aware_procedure", "frozen_v3_logistic_balanced"}
METRICS = ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity", "calibration_intercept", "calibration_slope"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit(clipped).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def reconstruct_metrics(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray | float) -> dict[str, float]:
    """Independent metric reconstruction from saved labels and probabilities."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=float), y.shape)
    pred = p >= thresholds
    intercept, slope = _calibration(y, p)
    return {
        "n": len(y),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "pr_auc_prevalence_reference": float(y.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "sensitivity": float(pred[y == 1].mean()),
        "specificity": float((~pred[y == 0]).mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance))


def verify_phase6_manifest() -> dict:
    manifest_path = PHASE6 / "PHASE6_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_results = []
    for rel, expected in manifest["outputs"].items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else None
        output_results.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected})
    input_results = []
    for rel, expected in manifest["inputs"].items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else None
        input_results.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected})
    return {"outputs": output_results, "inputs": input_results, "all_match": all(x["matches"] for x in output_results + input_results)}


def verify_audit_manifest() -> dict:
    manifest_path = OUT / "PHASE6_1_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for section in ["inputs", "outputs"]:
        for rel, expected in manifest[section].items():
            path = ROOT / rel
            actual = sha256(path) if path.is_file() else None
            results.append({"section": section, "path": rel, "matches": actual == expected})
    return {"all_match": all(row["matches"] for row in results), "count": len(results), "checks": results}


def reconcile() -> dict:
    data = pd.read_csv(DATA)
    y_all = data["Cath"].map({"Cad": 1, "Normal": 0})
    if len(data) != 303 or y_all.isna().any():
        raise ValueError("ZAS source data shape/target mapping differs from the Phase 6 contract")
    y_all = y_all.to_numpy(dtype=int)
    split_manifest = json.loads(SPLITS.read_text(encoding="utf-8"))
    groups: dict[tuple[int, int], dict[str, set[int]]] = {}
    for row in split_manifest["outer_assignments"]:
        groups.setdefault((int(row["repeat"]), int(row["fold"])), {"train": set(), "test": set()})[row["role"]].add(int(row["row_index"]))
    if len(groups) != 25:
        raise ValueError("Frozen split manifest does not contain 25 outer partitions")
    selected_splits = {key: value for key, value in groups.items() if key[0] <= 3}

    predictions = pd.read_csv(PHASE6 / "OUTER_FOLD_PREDICTIONS.csv")
    selections = pd.read_csv(PHASE6 / "OUTER_MODEL_SELECTIONS.csv")
    benchmark_fold = pd.read_csv(PHASE6 / "PERFORMANCE_COMPARISON.csv")
    benchmark_repeat = pd.read_csv(PHASE6 / "REPEAT_LEVEL_PERFORMANCE.csv")
    benchmark_paired = pd.read_csv(PHASE6 / "PAIRED_REPEAT_DIFFERENCES.csv")
    calibration = pd.read_csv(PHASE6 / "CALIBRATION_RESULTS.csv")
    inner_scores = pd.read_csv(PHASE6 / "INNER_CANDIDATE_SELECTION_SCORES.csv")

    expected_keys = {(repeat, fold, model) for repeat, fold in selected_splits for model in MODELS}
    selection_keys = set(zip(selections.repeat.astype(int), selections.fold.astype(int), selections.model_id))
    if selection_keys != expected_keys or selections.duplicated(["repeat", "fold", "model_id"]).any():
        raise ValueError("Selection rows do not uniquely cover the selected 15 outer partitions")
    pred_keys = set(zip(predictions.repeat.astype(int), predictions.fold.astype(int), predictions.model_id))
    if pred_keys != expected_keys or predictions.duplicated(["repeat", "fold", "row_index", "model_id"]).any():
        raise ValueError("Prediction keys do not uniquely cover the selected outer partitions")
    if set(predictions.model_id) != MODELS:
        raise ValueError("Unexpected prediction model identifier")

    fold_rows = []
    coverage_rows = []
    metadata_rows = []
    split_size_rows = []
    candidate_selection_rows = []
    for (repeat, fold), split in sorted(selected_splits.items()):
        train, test = split["train"], split["test"]
        if train & test or train | test != set(range(len(data))):
            raise ValueError(f"Invalid outer split membership at {repeat}/{fold}")
        train_rows, test_rows = sorted(train), sorted(test)
        # Reconstruct Phase 6's deterministic three-fold inner partition design;
        # no estimators are fitted here.
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + repeat * 100 + fold)
        inner_sizes = []
        for fit_pos, valid_pos in inner.split(np.arange(len(train_rows)), y_all[train_rows]):
            inner_sizes.append({"fit_n": len(fit_pos), "calibration_valid_n": len(valid_pos)})
        split_size_rows.append({"repeat": repeat, "fold": fold, "outer_train_n": len(train_rows), "outer_test_n": len(test_rows), "inner_fit_sizes": ";".join(str(x["fit_n"]) for x in inner_sizes), "inner_validation_sizes": ";".join(str(x["calibration_valid_n"]) for x in inner_sizes), "final_model_training_n": len(train_rows)})
        for model in sorted(MODELS):
            part = predictions[(predictions.repeat == repeat) & (predictions.fold == fold) & (predictions.model_id == model)].copy()
            got_rows = set(part.row_index.astype(int))
            if got_rows != test:
                raise ValueError(f"Outer test source rows disagree with split manifest at {repeat}/{fold}/{model}")
            if part.row_index.nunique() != len(test) or len(part) != len(test):
                raise ValueError(f"Wrong prediction count at {repeat}/{fold}/{model}")
            part["row_index"] = part.row_index.astype(int)
            part = part.sort_values("row_index")
            actual_y = y_all[part.row_index.to_numpy()]
            label_match = np.array_equal(part.y_true.to_numpy(dtype=int), actual_y)
            source_ids_ok = all(str(source_id) == f"source_row:{row}" for source_id, row in zip(part.source_row_id, part.row_index))
            if not label_match or not source_ids_ok:
                raise ValueError(f"Source labels or source-row labels mismatch at {repeat}/{fold}/{model}")
            sel = selections[(selections.repeat == repeat) & (selections.fold == fold) & (selections.model_id == model)].iloc[0]
            group_name = str(sel.selected_group)
            family = str(sel.selected_model_family)
            if not (part.selected_group == group_name).all() or not (part.selected_model_family == family).all():
                raise ValueError(f"Per-fold selection labels differ from prediction metadata at {repeat}/{fold}/{model}")
            if model == "frozen_v3_logistic_balanced" and (group_name != "v3_five_predictor" or family != "v3_logistic_balanced"):
                raise ValueError("V3 comparator metadata does not match its declared reference")
            if model == "selection_aware_procedure":
                scores = inner_scores[(inner_scores.repeat == repeat) & (inner_scores.fold == fold)].copy()
                if len(scores) != len(GROUPS) * len(FAMILY_ORDER) or scores.duplicated(["group", "model_family"]).any():
                    raise ValueError(f"Candidate-score table is incomplete/duplicated at {repeat}/{fold}")
                group_order = {name: idx for idx, name in enumerate(GROUPS)}
                family_order = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
                scores["_group_order"] = scores.group.map(group_order)
                scores["_family_order"] = scores.model_family.map(family_order)
                scores = scores.sort_values(["_group_order", "_family_order"], kind="stable")
                chosen_score = -np.inf
                expected_winner = None
                for score_row in scores.itertuples(index=False):
                    if float(score_row.inner_roc_auc) > chosen_score + 1e-12:
                        chosen_score = float(score_row.inner_roc_auc)
                        expected_winner = (str(score_row.group), str(score_row.model_family), str(score_row.best_params))
                if expected_winner is None or expected_winner[:2] != (group_name, family):
                    raise ValueError(f"Selected model/group is not the winner in saved inner scores at {repeat}/{fold}")
                selected_params = json.loads(sel.selected_params)
                if selected_params != json.loads(expected_winner[2]) or not _close(chosen_score, sel.inner_best_roc_auc):
                    raise ValueError(f"Selected params/score disagree with saved inner candidate ranking at {repeat}/{fold}")
                candidate_selection_rows.append({"repeat": repeat, "fold": fold, "candidate_count": len(scores), "winner_group": group_name, "winner_family": family, "winner_inner_roc_auc": chosen_score, "winner_params": sel.selected_params, "selection_metadata_matches_saved_inner_ranking": True})
            rec = reconstruct_metrics(actual_y, part.calibrated_probability.to_numpy(dtype=float), float(sel.threshold_selected_inner))
            benchmark_row = benchmark_fold[(benchmark_fold.repeat == repeat) & (benchmark_fold.fold == fold) & (benchmark_fold.model_id == model)]
            if len(benchmark_row) != 1:
                raise ValueError(f"Missing/duplicate benchmark fold metric row at {repeat}/{fold}/{model}")
            bench = benchmark_row.iloc[0]
            comparisons = {metric: _close(rec[metric], bench[metric]) for metric in METRICS}
            if not all(comparisons.values()):
                raise ValueError(f"Fold metrics do not reconcile at {repeat}/{fold}/{model}: {comparisons}")
            fold_rows.append({"repeat": repeat, "fold": fold, "model_id": model, "selected_group": group_name, "selected_model_family": family, **rec, "benchmark_metrics_match": all(comparisons.values())})
            coverage_rows.append({"repeat": repeat, "fold": fold, "model_id": model, "expected_test_rows": len(test), "observed_prediction_rows": len(part), "unique_source_rows": int(part.row_index.nunique()), "target_labels_match_source": label_match, "source_row_label_consistent": source_ids_ok, "manifest_partition_match": True})

            # Corrected display metadata: Phase 6 actually cross-fits over 3
            # inner folds with one already-selected candidate/parameter set.
            for inner_fold in range(1, len(inner_sizes) + 1):
                metadata_rows.append({"repeat": repeat, "outer_fold": fold, "model_id": model, "calibration_fold": inner_fold, "selected_group": group_name, "selected_model_family": family, "selected_params": sel.selected_params, "selection_scope": "once across all 3 inner folds before cross-fit scoring", "fold_specific_reselection": False, "metadata_correction": "Phase 6 saved four duplicate records (1-4); corrected display has 3 records (1-3), matching actual crossfit_selected_scores loop."})

    folds = pd.DataFrame(fold_rows)
    repeats = []
    for (repeat, model), part in predictions.groupby(["repeat", "model_id"], sort=True):
        part = part.sort_values("row_index")
        if len(part) != 303 or part.row_index.nunique() != 303:
            raise ValueError(f"Repeat {repeat}/{model} does not have exactly 303 unique source-row predictions")
        row_threshold = part.threshold_selected_inner.to_numpy(dtype=float)
        rec = reconstruct_metrics(y_all[part.row_index.to_numpy(dtype=int)], part.calibrated_probability.to_numpy(dtype=float), row_threshold)
        rec.update({"repeat": int(repeat), "model_id": model})
        bench = benchmark_repeat[(benchmark_repeat.repeat == repeat) & (benchmark_repeat.model_id == model)]
        if len(bench) != 1:
            raise ValueError(f"Missing/duplicate repeat benchmark at {repeat}/{model}")
        benchmark = bench.iloc[0]
        comparisons = {metric: _close(rec[metric], benchmark[metric]) for metric in METRICS}
        if not all(comparisons.values()):
            raise ValueError(f"Repeat metrics do not reconcile at {repeat}/{model}: {comparisons}")
        fold_auc = folds[(folds.repeat == repeat) & (folds.model_id == model)].roc_auc
        rec["unweighted_mean_fold_roc_auc_diagnostic_only"] = float(fold_auc.mean())
        rec["repeat_auc_not_fold_mean"] = not _close(rec["roc_auc"], float(fold_auc.mean()), tolerance=0.0)
        rec["benchmark_metrics_match"] = all(comparisons.values())
        repeats.append(rec)
    repeats_frame = pd.DataFrame(repeats)

    paired_rows = []
    for repeat in sorted(repeats_frame.repeat.unique()):
        this = repeats_frame[repeats_frame.repeat == repeat].set_index("model_id")
        selection = this.loc["selection_aware_procedure"]
        v3 = this.loc["frozen_v3_logistic_balanced"]
        values = {f"{metric}_difference_selection_minus_v3": float(selection[metric] - v3[metric]) for metric in ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity"]}
        reported = benchmark_paired[benchmark_paired.repeat == repeat]
        if len(reported) != 1:
            raise ValueError(f"Missing/duplicate paired repeat comparison for repeat {repeat}")
        for key, value in values.items():
            if not _close(value, reported.iloc[0][key]):
                raise ValueError(f"Paired repeat metric mismatch at repeat {repeat}/{key}")
        paired_rows.append({"repeat": int(repeat), **values, "benchmark_differences_match": True})
    paired_frame = pd.DataFrame(paired_rows)

    # Calibration result file labels outer-test metrics and training calibrator
    # parameters separately. Verify outer-test calibration entries from outputs.
    calibration_checks = []
    for row in calibration.itertuples(index=False):
        fold_row = folds[(folds.repeat == row.repeat) & (folds.fold == row.fold) & (folds.model_id == row.model_id)].iloc[0]
        calibration_checks.append({"repeat": int(row.repeat), "fold": int(row.fold), "model_id": row.model_id,
            "outer_test_intercept_match": _close(row.calibration_intercept_outer_test, fold_row.calibration_intercept),
            "outer_test_slope_match": _close(row.calibration_slope_outer_test, fold_row.calibration_slope),
            "training_calibrator_intercept_is_distinct_quantity": True})
    if not all(row["outer_test_intercept_match"] and row["outer_test_slope_match"] for row in calibration_checks):
        raise ValueError("Calibration evaluation metrics do not match outer-test recomputation")

    phase6_decision = (PHASE6 / "PHASE6_FINAL_DECISION.md").read_text(encoding="utf-8-sig")
    summary_auc = repeats_frame.groupby("model_id").roc_auc.mean()
    auc_delta = paired_frame.roc_auc_difference_selection_minus_v3
    decision_text_matches = all(f"{summary_auc['selection_aware_procedure']:.3f}" in phase6_decision and f"{summary_auc['frozen_v3_logistic_balanced']:.3f}" in phase6_decision and f"{auc_delta.mean():+.3f}" in phase6_decision for _ in [0])
    if not decision_text_matches:
        raise ValueError("Phase 6 final decision summary does not reconcile to repeat-level predictions")

    # Baseline split average and pooled-within-repeat AUC are deliberately both
    # returned so reports can demonstrate they are different estimands.
    split_sizes = pd.DataFrame(split_size_rows)
    return {
        "source_rows": len(data), "events": int(y_all.sum()), "prevalence": float(y_all.mean()),
        "predictions": predictions, "selections": selections, "folds": folds,
        "repeats": repeats_frame, "paired": paired_frame,
        "coverage": pd.DataFrame(coverage_rows), "metadata": pd.DataFrame(metadata_rows),
        "candidate_selections": pd.DataFrame(candidate_selection_rows),
        "split_sizes": split_sizes, "calibration_checks": pd.DataFrame(calibration_checks),
        "decision_text_matches": decision_text_matches,
        "phase6_manifest_check": verify_phase6_manifest(),
        "phase6_final_decision": phase6_decision,
    }


def write_package(result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folds: pd.DataFrame = result["folds"]
    repeats: pd.DataFrame = result["repeats"]
    paired: pd.DataFrame = result["paired"]
    coverage: pd.DataFrame = result["coverage"]
    metadata: pd.DataFrame = result["metadata"]
    splits: pd.DataFrame = result["split_sizes"]
    calibration_checks: pd.DataFrame = result["calibration_checks"]
    candidate_selections: pd.DataFrame = result["candidate_selections"]
    folds.to_csv(OUT / "RECONSTRUCTED_OUTER_FOLD_METRICS.csv", index=False)
    repeats.to_csv(OUT / "RECONSTRUCTED_REPEAT_METRICS.csv", index=False)
    paired.to_csv(OUT / "RECONSTRUCTED_PAIRED_REPEAT_DIFFERENCES.csv", index=False)
    coverage.to_csv(OUT / "OUTER_PREDICTION_COVERAGE.csv", index=False)
    metadata.to_csv(OUT / "CALIBRATION_SELECTION_METADATA_CORRECTED.csv", index=False)
    splits.to_csv(OUT / "CALIBRATION_TRAINING_SIZE_AUDIT.csv", index=False)
    calibration_checks.to_csv(OUT / "OUTER_CALIBRATION_METRIC_CHECKS.csv", index=False)
    candidate_selections.to_csv(OUT / "CANDIDATE_SELECTION_RECONCILIATION.csv", index=False)

    fold_averages = folds.groupby("model_id").roc_auc.mean().to_dict()
    repeats_means = repeats.groupby("model_id").roc_auc.mean().to_dict()
    actual_fold_counts = metadata.groupby(["repeat", "outer_fold", "model_id"]).calibration_fold.nunique()
    metadata_error_only = bool((actual_fold_counts == 3).all() and set(actual_fold_counts) == {3})
    if not metadata_error_only:
        raise RuntimeError("Corrected calibration-fold metadata does not match the three-fold implementation")
    metric_text = f"""# Phase 6 metric reconciliation

## Verification result

The saved outer predictions were independently matched to the original 303-row ZAS dataset and the frozen outer split manifest for repeats 1–3. There are 15 paired outer partitions and 1,818 saved prediction rows (303 rows × 3 repeats × 2 models). Every source row appears once in each repeat/model set, each partition matches the manifest test rows exactly, `source_row_id` matches the row index without asserting patient identity, and every saved `y_true` matches `Cath` mapped as Cad=1/Normal=0.

The selected feature group/model metadata matches every row in the corresponding outer fold. Reconstructed fold metrics are in `RECONSTRUCTED_OUTER_FOLD_METRICS.csv`; all ROC-AUC, average-precision PR-AUC, Brier, log loss, threshold-based sensitivity/specificity, and calibration intercept/slope values reconcile to `PERFORMANCE_COMPARISON.csv` within 1e-10. Every row has n=60 or 61. Average precision is shown with each fold's observed prevalence reference.

## Repeat-level reconstruction

Each repeat was rebuilt as one complete 303-source-row out-of-fold set before computing ROC-AUC and the other metrics. Results reconcile to `REPEAT_LEVEL_PERFORMANCE.csv`; paired differences reconcile to `PAIRED_REPEAT_DIFFERENCES.csv`.

| Model | Mean of repeat-level AUCs | Mean of 15 fold AUCs (diagnostic only) |
|---|---:|---:|
| Selection-aware procedure | {repeats_means['selection_aware_procedure']:.6f} | {fold_averages['selection_aware_procedure']:.6f} |
| Phase 6 V3 reference comparator | {repeats_means['frozen_v3_logistic_balanced']:.6f} | {fold_averages['frozen_v3_logistic_balanced']:.6f} |

The unweighted fold-AUC average is not substituted for the AUC computed across a repeat's full 303 out-of-fold predictions. The paired repeat-level ROC-AUC differences (selection minus V3 comparator) are {', '.join(f'{x:+.6f}' for x in paired.roc_auc_difference_selection_minus_v3)}; their mean is {paired.roc_auc_difference_selection_minus_v3.mean():+.6f}. This reconciles with the rounded Phase 6 decision claim.

## Conclusion

**Outer discrimination is verified from the saved predictions.** No metric discrepancy was found. This does not establish patient-level independence or population-level uncertainty; rows are the only verified split unit.
"""
    (OUT / "PHASE6_METRIC_RECONCILIATION.md").write_text(metric_text, encoding="utf-8")

    sizes = splits.iloc[0]
    calibration_text = f"""# Phase 6 calibration audit

## What the code does

Within each outer training partition, `choose_candidate` (or `choose_v3`) first uses all three inner folds to select the group/family/hyperparameters. Then `crossfit_selected_scores` holds out each of those three inner folds in turn and predicts it from a model fitted on the other two inner folds, with the selected parameters held fixed. `fit_sigmoid` fits a logistic map on the resulting outer-training OOF scores and labels. `choose_threshold` chooses Youden J on those same calibrated training scores. Finally the selected model is refit on all outer-training rows and the learned sigmoid is applied to the outer-test predictions.

## Calibration limitations

**The cross-fit predictions are selection-dependent.** Hyperparameter and feature-group selection uses labels from all outer-training rows, including labels for rows that later appear in the inner validation blocks used to generate the sigmoid-training scores. Thus each base-model score is out-of-fold with respect to the estimator fit, but not independent of the model-selection procedure. The sigmoid and threshold are also fitted/selected on those same training OOF scores.

There is a training-size/score-scale mismatch: each inner scoring model is fitted on roughly two thirds of the outer-training rows, while the final outer predictor is refit on all outer-training rows. For this split, the first outer training partition has {int(sizes.outer_train_n)} rows; inner fits have sizes {sizes.inner_fit_sizes}, inner validation blocks have sizes {sizes.inner_validation_sizes}, and the final model has {int(sizes.final_model_training_n)} training rows. Across folds the sizes vary by one row. The saved outer-test calibration results therefore evaluate the practical effect of applying this calibrator to the full-training refit, but they do not remove the calibration construction limitation.

## Outer test use and validity

Outer-test labels are not passed into model selection, preprocessing, sigmoid fitting, or threshold choice. They are used after the model/calibrator/threshold are fixed to produce performance metrics. The runner also uses outer-test features and labels for a post-fit permutation-importance analysis; those results do not feed back into selection. Thus the test set is not literally untouched by all analysis, but its labels do not determine the saved predictions or selected procedure. The paired outer discrimination estimate remains valid as an evaluation of the complete training/selection/calibration procedure represented by the code, subject to source-row independence and the three-repeat limitation.

Fold- and repeat-level calibration intercept/slope are recomputed from outer-test predictions only. The fields `calibration_intercept_training` and `calibration_slope_training` are fitted sigmoid parameters on training OOF scores, not test calibration metrics. The saved outer test estimates reconcile exactly to the independent calculations. These estimates can be unstable for folds of about 60 rows and do not validate a fixed final deployable model.

## Classification

Calibration is **limited, not independently cross-fitted with respect to feature/model selection**. This is a calibration limitation, not a discovered outer ROC-AUC leakage path. The outer folds still assess the whole selection procedure without outer-label selection.
"""
    (OUT / "PHASE6_CALIBRATION_AUDIT.md").write_text(calibration_text, encoding="utf-8")

    manifest_check = result["phase6_manifest_check"]
    metadata_text = f"""# Phase 6 implementation audit

## Calibration metadata count

The runner has a three-fold inner design (`StratifiedKFold(n_splits=3)`), and `crossfit_selected_scores` iterates exactly over those three splits. However, the reporting list `calibration_selection` (and its V3 analogue) is constructed with `range(1, 5)`, creating four metadata entries with calibration_fold labels 1–4. The fourth entry duplicates the same selected group, family and parameters; it does not correspond to a score block or a fourth model fit. `calibration_crossfit_rows` and the actual calibration scores are produced from the three loop iterations.

**Finding: metadata/reporting error only.** The saved four-entry JSON does not drive calibration, outer probability generation, threshold selection, metrics, or candidate selection. `CALIBRATION_SELECTION_METADATA_CORRECTED.csv` supplies a corrected three-row display for each of the 30 outer model records and explicitly records that selection occurred once over all three inner folds rather than once per calibration fold. Historical Phase 6 CSVs remain untouched.

## Split, selection and output checks

- Frozen split manifest: {len(result['coverage']) // 2} outer partition/model pairs verified over the first three of five repeats.
- Unique source rows per repeat/model: 303; original target labels match: all verified.
- Feature-group/model metadata agrees between predictions and outer selections: all verified.
- Phase 6 result/output and input hashes: {len(manifest_check['outputs'])} outputs and {len(manifest_check['inputs'])} inputs checked; all match: {manifest_check['all_match']}.
- The original Phase 6 output manifest itself is not self-hashed; this audit verifies every file enumerated by that manifest.

No predictive-classifier fitting, hyperparameter search, threshold optimization or Phase 6 command was run by this audit. Logistic regressions were fit only to recompute calibration intercept/slope as evaluation metrics from saved outer predictions; these fits did not produce predictions or influence selection.

## Project environment

The protocol commands invoke the project-local `.venv\\Scripts\\python.exe`. Its runtime resolves `sys.executable` and `sys.prefix` to `CARDIODIAG_FRESH\\.venv`, with base Python 3.11.16. Verified versions: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, pytest 9.2.0, and Ruff 0.17.4. `requirements.txt` and `requirements-runner.txt` contain ranges/unpinned packages rather than a lock file; these observed virtual-environment versions are the exact versions used here. No installation was performed.
"""
    (OUT / "PHASE6_IMPLEMENTATION_AUDIT.md").write_text(metadata_text, encoding="utf-8")

    decision_text = """# Phase 6.1 final decision

## Classification

**1. Outer discrimination verified; calibration limitations remain.**

The 1,818 saved predictions match the first 15 frozen outer partitions exactly; all source-row labels and selected-group/model annotations agree. Fold metrics, complete-repeat metrics, and paired repeat differences independently reconstruct to the Phase 6 tables and final decision. No material prediction or discrimination discrepancy was found.

Calibration has a real limitation: selected candidate/hyperparameters are chosen using all outer-training labels before inner OOF scores are used to fit the sigmoid and choose threshold. The crossfit scores are therefore estimator-out-of-fold but selection-dependent. Inner-fit models use about two thirds of the outer-training rows while the final predictor is refit on all of them. The calibration selection metadata also has four labels for a three-fold loop; this is a reporting-only defect, corrected in the new audit CSV.

This limitation does not expose outer-test labels to prediction selection or calibration. The outer test labels are used for evaluation and post-fit permutation importance only. Accordingly, outer discrimination remains verified as the performance of the complete training procedure; calibration evidence should be described as limited and must not be presented as independent calibration validation for a fixed final model.

## Supported, limited and unresolved

- **Supported:** source-row prediction coverage, label matching, per-fold and per-repeat metric reconstruction, paired comparison arithmetic, Phase 6 manifest file hashes, and the 4-versus-3 metadata error.
- **Limited:** calibration construction and calibration/threshold estimates; only three repeated outer splits; rows are not verified patients; calibration slope/intercept on small fold samples.
- **Unresolved:** patient identity/independence, feature acquisition timing, external generalization, and whether calibration would transport to a fixed model trained on all rows.

No patient-level confidence claim or external-validation claim is made. Historical Phase 1–6 files were not changed.

## Reproduction commands

From the project root in PowerShell, the documented project virtual environment is `.venv\\Scripts\\python.exe` (verified to resolve to the project-local interpreter). Run only the read-only audit and focused tests:

```powershell
$env:PYTHONPATH = 'src;.'
.\\.venv\\Scripts\\python.exe -m runner.phase6_1_audit.audit
.\\.venv\\Scripts\\python.exe -m pytest tests\\test_phase6_1_audit.py -q
```

Do not run `runner.phase6_zas_selection.run` to reproduce this audit; that command refits the Phase 6 candidate pool and is unnecessary here. The audit reads saved predictions/splits, recalculates metrics, verifies hashes and writes only `results/phase6_1_audit/`.
"""
    (OUT / "PHASE6_1_FINAL_DECISION.md").write_text(decision_text, encoding="utf-8")

    output_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "PHASE6_1_MANIFEST.json")
    phase6_inputs = [PHASE6 / name for name in ["OUTER_FOLD_PREDICTIONS.csv", "OUTER_MODEL_SELECTIONS.csv", "PERFORMANCE_COMPARISON.csv", "REPEAT_LEVEL_PERFORMANCE.csv", "PAIRED_REPEAT_DIFFERENCES.csv", "CALIBRATION_RESULTS.csv", "PHASE6_FINAL_DECISION.md", "PHASE6_MANIFEST.json"]]
    inputs = [DATA, SPLITS, PHASE6_RUNNER, PHASE6_PROTOCOL, *phase6_inputs, ROOT / "runner/phase6_1_audit/audit.py", ROOT / "tests/test_phase6_1_audit.py", ROOT / ".venv/pyvenv.cfg", ROOT / "requirements.txt", ROOT / "requirements-runner.txt"]
    manifest = {
        "audit": "Phase 6.1 independent saved-artifact verification",
        "created_utc": datetime.now(UTC).isoformat(),
        "decision": "1. Outer discrimination verified; calibration limitations remain.",
        "predictive_classifier_fits": 0,
        "hyperparameter_searches": 0,
        "threshold_optimizations": 0,
        "calibration_regression_metric_fits": 36,
        "historical_results_modified": False,
        "checks": {"source_rows": result["source_rows"], "events": result["events"], "outer_partitions": 15, "prediction_rows": len(result["predictions"]), "fold_metrics_reconciled": bool(result["folds"].benchmark_metrics_match.all()), "repeat_metrics_reconciled": bool(result["repeats"].benchmark_metrics_match.all()), "paired_metrics_reconciled": bool(result["paired"].benchmark_differences_match.all()), "phase6_manifest_all_hashes_match": result["phase6_manifest_check"]["all_match"], "phase6_manifest_output_hash_count": len(result["phase6_manifest_check"]["outputs"]), "phase6_manifest_input_hash_count": len(result["phase6_manifest_check"]["inputs"]), "calibration_metadata_reporting_count_error": True},
        "environment": {"python_executable": sys.executable, "venv_prefix": sys.prefix, "python_base_prefix": sys.base_prefix, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pytest": importlib.metadata.version("pytest"), "ruff": importlib.metadata.version("ruff")},
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in inputs if path.is_file()},
        "outputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in output_files},
    }
    (OUT / "PHASE6_1_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))


def main() -> None:
    result = reconcile()
    if not result["phase6_manifest_check"]["all_match"]:
        raise RuntimeError("Phase 6 manifest verification failed")
    write_package(result)


if __name__ == "__main__":
    main()
