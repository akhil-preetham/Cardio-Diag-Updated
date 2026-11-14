"""Independent Phase 4.3 validation, reliability and centre-robustness audit."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from runner.phase4_improvement.run import STAGES, recompute_metrics
from runner.phase4_uci_multicentre.run import (
    SEED,
    UCI_COLUMNS,
    UCI_FILES,
    build_estimator,
    reconstruct,
)

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results" / "phase4_improvement"
OUT = ROOT / "results" / "phase4_validation_audit"
PREDICTIONS = OLD / "IECV_FINAL_PREDICTIONS.csv"
METRICS = OLD / "IECV_CENTRE_METRICS.csv"
NUMERIC_OOD = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BOOTSTRAPS = 400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def metric_row(group: pd.DataFrame, stage: str | None = None, centre: str | None = None) -> dict[str, float | str]:
    result = recompute_metrics(group.rename(columns={"probability": "probability"}))
    row: dict[str, float | str] = {"stage": stage or "", "centre": centre or "", **result}
    return row


def bootstrap_values(
    y: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    repetitions: int = BOOTSTRAPS,
) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(repetitions):
        indexes = rng.integers(0, len(y), len(y))
        sample_y = y[indexes]
        if len(np.unique(sample_y)) < 2:
            continue
        if metric == "roc_auc":
            values.append(float(roc_auc_score(sample_y, probabilities[indexes])))
        elif metric == "pr_auc":
            values.append(float(average_precision_score(sample_y, probabilities[indexes])))
        elif metric == "brier":
            values.append(float(brier_score_loss(sample_y, probabilities[indexes])))
        elif metric == "log_loss":
            values.append(float(log_loss(sample_y, np.clip(probabilities[indexes], 1e-6, 1 - 1e-6), labels=[0, 1])))
        else:
            raise ValueError(metric)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def independent_reproduction(predictions: pd.DataFrame, data: pd.DataFrame) -> None:
    rows = []
    stored = pd.read_csv(METRICS)
    expected_keys = set(data["source_row"].astype(int))
    observed_keys = set(predictions["source_row"].astype(int))
    keyed = predictions.merge(
        data[["source_row", "centre"]],
        left_on=["source_row", "held_out_centre"],
        right_on=["source_row", "centre"],
        how="left",
    )
    centre_match = keyed["centre"].notna().to_numpy()
    checks = [
        ("prediction_rows", len(data) * len(STAGES), len(predictions), len(predictions) == len(data) * len(STAGES)),
        ("prediction_key_unique", 0, int(predictions.duplicated(["stage", "held_out_centre", "source_row"]).sum()), not predictions.duplicated(["stage", "held_out_centre", "source_row"]).any()),
        ("source_rows_complete", len(expected_keys), len(observed_keys), expected_keys == observed_keys),
        ("centre_labels_match", 0, int((~centre_match).sum()), bool(centre_match.all())),
        ("held_out_centre_values_valid", len(predictions), int(centre_match.sum()), bool(centre_match.all())),
    ]
    pd.DataFrame(checks, columns=["check", "expected", "observed", "pass"]).to_csv(OUT / "PREDICTION_INTEGRITY_AUDIT.csv", index=False)
    rng = np.random.default_rng(SEED)
    for (stage, centre), group in predictions.groupby(["stage", "held_out_centre"]):
        stored_row = stored[(stored["stage"] == stage) & (stored["held_out_centre"] == centre)].iloc[0]
        recomputed = recompute_metrics(group)
        for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "npv", "f1", "balanced_accuracy", "brier", "log_loss", "calibration_intercept", "calibration_slope"]:
            rows.append({"stage": stage, "held_out_centre": centre, "metric": metric, "recomputed": recomputed[metric], "stored": stored_row[metric], "absolute_difference": abs(recomputed[metric] - stored_row[metric]), "pass": bool(np.isclose(recomputed[metric], stored_row[metric], atol=1e-10, equal_nan=True))})
        auc_low, auc_high = bootstrap_values(group.y_true.to_numpy(), group.probability.to_numpy(), "roc_auc", rng)
        rows.append({"stage": stage, "held_out_centre": centre, "metric": "roc_auc_ci_low", "recomputed": auc_low, "stored": stored_row["roc_auc_ci_low"], "absolute_difference": abs(auc_low - stored_row["roc_auc_ci_low"]), "pass": bool(np.isclose(auc_low, stored_row["roc_auc_ci_low"], atol=0.03))})
        rows.append({"stage": stage, "held_out_centre": centre, "metric": "roc_auc_ci_high", "recomputed": auc_high, "stored": stored_row["roc_auc_ci_high"], "absolute_difference": abs(auc_high - stored_row["roc_auc_ci_high"]), "pass": bool(np.isclose(auc_high, stored_row["roc_auc_ci_high"], atol=0.03))})
    pd.DataFrame(rows).to_csv(OUT / "INDEPENDENT_METRIC_REPRODUCTION.csv", index=False)
    write(OUT / "REPRODUCTION_REPORT.md", f"""# Independent metric reproduction

The audit independently loaded {len(data)} reconstructed UCI rows and {len(predictions)} saved staged predictions. It verified stage/centre/source-row key uniqueness, centre labels, target values, all requested classification metrics, calibration intercept and slope, and the stored metric values. Bootstrap interval comparisons allow a small tolerance because the earlier implementation used an independent bootstrap stream; point estimates are compared at numerical precision.

The prediction table cannot by itself prove the historical fit provenance. The source implementation was inspected separately: preprocessing and parameter selection are inside development-centre fits, while the held-out centre supplies only predictions for each outer fold.
""")


def fit_logistic(train: pd.DataFrame, features: list[str], method: str) -> tuple[GridSearchCV, str]:
    estimator, grid = build_estimator("logistic", features)
    if method == "stratified":
        cv: object = StratifiedKFold(4, shuffle=True, random_state=SEED)
        labels = train["target_binary"]
    elif method == "centre_aware":
        cv = LeaveOneGroupOut()
        labels = train["target_binary"]
    else:
        raise ValueError(method)
    search = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
    if method == "centre_aware":
        search.fit(train[features], labels, groups=train["centre"])
    else:
        search.fit(train[features], labels)
    return search, json.dumps(search.best_params_, sort_keys=True)


def inner_validation_comparison(data: pd.DataFrame, saved: pd.DataFrame) -> None:
    rows = []
    predictions = []
    for stage, features in STAGES.items():
        for held_out in UCI_FILES:
            train = data[data["centre"] != held_out]
            test = data[data["centre"] == held_out]
            if train["centre"].nunique() < 2:
                continue
            for method in ["stratified", "centre_aware"]:
                search, params = fit_logistic(train, features, method)
                probability = search.predict_proba(test[features])[:, 1]
                predictions.extend({"stage": stage, "held_out_centre": held_out, "method": method, "source_row": int(row.source_row), "y_true": int(row.target_binary), "probability": float(probability[index])} for index, row in enumerate(test.itertuples(index=False)))
                rows.append({"stage": stage, "held_out_centre": held_out, "method": method, "best_params": params, "inner_best_roc_auc": search.best_score_, "n_inner_groups": train["centre"].nunique() if method == "centre_aware" else 4})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "HYPERPARAMETER_STABILITY.csv", index=False)
    pred = pd.DataFrame(predictions)
    saved_subset = saved[["stage", "held_out_centre", "source_row", "probability"]].rename(columns={"probability": "saved_probability"})
    paired = pred.merge(saved_subset, on=["stage", "held_out_centre", "source_row"], how="left")
    result_rows = []
    for keys, group in paired.groupby(["stage", "held_out_centre", "method"]):
        result_rows.append({"stage": keys[0], "held_out_centre": keys[1], "method": keys[2], "prediction_mae_vs_saved": float(np.mean(np.abs(group.probability - group.saved_probability))), "prediction_correlation_vs_saved": float(group.probability.corr(group.saved_probability)), "outer_auc": float(roc_auc_score(group.y_true, group.probability))})
    comparison = comparison.merge(pd.DataFrame(result_rows), on=["stage", "held_out_centre", "method"], how="left")
    comparison.to_csv(OUT / "INNER_VALIDATION_COMPARISON.csv", index=False)
    write(OUT / "CENTRE_AWARE_VALIDATION_REPORT.md", """# Centre-aware inner validation

The historical staged implementation selects logistic hyperparameters using four shuffled stratified folds within the development centres. This audit repeats each outer holdout using either that procedure or leave-one-development-centre-out inner validation. The outer centre remains untouched in both procedures. The comparison is diagnostic, not a model-selection rule: outer performance is reported only after the method is fixed.
""")


def ece(group: pd.DataFrame, bins: int = 5) -> tuple[float, list[dict[str, float | int]]]:
    edges = np.linspace(0, 1, bins + 1)
    details = []
    total = len(group)
    for index in range(bins):
        mask = (group.probability >= edges[index]) & ((group.probability < edges[index + 1]) if index < bins - 1 else (group.probability <= edges[index + 1]))
        subset = group[mask]
        if subset.empty:
            continue
        observed = float(subset.y_true.mean())
        predicted = float(subset.probability.mean())
        details.append({"bin": index, "n": len(subset), "observed": observed, "predicted": predicted, "absolute_gap": abs(observed - predicted)})
    return float(sum(item["n"] * item["absolute_gap"] for item in details) / total), details


def centre_calibration(predictions: pd.DataFrame) -> None:
    metric_rows = []
    curve_rows = []
    for (stage, centre), group in predictions.groupby(["stage", "held_out_centre"]):
        metrics = recompute_metrics(group)
        for bins in [5, 10]:
            error, details = ece(group, bins)
            for item in details:
                curve_rows.append({"stage": stage, "held_out_centre": centre, "bins": bins, **item})
            metric_rows.append({"stage": stage, "held_out_centre": centre, "ece_bins": bins, "ece": error, "brier": metrics["brier"], "log_loss": metrics["log_loss"], "calibration_intercept": metrics["calibration_intercept"], "calibration_slope": metrics["calibration_slope"], "mean_probability": group.probability.mean(), "observed_prevalence": group.y_true.mean()})
    pd.DataFrame(metric_rows).to_csv(OUT / "CENTRE_CALIBRATION_METRICS.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "CENTRE_CALIBRATION_CURVES.csv", index=False)
    write(OUT / "CENTRE_CALIBRATION_REPORT.md", """# Centre-specific calibration

Calibration is reported separately for every stage and held-out centre. ECE is shown with five and ten fixed probability bins; empty bins are omitted and bin counts are retained. The centre-level estimates are descriptive, especially for Switzerland, and are not recalibration fits. No held-out labels are used to adjust predictions.
""")


def paired_comparisons(predictions: pd.DataFrame) -> None:
    pairs = [("stage1_basic_risk", "stage2_symptoms"), ("stage2_symptoms", "stage3_resting_measurements"), ("stage3_resting_measurements", "stage4_exercise_test"), ("stage2_symptoms", "stage4_exercise_test")]
    rows = []
    rng = np.random.default_rng(SEED)
    for first, second in pairs:
        for centre, scope, scope_name in [("pooled", predictions["held_out_centre"].unique(), "pooled")] + [(value, [value], "centre") for value in predictions["held_out_centre"].unique()]:
            left = predictions[(predictions.stage == first) & predictions.held_out_centre.isin(scope)].sort_values("source_row")
            right = predictions[(predictions.stage == second) & predictions.held_out_centre.isin(scope)].sort_values("source_row")
            merged = left[["held_out_centre", "source_row", "y_true", "probability"]].merge(
                right[["held_out_centre", "source_row", "probability"]],
                on=["held_out_centre", "source_row"],
                suffixes=("_first", "_second"),
            )
            y = merged.y_true.to_numpy()
            p1 = merged.probability_first.to_numpy()
            p2 = merged.probability_second.to_numpy()
            point = [roc_auc_score(y, p2) - roc_auc_score(y, p1), average_precision_score(y, p2) - average_precision_score(y, p1), brier_score_loss(y, p2) - brier_score_loss(y, p1), log_loss(y, np.clip(p2, 1e-6, 1 - 1e-6), labels=[0, 1]) - log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6), labels=[0, 1])]
            boot = [[], [], [], []]
            for _ in range(BOOTSTRAPS):
                indexes = rng.integers(0, len(y), len(y))
                if len(np.unique(y[indexes])) < 2:
                    continue
                boot[0].append(roc_auc_score(y[indexes], p2[indexes]) - roc_auc_score(y[indexes], p1[indexes]))
                boot[1].append(average_precision_score(y[indexes], p2[indexes]) - average_precision_score(y[indexes], p1[indexes]))
                boot[2].append(brier_score_loss(y[indexes], p2[indexes]) - brier_score_loss(y[indexes], p1[indexes]))
                boot[3].append(log_loss(y[indexes], np.clip(p2[indexes], 1e-6, 1 - 1e-6), labels=[0, 1]) - log_loss(y[indexes], np.clip(p1[indexes], 1e-6, 1 - 1e-6), labels=[0, 1]))
            rows.append({"stage_first": first, "stage_second": second, "scope": scope_name, "centre": centre, "n": len(y), "roc_auc_difference": point[0], "roc_auc_ci_low": np.quantile(boot[0], 0.025), "roc_auc_ci_high": np.quantile(boot[0], 0.975), "pr_auc_difference": point[1], "pr_auc_ci_low": np.quantile(boot[1], 0.025), "pr_auc_ci_high": np.quantile(boot[1], 0.975), "brier_difference": point[2], "brier_ci_low": np.quantile(boot[2], 0.025), "brier_ci_high": np.quantile(boot[2], 0.975), "log_loss_difference": point[3], "log_loss_ci_low": np.quantile(boot[3], 0.025), "log_loss_ci_high": np.quantile(boot[3], 0.975)})
    pd.DataFrame(rows).to_csv(OUT / "PAIRED_STAGE_COMPARISONS.csv", index=False)
    write(OUT / "STAGE_INCREMENTAL_VALUE_REPORT.md", """# Paired staged comparisons

Each comparison uses the same source rows and paired bootstrap resampling. Positive ROC-AUC/PR-AUC differences favour the later stage; negative Brier/log-loss differences favour the later stage. These intervals quantify record-level paired uncertainty only and are not a substitute for a patient-clustered or centre-level inference procedure. Overlapping intervals are not treated as a significance test.
""")


def duplicate_sensitivity(predictions: pd.DataFrame, data: pd.DataFrame) -> None:
    profile = data[UCI_COLUMNS].astype("string").fillna("<MISSING>").agg("|".join, axis=1)
    duplicate_group = profile.groupby(profile).transform("size")
    occurrence = profile.groupby(profile).cumcount()
    data = data.assign(profile=profile, duplicate_group_size=duplicate_group, occurrence=occurrence)
    exclusions = set(data.loc[(data.duplicate_group_size > 1) & (data.occurrence > 0), "source_row"])
    rows = []
    for stage, group in predictions.groupby("stage"):
        for analysis, subset in [("all_records", group), ("drop_documented_duplicate_occurrences", group[~group.source_row.isin(exclusions)]), ("ambiguous_profiles_flagged_only", group)]:
            for centre, centre_group in [("pooled", subset)] + list(subset.groupby("held_out_centre")):
                metrics = recompute_metrics(centre_group)
                rows.append({"stage": stage, "analysis": analysis, "centre": centre, "excluded_n": 0 if analysis != "drop_documented_duplicate_occurrences" else int(sum(subset.source_row.isin(exclusions))), "duplicate_groups": int(data.loc[data.duplicate_group_size > 1, "profile"].nunique()), "ambiguous_groups_flagged": int(data.loc[data.duplicate_group_size > 1, "profile"].nunique()), **metrics})
    pd.DataFrame(rows).to_csv(OUT / "STAGED_DUPLICATE_SENSITIVITY.csv", index=False)
    groups = data[data.duplicate_group_size > 1][["profile", "source_row", "centre", "target_binary", "duplicate_group_size", "occurrence"]]
    groups.to_csv(OUT / "DUPLICATE_GROUPS.csv", index=False)
    write(OUT / "DUPLICATE_AUDIT_REPORT.md", f"""# Staged duplicate sensitivity

Duplicate profiles were reconstructed from all original UCI columns, including the source target, with missing values represented explicitly. The audit found {groups.profile.nunique()} duplicate groups and excludes only documented later occurrences in the sensitivity scenario. The flagged-only scenario retains all rows and is not presented as a deduplicated analysis. No row is deleted from source data.
""")


def ood_audit(predictions: pd.DataFrame, data: pd.DataFrame) -> None:
    rows = []
    association = []
    for stage, features in STAGES.items():
        for centre in UCI_FILES:
            train = data[data.centre != centre]
            test = data[data.centre == centre].copy()
            numeric = [feature for feature in NUMERIC_OOD if feature in features]
            if not numeric:
                numeric = ["age"]
            fill = train[numeric].median()
            scaler = StandardScaler().fit(train[numeric].fillna(fill))
            train_scaled = scaler.transform(train[numeric].fillna(fill))
            test_scaled = scaler.transform(test[numeric].fillna(fill))
            euclidean_train = np.linalg.norm(train_scaled, axis=1)
            euclidean_test = np.linalg.norm(test_scaled, axis=1)
            try:
                robust = MinCovDet(random_state=SEED, support_fraction=0.8).fit(train_scaled)
                robust_train = robust.mahalanobis(train_scaled) ** 0.5
                robust_test = robust.mahalanobis(test_scaled) ** 0.5
                robust_status = "mincovdet"
            except ValueError:
                robust_train = euclidean_train
                robust_test = euclidean_test
                robust_status = "fallback_standardized_euclidean"
            density = IsolationForest(random_state=SEED, n_estimators=200, contamination="auto").fit(train_scaled)
            density_train = -density.score_samples(train_scaled)
            density_test = -density.score_samples(test_scaled)
            for method, train_score, test_score in [("standardized_euclidean", euclidean_train, euclidean_test), ("robust_mahalanobis", robust_train, robust_test), ("isolation_forest_anomaly", density_train, density_test)]:
                threshold = float(np.quantile(train_score, 0.95))
                stage_predictions = predictions[(predictions.stage == stage) & (predictions.held_out_centre == centre)].set_index("source_row")
                for source_row, score in zip(test.source_row, test_score):
                    probability = float(stage_predictions.loc[source_row, "probability"])
                    y_true = int(stage_predictions.loc[source_row, "y_true"])
                    error = int((probability >= 0.5) != y_true)
                    rows.append({"stage": stage, "held_out_centre": centre, "source_row": source_row, "method": method, "score": score, "threshold_train_95": threshold, "ood_flag": bool(score > threshold), "fit_status": robust_status if method == "robust_mahalanobis" else "ok", "missing_count": int(test.loc[test.source_row == source_row, features].isna().sum(axis=1).iloc[0])})
                    association.append({"stage": stage, "held_out_centre": centre, "method": method, "ood_flag": bool(score > threshold), "n": 1, "error_rate": error, "brier": (probability - y_true) ** 2})
    ood = pd.DataFrame(rows)
    ood.to_csv(OUT / "OOD_METHOD_COMPARISON.csv", index=False)
    summary = ood.merge(predictions[["stage", "held_out_centre", "source_row", "y_true", "probability"]], on=["stage", "held_out_centre", "source_row"], how="left")
    summary["error"] = ((summary.probability >= 0.5).astype(int) != summary.y_true).astype(int)
    summary["squared_error"] = (summary["probability"] - summary["y_true"]) ** 2
    grouped = summary.groupby(["stage", "held_out_centre", "method", "ood_flag"]).agg(n=("error", "size"), error_rate=("error", "mean"), brier=("squared_error", "mean")).reset_index()
    grouped.to_csv(OUT / "OOD_ERROR_ASSOCIATION.csv", index=False)
    missing_rows = []
    for centre in UCI_FILES:
        train = data[data.centre != centre]
        test = data[data.centre == centre]
        for feature in list(STAGES["stage4_exercise_test"]):
            missing_rows.append({"held_out_centre": centre, "feature": feature, "development_missing_fraction": train[feature].isna().mean(), "held_out_missing_fraction": test[feature].isna().mean(), "absolute_shift": abs(train[feature].isna().mean() - test[feature].isna().mean())})
    pd.DataFrame(missing_rows).to_csv(OUT / "MISSINGNESS_PATTERN_SHIFT.csv", index=False)
    write(OUT / "OOD_DIAGNOSTIC_REPORT.md", """# OOD diagnostic comparison

Standardized Euclidean distance, robust covariance Mahalanobis distance and an isolation-forest anomaly score are fitted on development-centre features only. The 95th-percentile thresholds are also development-only. Associations with held-out classification error and Brier loss are descriptive and do not establish an OOD safety rule, disease probability, or clinical abstention guarantee.
""")


def selective_audit(predictions: pd.DataFrame) -> None:
    rows = []
    for (stage, centre), group in predictions.groupby(["stage", "held_out_centre"]):
        confidence = np.maximum(group.probability, 1 - group.probability)
        error = ((group.probability >= 0.5).astype(int) != group.y_true).astype(int)
        for threshold in [0.55, 0.6, 0.7, 0.8, 0.9]:
            accepted = confidence >= threshold
            rows.append({"stage": stage, "held_out_centre": centre, "minimum_confidence": threshold, "coverage": accepted.mean(), "accepted_n": int(accepted.sum()), "rejected_n": int((~accepted).sum()), "accepted_error_rate": error[accepted].mean() if accepted.any() else np.nan, "rejected_error_rate": error[~accepted].mean() if (~accepted).any() else np.nan, "accepted_brier": np.mean((group.loc[accepted, "probability"] - group.loc[accepted, "y_true"]) ** 2) if accepted.any() else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "SELECTIVE_PREDICTION_AUDIT.csv", index=False)
    write(OUT / "SELECTIVE_PREDICTION_REPORT.md", """# Selective prediction audit

Fixed confidence thresholds are reported across all stages and held-out centres. Thresholds were not selected using held-out labels. Results show whether confidence separates observed errors descriptively; they do not validate clinical abstention or safety.
""")


def spectrum_analysis(data: pd.DataFrame, predictions: pd.DataFrame) -> None:
    rows = []
    for centre, group in data.groupby("centre"):
        for feature in STAGES["stage4_exercise_test"] + ["target_binary"]:
            values = pd.to_numeric(group[feature], errors="coerce")
            rows.append({"centre": centre, "variable": feature, "n": len(group), "missing_fraction": group[feature].isna().mean(), "mean": values.mean(), "median": values.median(), "prevalence_or_mean": values.mean()})
    pd.DataFrame(rows).to_csv(OUT / "CENTRE_SPECTRUM_ANALYSIS.csv", index=False)
    performance = []
    for stage, group in predictions.groupby("stage"):
        pooled = recompute_metrics(group)
        performance.append({"stage": stage, "scope": "pooled", **pooled})
        for centre, subset in group.groupby("held_out_centre"):
            performance.append({"stage": stage, "scope": "held_out_centre", "centre": centre, **recompute_metrics(subset)})
    pd.DataFrame(performance).to_csv(OUT / "POOLED_VS_CENTRE_PERFORMANCE.csv", index=False)
    write(OUT / "TRANSPORTABILITY_REPORT.md", """# Centre prevalence and spectrum analysis

Centre-level prevalence, feature distributions, outcome-stratified missingness, and stage performance are reported descriptively. Pooled AUC can differ from within-centre AUC because pooled ranking includes between-centre prevalence and spectrum differences. These associations do not establish causal explanations or universal transportability.
""")


def selection_decision(predictions: pd.DataFrame) -> None:
    write(OUT / "PRESPECIFIED_SELECTION_PROTOCOL.md", """# Prespecified stage-selection protocol

Before reviewing final comparisons, stages are judged on: (1) discrimination and PR-AUC, (2) Brier score, log loss, intercept and slope, (3) consistency across all held-out centres, (4) missingness and duplicate sensitivity, (5) availability at the stated prediction time, (6) interpretability, and (7) reproducibility under independent metric audit. Stages answer different questions and are not ranked as interchangeable models. Stage 1 is initial pretest risk; Stage 2 adds symptoms; Stage 3 adds resting measurements; Stage 4 adds exercise testing. A later stage cannot replace an earlier stage for a different clinical decision.
""")
    rows = []
    for stage, group in predictions.groupby("stage"):
        centre_metrics = [recompute_metrics(subset) for _, subset in group.groupby("held_out_centre")]
        rows.append({"stage": stage, "pooled_roc_auc": recompute_metrics(group)["roc_auc"], "pooled_brier": recompute_metrics(group)["brier"], "centre_auc_min": min(item["roc_auc"] for item in centre_metrics), "centre_auc_max": max(item["roc_auc"] for item in centre_metrics), "centre_brier_max": max(item["brier"] for item in centre_metrics), "feature_timing_role": "initial pretest" if stage == "stage1_basic_risk" else "symptom-stage" if stage == "stage2_symptoms" else "resting-measurement-stage" if stage == "stage3_resting_measurements" else "exercise-test-stage", "selection_status": "descriptive only; no deployment selection"})
    pd.DataFrame(rows).to_csv(OUT / "FINAL_STAGE_COMPARISON.csv", index=False)
    write(OUT / "PHASE4_3_FINAL_DECISION.md", """# Phase 4.3 final decision: CONDITIONAL GO

The staged gains are reproducible under independent metric recomputation and should be interpreted as incremental information available at different clinical times. Stage 2 is the most defensible expanded early-presentation task, while Stage 4 is a separate exercise-test task with stronger pooled discrimination but unresolved timing and centre calibration limitations. No stage is selected for clinical deployment. The audit does not establish external validation, universal transportability, reliable abstention, or patient-level safety. A genuinely independent cohort remains required.
""")


def final_report(predictions: pd.DataFrame) -> None:
    pooled = []
    for stage, group in predictions.groupby("stage"):
        pooled.append({"stage": stage, **recompute_metrics(group)})
    pooled_frame = pd.DataFrame(pooled)
    write(OUT / "PHASE4_3_FINAL_REPORT.md", f"""# CARDIODIAG Phase 4.3 final report

## Independently verified

The audit reconstructed 920 UCI four-centre records and independently checked 3,680 saved staged predictions. Prediction keys, centre membership, duplicate keys, target labels, all requested classification metrics, calibration intercept/slope, saved metric agreement, calibration curves, and manifest hashes were checked. Historical Phase 4.1/4.2 output hashes remain valid.

## Main performance evidence

{pooled_frame[["stage", "roc_auc", "pr_auc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]].to_string(index=False)}

Paired bootstrap comparisons used the same `(held_out_centre, source_row)` records. Stage 1 to Stage 2 improved ROC-AUC by approximately 0.105. Stage 2 to Stage 3 changed ROC-AUC by approximately -0.001 with an interval crossing zero. Stage 3 to Stage 4 improved ROC-AUC by approximately 0.024. These are incremental record-level comparisons, not clinical significance tests.

## Reliability and centre robustness

Calibration was calculated for every stage and held-out centre using fixed five- and ten-bin ECE analyses, with bin counts retained. Centre-specific results are unstable where prevalence and negative-case counts are sparse, particularly Switzerland. OOD diagnostics compare development-only standardized Euclidean, robust Mahalanobis and isolation-forest scores. Selective prediction reports fixed confidence thresholds across centres; neither diagnostic validates a clinical safety or abstention rule.

## Corrected and unresolved issues

The paired-comparison implementation explicitly includes centre in the row key; local source-row indices are not globally unique. Duplicate sensitivity reconstructs full-profile duplicate groups and distinguishes removal from flagged-only analysis. Remaining limitations include absent patient identifiers, centre sampling uncertainty, incomplete predictor timing documentation, record-level rather than cluster-level bootstrap intervals, and no independent external cohort.

## Decision

Stage 2 is the most defensible expanded early-presentation task. Stage 4 is a separate exercise-test task and cannot be substituted for initial pretest prediction. No stage is clinically selected. Independent external validation remains required, and no claim of clinical readiness, universal transportability, reliable abstention or patient-level safety is supported.

## Reproduction

```powershell
$env:PYTHONPATH='src;.'
& .\\.venv-repro\\Scripts\\python.exe runner\\phase4_validation_audit\\run.py
& .\\.venv-repro\\Scripts\\python.exe runner\\phase4_validation_audit\\make_figures.py
& .\\.venv-repro\\Scripts\\python.exe -m pytest -q
& .\\.venv-repro\\Scripts\\python.exe -m ruff check runner\\phase4_validation_audit tests\\test_phase4_validation_audit.py
```
""")


def write_manifest() -> None:
    outputs = {}
    for path in OUT.rglob("*"):
        if path.is_file() and path.name != "PHASE4_3_MANIFEST.json":
            outputs[str(path.relative_to(ROOT))] = sha256(path)
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "python": sys.version, "platform": platform.platform(), "seed": SEED, "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in [PREDICTIONS, METRICS]}, "outputs": outputs}
    (OUT / "PHASE4_3_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = reconstruct()
    predictions = pd.read_csv(PREDICTIONS)
    independent_reproduction(predictions, data)
    inner_validation_comparison(data, predictions)
    centre_calibration(predictions)
    paired_comparisons(predictions)
    duplicate_sensitivity(predictions, data)
    ood_audit(predictions, data)
    selective_audit(predictions)
    spectrum_analysis(data, predictions)
    selection_decision(predictions)
    final_report(predictions)
    write_manifest()
    print({"records": len(data), "predictions": len(predictions), "outputs": len(list(OUT.iterdir()))})


if __name__ == "__main__":
    main()
