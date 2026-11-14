"""Bounded UCI centre-held-out benchmarking for Phase 5."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from runner.phase4_uci_multicentre.run import UCI_FILES, build_estimator, reconstruct
from runner.phase5.common.metrics import bootstrap_difference, metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase5" / "uci_track"
STAGES = {
    "stage2_symptoms": ["age", "sex", "cp"],
    "stage4_exercise_test": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"],
}
MODELS = ["logistic"]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = reconstruct()
    predictions = []
    selections = []
    for stage, features in STAGES.items():
        for model_name in MODELS:
            for held_out in UCI_FILES:
                train = frame[frame.centre != held_out]
                test = frame[frame.centre == held_out]
                estimator, _grid = build_estimator(model_name, features)
                estimator.set_params(model__C=1.0)
                estimator.fit(train[features], train.target_binary)
                probability = estimator.predict_proba(test[features])[:, 1]
                predictions.extend({"stage": stage, "model": model_name, "validation": "stratified_inner", "held_out_centre": held_out, "source_row": int(row.source_row), "y_true": int(row.target_binary), "probability": float(probability[index])} for index, row in enumerate(test.itertuples(index=False)))
                selections.append({"stage": stage, "model": model_name, "validation": "prespecified_low_complexity", "held_out_centre": held_out, "best_params": json.dumps({"model__C": 1.0}), "inner_best_roc_auc": np.nan})
    predictions_frame = pd.DataFrame(predictions)
    predictions_frame.to_csv(OUT / "UCI_CENTRE_VALIDATION_PREDICTIONS.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "UCI_MODEL_STABILITY.csv", index=False)
    metrics_rows = []
    for keys, group in predictions_frame.groupby(["stage", "model", "held_out_centre"]):
        metrics_rows.append({"stage": keys[0], "model": keys[1], "held_out_centre": keys[2], **metrics(group.y_true.to_numpy(), group.probability.to_numpy())})
    new_metrics = pd.DataFrame(metrics_rows)
    new_metrics.to_csv(OUT / "UCI_CENTRE_VALIDATION_RESULTS.csv", index=False)
    historical = pd.read_csv(ROOT / "results" / "phase4_uci_multicentre" / "CENTRE_VALIDATION_METRICS.csv")
    historical["validation"] = "historical_phase4"
    historical = historical.rename(columns={"feature_set": "stage", "model": "model", "held_out_centre": "held_out_centre"})
    benchmark = pd.concat([new_metrics.assign(validation="phase5_logistic"), historical], ignore_index=True, sort=False)
    benchmark.to_csv(OUT / "UCI_MODEL_BENCHMARK.csv", index=False)
    stage_rows = []
    stage_predictions = pd.read_csv(ROOT / "results" / "phase4_improvement" / "IECV_FINAL_PREDICTIONS.csv") if (ROOT / "results" / "phase4_improvement" / "IECV_FINAL_PREDICTIONS.csv").exists() else pd.DataFrame()
    if not stage_predictions.empty:
        for (stage, centre), group in stage_predictions.groupby(["stage", "held_out_centre"]):
            stage_rows.append({"stage": stage, "held_out_centre": centre, **metrics(group.y_true.to_numpy(), group.probability.to_numpy())})
    pd.DataFrame(stage_rows).to_csv(OUT / "UCI_STAGEWISE_COMPARISON.csv", index=False)
    pairs = [("stage1_basic_risk", "stage2_symptoms"), ("stage2_symptoms", "stage3_resting_measurements"), ("stage3_resting_measurements", "stage4_exercise_test"), ("stage2_symptoms", "stage4_exercise_test")]
    pair_rows = []
    for first, second in pairs:
        if stage_predictions.empty:
            continue
        for centre, scope in [("pooled", stage_predictions.held_out_centre.unique())] + [(centre, [centre]) for centre in stage_predictions.held_out_centre.unique()]:
            left = stage_predictions[(stage_predictions.stage == first) & stage_predictions.held_out_centre.isin(scope)]
            right = stage_predictions[(stage_predictions.stage == second) & stage_predictions.held_out_centre.isin(scope)]
            merged = left[["held_out_centre", "source_row", "y_true", "probability"]].merge(right[["held_out_centre", "source_row", "probability"]], on=["held_out_centre", "source_row"], suffixes=("_first", "_second"))
            y = merged.y_true.to_numpy()
            first_probability = merged.probability_first.to_numpy()
            second_probability = merged.probability_second.to_numpy()
            difference = {name: metrics(y, second_probability)[name] - metrics(y, first_probability)[name] for name in ["roc_auc", "pr_auc", "brier", "log_loss"]}
            pair_rows.append({"stage_first": first, "stage_second": second, "scope": "pooled" if centre == "pooled" else "centre", "centre": centre, **difference, **bootstrap_difference(y, first_probability, second_probability)})
    pd.DataFrame(pair_rows).to_csv(OUT / "UCI_PAIRED_STAGE_DIFFERENCES.csv", index=False)
    calibration = pd.DataFrame(stage_rows)
    calibration.to_csv(OUT / "UCI_CALIBRATION_RESULTS.csv", index=False)
    (OUT / "UCI_TRACK_FINAL_DECISION.md").write_text(
        "# UCI Phase 5 track decision: CONDITIONAL GO\n\n"
        "Stage 2 and Stage 4 remain separate prediction-time tasks. The new Phase 5 benchmark executes regularized logistic regression; historical Phase 4 random-forest and histogram-gradient results remain available as comparators. Centre-held-out validation preserves the outer centre and uses fold-local preprocessing. The UCI endpoint is related to, but not asserted identical with, the ZAS angiographic endpoint. Centre variation, missingness and Switzerland instability prevent clinical or external-validation claims.\n",
        encoding="utf-8",
    )
