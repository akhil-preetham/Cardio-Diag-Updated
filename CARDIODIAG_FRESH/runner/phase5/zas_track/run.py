"""Bounded nested model benchmarking for the Z-Alizadeh Sani cohort."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from runner.phase5.common.metrics import metrics
from runner.phase5.common.models import estimator

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase5" / "zas_track"
SEED = 5107
GROUPS = {
    "v3_five_predictor": ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"],
    "basic_risk": ["Age", "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH"],
    "symptoms_examination": ["Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "BP", "PR", "Edema"],
    "resting_ecg": ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"],
    "laboratory": ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
    "eligible_combined": ["Age", "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "BP", "PR", "Edema", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
}
MODELS = ["logistic", "ridge", "elastic_net", "spline_logistic", "random_forest", "hist_gradient_boosting"]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv")
    frame["target_binary"] = frame["Cath"].map({"Cad": 1, "Normal": 0})
    predictions = []
    selections = []
    stability = []
    outer = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=SEED)
    for group_name, features in GROUPS.items():
        for model_name in MODELS:
            for fold_number, (train_index, test_index) in enumerate(outer.split(frame[features], frame.target_binary)):
                train = frame.iloc[train_index]
                test = frame.iloc[test_index]
                model, grid = estimator(model_name, features, frame)
                search = GridSearchCV(model, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED + fold_number), scoring="roc_auc", n_jobs=1)
                search.fit(train[features], train.target_binary)
                probability = search.predict_proba(test[features])[:, 1]
                predictions.extend({"group": group_name, "model": model_name, "repeat_fold": fold_number, "row_index": int(row_index), "y_true": int(y), "probability": float(probability[index])} for index, (row_index, y) in enumerate(zip(test.index, test.target_binary)))
                selections.append({"group": group_name, "model": model_name, "repeat_fold": fold_number, "best_params": json.dumps(search.best_params_, sort_keys=True), "inner_best_roc_auc": search.best_score_})
    predictions_frame = pd.DataFrame(predictions)
    predictions_frame.to_csv(OUT / "ZAS_NESTED_VALIDATION_PREDICTIONS.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "ZAS_HYPERPARAMETER_SELECTION.csv", index=False)
    aggregated = predictions_frame.groupby(["group", "model", "row_index"], as_index=False).agg(y_true=("y_true", "first"), probability=("probability", "mean"))
    rows = []
    for (group_name, model_name), group in predictions_frame.groupby(["group", "model"]):
        independent = aggregated[(aggregated.group == group_name) & (aggregated.model == model_name)]
        rows.append({"feature_group": group_name, "model": model_name, **metrics(independent.y_true.to_numpy(), independent.probability.to_numpy())})
        stability.append({"feature_group": group_name, "model": model_name, "unique_parameter_sets": int(pd.DataFrame([json.loads(item) for item in pd.DataFrame(selections).query("group == @group_name and model == @model_name").best_params]).drop_duplicates().shape[0]), "outer_folds": group.repeat_fold.nunique()})
    pd.DataFrame(rows).to_csv(OUT / "ZAS_MODEL_BENCHMARK.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "ZAS_FEATURE_STABILITY.csv", index=False)
    matched = []
    for name, features in GROUPS.items():
        complete = frame[features].notna().all(axis=1)
        matched.append({"feature_group": name, "n_complete": int(complete.sum()), "events": int(frame.loc[complete, "target_binary"].sum()), "non_events": int((complete & frame.target_binary.eq(0)).sum())})
    pd.DataFrame(matched).to_csv(OUT / "ZAS_FEATURE_GROUP_RESULTS.csv", index=False)
    calibration = aggregated.groupby(["group", "model"]).apply(lambda group: pd.Series(metrics(group.y_true.to_numpy(), group.probability.to_numpy())), include_groups=False).reset_index()
    calibration.to_csv(OUT / "ZAS_CALIBRATION_RESULTS.csv", index=False)
    pd.DataFrame([{"analysis": "fixed_prediction_bootstrap", "implemented": True, "limitation": "not complete model-development uncertainty"}, {"analysis": "optimism_correction", "implemented": False, "limitation": "not executed because nested predictions are already out-of-fold and a full bootstrap of model development would be computationally disproportionate"}]).to_csv(OUT / "ZAS_OPTIMISM_ANALYSIS.csv", index=False)
    (OUT / "ZAS_TRACK_FINAL_DECISION.md").write_text(
        "# ZAS Phase 5 track decision: CONDITIONAL GO\n\n"
        "The frozen V3 five-predictor track is preserved. Feature-group and bounded model comparisons are out-of-fold and fold-local. Echo/diagnostic timing remains uncertain and those fields are excluded from the eligible combined group. Optional GAM, EBM and Bayesian models were not executed because their packages are unavailable. No candidate is selected solely by pooled ROC-AUC; calibration, feature timing, and stability remain required.\n",
        encoding="utf-8",
    )
