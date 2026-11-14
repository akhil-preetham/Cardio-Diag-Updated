"""Reconcile saved model predictions and historical manifests without model fitting."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from runner.phase4_uci_multicentre.run import UCI_COLUMNS, reconstruct
from runner.phase5.common.metrics import metrics

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "final_project"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    checks: dict[str, bool] = {}
    for relative, expected in manifest.get("outputs", {}).items():
        artifact = ROOT / Path(relative.replace("\\", "/"))
        checks[relative] = artifact.is_file() and sha256(artifact) == expected
    return {
        "manifest": str(path.relative_to(ROOT)),
        "declared_outputs": len(checks),
        "present_and_hash_matched": sum(checks.values()),
        "mismatches": [name for name, matched in checks.items() if not matched],
    }


def reconcile_zas() -> dict[str, object]:
    base = ROOT / "results" / "phase5" / "zas_track"
    predictions = pd.read_csv(base / "ZAS_NESTED_VALIDATION_PREDICTIONS.csv")
    benchmark = pd.read_csv(base / "ZAS_MODEL_BENCHMARK.csv")
    selections = pd.read_csv(base / "ZAS_HYPERPARAMETER_SELECTION.csv")
    source = pd.read_csv(ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv")
    source_target = source["Cath"].map({"Cad": 1, "Normal": 0})
    candidate_counts: list[dict[str, object]] = []
    for (group, model), rows in predictions.groupby(["group", "model"]):
        aggregated = rows.groupby("row_index", as_index=False).agg(
            y_true=("y_true", "first"),
            probability=("probability", "mean"),
            label_count=("y_true", "nunique"),
            prediction_count=("repeat_fold", "size"),
        )
        if aggregated["row_index"].min() < 0 or aggregated["row_index"].max() >= len(source):
            raise ValueError(f"Out-of-range source-row index in {(group, model)}")
        expected_y = source_target.iloc[aggregated["row_index"].to_numpy()].to_numpy()
        if not np.array_equal(aggregated["y_true"].to_numpy(), expected_y):
            raise ValueError(f"Prediction labels disagree with source target in {(group, model)}")
        candidate_counts.append(
            {
                "group": group,
                "model": model,
                "prediction_rows": len(rows),
                "source_rows": aggregated["row_index"].nunique(),
                "repeats_per_row": sorted(aggregated["prediction_count"].unique().tolist()),
                "label_conflicts": int((aggregated["label_count"] > 1).sum()),
            }
        )
        if group == "eligible_combined" and model in {"logistic", "ridge"}:
            values = metrics(aggregated["y_true"].to_numpy(), aggregated["probability"].to_numpy())
            stored = benchmark.loc[
                (benchmark["feature_group"] == group) & (benchmark["model"] == model)
            ].iloc[0]
            candidate_counts[-1]["recomputed_metrics"] = values
            candidate_counts[-1]["benchmark_matches"] = all(
                np.isclose(values[key], stored[key], atol=1e-12)
                for key in ("roc_auc", "pr_auc", "brier", "log_loss")
            )
    alias = predictions[
        (predictions["group"] == "eligible_combined")
        & predictions["model"].isin(["logistic", "ridge"])
    ].pivot(index=["repeat_fold", "row_index"], columns="model", values="probability")
    if alias.isna().any().any():
        raise ValueError("Logistic/Ridge alias keys are incomplete")
    summary = {
        "benchmark_rows": len(benchmark),
        "selection_rows": len(selections),
        "unique_prediction_keys": int(
            predictions[["group", "model", "repeat_fold", "row_index"]]
            .drop_duplicates()
            .shape[0]
        ),
        "prediction_rows": len(predictions),
        "candidate_count_checks": candidate_counts,
        "logistic_ridge_max_probability_difference": float(
            (alias["logistic"] - alias["ridge"]).abs().max()
        ),
    }
    pd.DataFrame(candidate_counts).to_csv(FINAL / "PHASE5_ZAS_PREDICTION_RECONCILIATION.csv", index=False)
    return summary


def corrected_duplicate_sensitivity() -> dict[str, object]:
    data = reconstruct()
    profile = data[UCI_COLUMNS].astype("string").fillna("<MISSING>").agg("|".join, axis=1)
    group_size = profile.groupby(profile).transform("size")
    occurrence = profile.groupby(profile).cumcount()
    data = data.assign(profile=profile, group_size=group_size, occurrence=occurrence)
    exclusions = set(
        zip(
            data.loc[(group_size > 1) & (occurrence > 0), "centre"],
            data.loc[(group_size > 1) & (occurrence > 0), "source_row"],
            strict=True,
        )
    )
    predictions = pd.read_csv(
        ROOT / "results" / "phase4_improvement" / "IECV_FINAL_PREDICTIONS.csv"
    )
    output: list[dict[str, object]] = []
    for stage, rows in predictions.groupby("stage"):
        for analysis, subset in [
            ("all_records", rows),
            (
                "exclude_later_exact_duplicate_within_centre",
                rows[
                    ~rows.apply(
                        lambda row: (row["held_out_centre"], row["source_row"]) in exclusions,
                        axis=1,
                    )
                ],
            ),
        ]:
            scopes = [("pooled", subset)] + list(subset.groupby("held_out_centre"))
            for centre, part in scopes:
                y = part["y_true"].to_numpy(dtype=int)
                probability = part["probability"].to_numpy(dtype=float)
                result = metrics(y, probability)
                output.append(
                    {
                        "stage": stage,
                        "analysis": analysis,
                        "scope": centre,
                        "n": len(part),
                        "events": int(y.sum()),
                        "excluded_exact_duplicate_rows": len(rows) - len(part),
                        **result,
                    }
                )
    result_path = FINAL / "UCI_DUPLICATE_SENSITIVITY_CORRECTED.csv"
    pd.DataFrame(output).to_csv(result_path, index=False)
    duplicate_records = data.loc[
        (group_size > 1), ["profile", "centre", "source_row", "target_binary", "occurrence"]
    ]
    duplicate_records.to_csv(FINAL / "UCI_DUPLICATE_ROWS_RECONSTRUCTED.csv", index=False)
    return {
        "duplicate_groups": int(duplicate_records["profile"].nunique()),
        "later_duplicate_keys": sorted([[centre, int(row)] for centre, row in exclusions]),
        "corrected_pooled_rows_after_exclusion": int(
            len(predictions.drop_duplicates(subset=["held_out_centre", "source_row"]))
            - len(exclusions)
        ),
        "corrected_metrics_path": str(result_path.relative_to(ROOT)),
    }


def syntax_and_input_integrity() -> dict[str, object]:
    python_files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("runner/**/*.py"), *ROOT.glob("tests/**/*.py")]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    required_csv = [
        ROOT / "results" / "phase5" / "zas_track" / "ZAS_NESTED_VALIDATION_PREDICTIONS.csv",
        ROOT / "results" / "phase5" / "zas_track" / "ZAS_MODEL_BENCHMARK.csv",
        ROOT / "results" / "phase4_improvement" / "IECV_FINAL_PREDICTIONS.csv",
        ROOT / "results" / "phase4_validation_audit" / "PAIRED_STAGE_COMPARISONS.csv",
    ]
    csv_counts: dict[str, int] = {}
    for path in required_csv:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            csv_counts[str(path.relative_to(ROOT))] = sum(1 for _ in csv.reader(stream)) - 1
    manifests = [
        ROOT / "results" / "phase5" / "manifests" / "PHASE5_MANIFEST.json",
        ROOT / "results" / "phase4_uci_multicentre" / "PHASE4_REPRODUCIBILITY_MANIFEST.json",
        ROOT / "results" / "phase4_validation_audit" / "PHASE4_3_MANIFEST.json",
    ]
    return {
        "python_source_files_parsed": len(python_files),
        "required_csv_data_rows": csv_counts,
        "manifests": [verify_manifest(path) for path in manifests],
    }


def run() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    report = {
        "python": platform.python_version(),
        "model_fitting_performed": False,
        "zas": reconcile_zas(),
        "uci_duplicate_sensitivity": corrected_duplicate_sensitivity(),
        "integrity": syntax_and_input_integrity(),
    }
    (FINAL / "SAVED_ARTIFACTS_QA.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    run()
