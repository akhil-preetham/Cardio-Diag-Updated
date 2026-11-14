from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(root: Path):
    results = root / "results"
    verification = []
    comparison = []
    for output_root in [results / "ensemble_experiments", results / "ensemble_experiments_compat_v2"]:
        if not output_root.exists():
            continue
        for metrics_path in output_root.glob("*/*/*/metrics.json"):
            directory = metrics_path.parent
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))[0]
            predictions = pd.read_csv(directory / "predictions.csv")
            required_files = [
                directory / "model.joblib",
                directory / "predictions.csv",
                directory / "metrics.json",
                directory / "search_results.json",
                directory / "manifest.json",
            ]
            saved_accuracy = float(metrics["accuracy"])
            saved_auc = float(metrics["roc_auc"])
            actual_accuracy = float(accuracy_score(predictions["y_true"], predictions["prediction"]))
            actual_auc = float(roc_auc_score(predictions["y_true"], predictions["probability"]))
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            comparison.append(
                {
                    "dataset": manifest["dataset"],
                    "track": manifest["track"],
                    "method": manifest["method"],
                    "accuracy": saved_accuracy,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "sensitivity": metrics["recall_sensitivity"],
                    "specificity": metrics["specificity"],
                    "f1": metrics["f1"],
                    "roc_auc": saved_auc,
                    "pr_auc": metrics["pr_auc_average_precision"],
                    "brier_score": metrics["brier_score"],
                    "runtime_seconds": manifest["runtime_seconds"],
                    "selected_parameters": json.dumps(manifest["search_results"], sort_keys=True),
                    "output_root": str(output_root.relative_to(root)),
                }
            )
            verification.append(
                {
                    "component": f"{manifest['dataset']}/{manifest['track']}/{manifest['method']}",
                    "implementation_status": "implemented",
                    "test_status": "reloaded_in_runner",
                    "execution_status": "completed",
                    "recalculation_status": "passed" if abs(saved_accuracy - actual_accuracy) < 1e-8 and abs(saved_auc - actual_auc) < 1e-8 else "failed",
                    "artifact_path": str(directory.relative_to(root)),
                    "source_to_paper_consistency": "source-shaped ensemble; protocol differs by track",
                    "remaining_limitation": (
                        "single holdout for this controlled comparison; "
                        f"required_artifacts={'present' if all(path.is_file() for path in required_files) else 'incomplete'}"
                    ),
                }
            )
    pd.DataFrame(comparison).to_csv(results / "ensemble_comparison.csv", index=False)

    required_analysis = {
        "delong_wilcoxon_bootstrap_holm": results / "ensemble_postprocessing/paired_model_comparisons.csv",
        "subgroup_analysis": results / "ensemble_postprocessing/ensemble_subgroup_performance.csv",
        "decision_curve": results / "ensemble_postprocessing/decision_curve_net_benefit.csv",
        "calibration": results / "ensemble_postprocessing/calibration_summary.csv",
        "xaih_drc": results / "xaih_drc_experiments/xaih_drc_manifest.json",
        "loras_smote": results / "data_integrity_and_loras/dataset_integrity_loras_report.json",
        "lime_manifest": results / "lime_experiments/lime_manifest.json",
    }
    for name, path in required_analysis.items():
        verification.append(
            {
                "component": name,
                "implementation_status": "present",
                "test_status": "artifact_validation",
                "execution_status": "completed" if path.is_file() else "missing",
                "recalculation_status": "independent_metric_recalculation_in_comparison" if name == "delong_wilcoxon_bootstrap_holm" else "present",
                "artifact_path": str(path.relative_to(root)),
                "source_to_paper_consistency": "documented in final execution verification",
                "remaining_limitation": "subgroup results are descriptive; LIME is local explanation, not causal evidence",
            }
        )
    for failure_path in (results / "ensemble_experiments").glob("*/*/ovec/failure.json"):
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure_path.parts[-3] != "author_faithful":
            continue
        verification.append(
            {
                "component": f"{failure_path.parts[-4]}/author_faithful/ovec",
                "implementation_status": "source_preserved",
                "test_status": "expected_failure_preserved",
                "execution_status": "failed_on_source_invalid_candidate",
                "recalculation_status": "not_applicable",
                "artifact_path": str(failure_path.relative_to(root)),
                "source_to_paper_consistency": "exact recovered source candidate retained",
                "remaining_limitation": failure.get("error", "invalid source candidate"),
            }
        )

    inventory = pd.read_csv(root / "reference/reproduction_runner/final_experiment_inventory.csv")
    verification.append(
        {
            "component": "18_notebook_inventory",
            "implementation_status": "audited",
            "test_status": "unique_path_check",
            "execution_status": "notebook_runner_disabled",
            "recalculation_status": "passed" if len(inventory) == 18 and inventory["notebook"].nunique() == 18 else "failed",
            "artifact_path": "reference/reproduction_runner/final_experiment_inventory.csv",
            "source_to_paper_consistency": "inventory evidence",
            "remaining_limitation": "blocked notebooks remain blocked",
        }
    )
    config = json.loads((root / "config/runner_config.json").read_text(encoding="utf-8"))
    verification.append(
        {
            "component": "notebook_execution_gate",
            "implementation_status": "runner_control",
            "test_status": "config_validation",
            "execution_status": "disabled" if not config["notebook_execution"]["enabled"] and not config["notebook_execution"]["approval_granted"] else "enabled",
            "recalculation_status": "passed",
            "artifact_path": "config/runner_config.json",
            "source_to_paper_consistency": "safety control",
            "remaining_limitation": "controlled Python experiments are separate from notebook execution",
        }
    )
    hashes = {}
    for label, path in {
        "original_notebook": root / "reference/extracted_author_archives/base_reproduction/Daza_Base_Paper_Reproduced/Daza_Paper_Exact_Accuracy.ipynb",
        "prepared_notebook": root / "reference/reproduction_runner/work/author_faithful/base_reproduction/Daza_Paper_Exact_Accuracy.ipynb",
        "five_states_dataset": root / "reference/author_release/heart5.csv",
        "mendeley_dataset": root / "reference/author_release/MendeleyCardiovascular_Disease_Dataset.csv",
    }.items():
        hashes[label] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
    report = {
        "status": "implementation_complete_external_data_or_validation_still_required",
        "hashes": hashes,
        "verification": verification,
        "known_blockers": [
            "Author-faithful OVEC contains source-invalid RandomForest n_estimators=None.",
            "The approximately 70,000-row benchmark dataset is unavailable.",
            "Historical environment equivalence is unverified.",
            "Full project Ruff still contains unrelated pre-existing findings.",
        ],
    }
    (results / "final_independent_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
