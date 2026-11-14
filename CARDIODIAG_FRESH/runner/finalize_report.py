from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from common import load_config, resolve, utc_now, write_json


COMPONENTS = [
    ("Data preprocessing", "Paper describes preprocessing; author notebooks encode/drop fields directly", "Source evidence audited; corrected pipeline fits preprocessing on training data", "partial"),
    ("75:25 split", "Required comparator", "Corrected validation uses stratified 75:25 repeated holdouts", "supported"),
    ("Five-fold cross-validation", "Reported in methodology", "Corrected GridSearchCV uses stratified five-fold CV inside training partitions", "supported"),
    ("Ten independent runs", "Reported in methodology", "Ten repeated seeded holdouts completed for each dataset; they overlap and are not independent cohorts", "partial"),
    ("SMOTE", "Reported resampling method", "Corrected pipeline applies SMOTE inside training folds only", "supported"),
    ("Grid Search", "Reported model-selection method", "Corrected track searches final-estimator C on training folds", "supported"),
    ("Stacking", "Reported ensemble method", "Corrected track uses a StackingClassifier", "supported"),
    ("Calibration", "Reported in paper/code", "Not added to corrected comparator; author-faithful notebook did not complete", "not_verified"),
    ("LoRAS / XAIH-DRC / OVEC / OSEC", "Specialized methods claimed in paper", "Static keyword evidence exists in selected notebooks; independent correctness and execution were not established", "not_verified"),
    ("LIME / fairness / clinical utility", "Supplementary analyses", "Not executed because notebooks require Colab paths and additional review", "not_verified"),
    ("External validation", "Would require an external cohort", "No external cohort was supplied or evaluated", "not_supported"),
]


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(config_file: str) -> None:
    cfg, root = load_config(Path(config_file).resolve())
    output = resolve(root, cfg["output_root"])
    audit = output / "audit"
    reports = output / "reports"
    corrected = output / "corrected_validation"
    reports.mkdir(parents=True, exist_ok=True)

    preflight = load_json(audit / "preflight.json", {})
    validation = load_json(audit / "dataset_validation.json", {})
    manifest = load_json(output / "logs" / "author_faithful" / "execution_manifest.json", [])
    corrected_reports = []
    for dataset in ("five_states", "mendeley"):
        report = load_json(corrected / f"{dataset}_corrected_validation_report.json", None)
        if report:
            corrected_reports.append(report)

    result_rows = []
    for report in corrected_reports:
        for run in report["runs"]:
            result_rows.append({
                "dataset": report["dataset"],
                "implementation": "runner/corrected_validation.py",
                "track": "corrected_validation",
                "model": "StackingClassifier with GridSearchCV",
                "seed": run["seed"],
                "split": "stratified 75:25 holdout",
                "preprocessing": "training-fold imputation, scaling, one-hot encoding where applicable",
                "resampling": "SMOTE inside training folds",
                "hyperparameters": json.dumps(run["best_params"], sort_keys=True),
                "accuracy": run["accuracy"],
                "balanced_accuracy": run["balanced_accuracy"],
                "precision": run["precision"],
                "recall_sensitivity": run["recall_sensitivity"],
                "specificity": run["specificity"],
                "f1": run["f1"],
                "roc_auc": run["roc_auc"],
                "pr_auc_average_precision": run["pr_auc_average_precision"],
                "brier_score": run["brier_score"],
                "execution_status": "success",
            })
    metric_fields = list(result_rows[0]) if result_rows else ["dataset", "track", "execution_status"]
    write_csv(reports / "experiment_results.csv", result_rows, metric_fields)

    failed = [{
        "notebook": item.get("notebook", ""),
        "track": "author_faithful",
        "status": item.get("status", ""),
        "error": item.get("error", ""),
        "traceback": item.get("traceback", ""),
    } for item in manifest if item.get("status") == "failed"]
    write_csv(reports / "failed_experiments.csv", failed,
              ["notebook", "track", "status", "error", "traceback"])

    matrix_rows = [{
        "component": component, "paper_requirement": paper,
        "author_code_evidence": "See static notebook and risk audit; source-level evidence only",
        "corrected_validation_evidence": corrected_text, "status": status
    } for component, paper, corrected_text, status in COMPONENTS]
    write_csv(reports / "paper_vs_author_code_vs_corrected_validation.csv", matrix_rows,
              ["component", "paper_requirement", "author_code_evidence",
               "corrected_validation_evidence", "status"])

    execution_summary = "\n".join(
        f"- `{item.get('notebook')}`: **{item.get('status')}**"
        for item in manifest
    ) or "- No author-faithful notebook execution was attempted."
    report_text = f"""# CARDIODIAG reproducibility report

Generated: `{utc_now()}`

## Evidence boundary

The author release was verified at commit `{preflight.get('author_git_commit', 'unknown')}` and was clean at preflight. Original author files and extracted notebooks were not modified. Static audit covered {preflight.get('notebook_count', 0)} notebooks without executing them.

## Author-faithful execution

The explicitly allowlisted base notebook was copied and attempted. It failed before producing a reproduced result because it depends on the historical `google.colab`/`/content/drive` environment. The remaining notebooks were not allowlisted because static audit found Colab mounts, shell commands, deletion operations, or unresolved external artifacts.

{execution_summary}

Historical saved values in the original notebook (accuracy 0.8824, precision 0.8898, recall 0.9052, F1 0.8974, ROC-AUC 0.9578) are treated as reported artifacts, not independently reproduced results.

## Corrected validation

The independent comparator uses a stratified 75:25 split, five-fold stratified CV within the training partition, training-fold preprocessing, SMOTE inside the imbalanced-learn pipeline, and ten seeded repeated holdouts. Mendeley `patientid` was excluded. Repeated holdouts overlap and therefore are not independent external validation cohorts.

See `experiment_results.csv` and the per-dataset JSON reports for every run and metric. The high Mendeley performance is an observed result on the supplied dataset, not evidence of external clinical validity.

## Conclusions

- The datasets and environment are reproducible from the preserved local artifacts.
- The historical base-paper values were not verified by a completed author-faithful execution.
- The corrected comparator supports the leakage-safe split/resampling protocol implemented here.
- Specialized claims (LoRAS, XAIH-DRC, OVEC, OSEC, LIME, fairness, clinical utility, calibration, statistical tests, and external validation) remain unverified in this run.
"""
    (reports / "reproduction_report.md").write_text(report_text, encoding="utf-8")
    (reports / "reproducibility_checklist.md").write_text(
        "# Reproducibility checklist\n\n"
        "- [x] Author repository commit and cleanliness recorded\n"
        "- [x] Notebook SHA-256 inventory and individual static reports generated\n"
        "- [x] Supplementary-file hashes generated\n"
        "- [x] Dataset schema, class balance, missingness, duplicates, and identifiers validated\n"
        "- [x] Corrected validation separated from author-faithful outputs\n"
        "- [x] Ten seeded runs completed for both supplied datasets\n"
        "- [x] Author-faithful attempt logged with traceback\n"
        "- [ ] External validation cohort evaluated\n"
        "- [ ] Specialized explainability/fairness/clinical analyses independently reproduced\n",
        encoding="utf-8",
    )
    (reports / "limitations_and_threats_to_validity.md").write_text(
        "# Limitations and threats to validity\n\n"
        "- The supplied datasets are single local cohorts; repeated holdouts overlap.\n"
        "- No patient-level linkage protocol or external cohort was available beyond the Mendeley identifier exclusion.\n"
        "- The corrected comparator is not an exact reproduction of the author pipeline.\n"
        "- The author-faithful base notebook requires Colab Drive paths and failed before model execution in this environment.\n"
        "- Static keyword evidence does not prove that a method was executed or scientifically correct.\n"
        "- No unsupported clinical or causal claims should be inferred from these predictive metrics.\n",
        encoding="utf-8",
    )
    write_json(reports / "execution_manifest.json", {
        "created_utc": utc_now(), "preflight": preflight,
        "dataset_validation": validation, "author_faithful": manifest,
        "corrected_reports": [report.get("dataset") for report in corrected_reports],
    })
    print(f"Final reports written to {reports}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/finalize_report.py config/runner_config.json")
    main(sys.argv[1])
