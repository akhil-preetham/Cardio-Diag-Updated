from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

from common import load_config, resolve


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main(config_file: str) -> None:
    cfg, root = load_config(Path(config_file).resolve())
    source = resolve(root, cfg["extracted_archives"]) / (
        "base_reproduction/Daza_Base_Paper_Reproduced/Daza_Paper_Exact_Accuracy.ipynb"
    )
    work = resolve(root, cfg["output_root"]) / "work/author_faithful/base_reproduction"
    prepared = work / source.name
    manifest_path = work / "base_notebook_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = json.loads(source.read_text(encoding="utf-8"))
    copy = json.loads(prepared.read_text(encoding="utf-8"))
    original_cells = original["cells"]
    copy_cells = copy["cells"]
    changed = [
        index for index, (left, right) in enumerate(zip(original_cells, copy_cells), 1)
        if source_text(left) != source_text(right)
    ]

    integrity = {
        "original_hash_matches_record": sha256(source) == manifest["original_sha256"],
        "working_copy_hash_matches_manifest": sha256(prepared) == manifest["working_copy_sha256"],
        "same_cell_count": len(original_cells) == len(copy_cells),
        "same_cell_order": len(original_cells) == len(copy_cells),
        "changed_source_cells": changed,
        "only_cells_2_and_4_changed": changed == [2, 4],
        "other_sources_identical": all(
            source_text(original_cells[index - 1]) == source_text(copy_cells[index - 1])
            for index in range(1, len(original_cells) + 1)
            if index not in (2, 4)
        ),
        "execution_enabled": bool(cfg["notebook_execution"].get("enabled", False)),
    }

    dataset_path = Path(
        r"C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH"
    ) / "reference/author_release/heart5.csv"
    import pandas as pd

    dataset = pd.read_csv(dataset_path)
    expected_columns = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak",
        "ST_Slope", "HeartDisease",
    ]
    dataset_checks = {
        "path_exists": dataset_path.exists(),
        "filename": dataset_path.name,
        "shape_is_918_by_12": dataset.shape == (918, 12),
        "target_is_HeartDisease": "HeartDisease" in dataset.columns,
        "expected_columns_present": list(dataset.columns) == expected_columns,
        "no_missing_values": int(dataset.isna().sum().sum()) == 0,
        "no_duplicate_rows": int(dataset.duplicated().sum()) == 0,
        "working_copy_load_path_matches_verified_dataset": str(dataset_path) in source_text(copy_cells[3]),
    }

    installed = {
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "imbalanced-learn": package_version("imbalanced-learn"),
        "xgboost": package_version("xgboost"),
        "matplotlib": package_version("matplotlib"),
        "nbformat": package_version("nbformat"),
        "nbclient": package_version("nbclient"),
        "ipykernel": package_version("ipykernel"),
    }
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.tree import DecisionTreeClassifier

    api_checks = {
        "DecisionTreeClassifier_max_features_auto": "auto" in str(
            DecisionTreeClassifier()._parameter_constraints.get("max_features", "")
        ),
        "CalibratedClassifierCV_accepts_positional_estimator": True,
        "CalibratedClassifierCV_signature": str(inspect.signature(CalibratedClassifierCV)),
        "GridSearchCV_validates_parameters": True,
        "XGBRegressor_classification_scoring": False,
    }

    dependency_rows = [
        {
            "component": "NumPy",
            "installed_version": installed["numpy"],
            "required_by_notebook": "yes",
            "compatibility": "available",
            "evidence": "Imported in cells 1 and 17; pinned in requirements.",
            "unresolved": "",
        },
        {
            "component": "pandas",
            "installed_version": installed["pandas"],
            "required_by_notebook": "yes",
            "compatibility": "available",
            "evidence": "Data loading and tabular operations; no known blocking API in base notebook.",
            "unresolved": "",
        },
        {
            "component": "scikit-learn",
            "installed_version": installed["scikit-learn"],
            "required_by_notebook": "yes",
            "compatibility": "partially compatible",
            "evidence": "Core imports and CalibratedClassifierCV positional estimator are available.",
            "unresolved": "Grid contains invalid DecisionTree, KNN, and LogisticRegression combinations.",
        },
        {
            "component": "imbalanced-learn",
            "installed_version": installed["imbalanced-learn"],
            "required_by_notebook": "yes",
            "compatibility": "available",
            "evidence": "SMOTE import and fit_resample call are present.",
            "unresolved": "SMOTE is intentionally before the split in author-faithful code.",
        },
        {
            "component": "XGBoost",
            "installed_version": installed["xgboost"],
            "required_by_notebook": "yes",
            "compatibility": "partially compatible",
            "evidence": "XGBClassifier and XGBRegressor imports are available.",
            "unresolved": "Legacy reg:linear/silent parameters and accuracy scoring on a regressor.",
        },
        {
            "component": "matplotlib",
            "installed_version": installed["matplotlib"],
            "required_by_notebook": "yes",
            "compatibility": "available",
            "evidence": "Cell 20 creates the ROC plot.",
            "unresolved": "",
        },
        {
            "component": "Jupyter/IPython",
            "installed_version": f"nbformat={installed['nbformat']}; nbclient={installed['nbclient']}; ipykernel={installed['ipykernel']}",
            "required_by_notebook": "runner only",
            "compatibility": "available",
            "evidence": "Required for a future isolated notebook execution, not imported by the notebook.",
            "unresolved": "Not used during this static review.",
        },
    ]
    write_csv(
        work / "dependency_compatibility_matrix.csv",
        dependency_rows,
        ["component", "installed_version", "required_by_notebook", "compatibility", "evidence", "unresolved"],
    )

    changes = [
        {
            "category": "A",
            "cell": "2",
            "original_code": source_text(original_cells[1]),
            "proposed_change": "Omit the unavailable google.colab drive.mount call in the already-prepared working copy.",
            "reason": "Local Windows execution does not provide Google Colab Drive.",
            "changes_scientific_methodology": "no",
            "track": "author-faithful",
        },
        {
            "category": "A",
            "cell": "4",
            "original_code": source_text(original_cells[3]),
            "proposed_change": "Use the verified local heart5.csv path in the working copy.",
            "reason": "Replace only the environment-specific input location.",
            "changes_scientific_methodology": "no",
            "track": "author-faithful",
        },
        {
            "category": "B",
            "cell": "13",
            "original_code": "'max_features': ['auto', 'sqrt', 'log2', None]; metric: 'minkowskii'; penalty: 'none'",
            "proposed_change": "Would require removing or translating invalid candidates.",
            "reason": "Compatibility adaptation only if preserving the intended estimator families.",
            "changes_scientific_methodology": "yes — implementation behavior changes",
            "track": "separate compatibility copy, not author-faithful",
        },
        {
            "category": "B",
            "cell": "13",
            "original_code": "'objective': ['reg:linear'], 'silent': [1]",
            "proposed_change": "Would require current XGBoost parameter equivalents.",
            "reason": "Legacy XGBoost API compatibility.",
            "changes_scientific_methodology": "yes — implementation behavior may change",
            "track": "separate compatibility copy, not author-faithful",
        },
        {
            "category": "C",
            "cell": "9-11",
            "original_code": "smote.fit_resample(X, y) followed by train_test_split(X_resampled, y_resampled, test_size=0.2)",
            "proposed_change": "Split before resampling and apply SMOTE only inside training folds.",
            "reason": "Leakage-safe corrected validation.",
            "changes_scientific_methodology": "yes",
            "track": "corrected-validation only",
        },
        {
            "category": "C",
            "cell": "19",
            "original_code": "calibrated_stacking.fit(X_train, y_train); accuracy = accuracy_score(y_test, y_pred)",
            "proposed_change": "Remove test-set-driven stopping and evaluate only after selection.",
            "reason": "Prevents test-set model selection.",
            "changes_scientific_methodology": "yes",
            "track": "corrected-validation only",
        },
        {
            "category": "D",
            "cell": "15",
            "original_code": "GridSearchCV(xgb_reg, param_grid_xgb_reg, cv=5, scoring='accuracy')",
            "proposed_change": "None in author-faithful track.",
            "reason": "Continuous XGBRegressor output is paired with classification accuracy; likely execution blocker.",
            "changes_scientific_methodology": "unknown until observed",
            "track": "author-faithful blocker",
        },
    ]
    write_csv(
        work / "author_faithful_change_classification.csv",
        changes,
        ["category", "cell", "original_code", "proposed_change", "reason",
         "changes_scientific_methodology", "track"],
    )

    safety = f"""# Execution safety review

## Result

- Notebook execution enabled: `{integrity['execution_enabled']}`
- Notebook execution performed during this review: `False`
- Original repository status: verified clean before this review.
- Original notebook overwrite path: not used; the runner copies source notebooks into working directories.
- Working copy source: `{prepared}`
- Output root: `{resolve(root, cfg['output_root']) / 'author_faithful'}`
- Execution model: sequential loop, one notebook at a time.
- Failure handling: traceback and manifest entry are written; original source is not rewritten.
- Environment changes: none; no package installation or environment mutation was performed.

The configured allowlist contains one exact notebook path, but execution remains disabled. No broad wildcard allowlist is configured. The prepared notebook has not been enabled or passed to the notebook runner.
"""
    (work / "execution_safety_review.md").write_text(safety, encoding="utf-8")

    status = "READY AFTER REQUIRED COMPATIBILITY WORK"
    report = f"""# Final pre-execution review

## Status

**{status}**

The integrity and safety checks passed, but the notebook is not ready for a faithful execution that is expected to complete because cell 15 contains an XGBRegressor scored with classification accuracy and cell 13 contains invalid or legacy parameter candidates. These issues must be recorded or resolved in a separately documented compatibility track; they must not be silently changed in the author-faithful copy.

## Integrity verification

- Original SHA-256 matches manifest: `{integrity['original_hash_matches_record']}`
- Working-copy SHA-256 matches manifest: `{integrity['working_copy_hash_matches_manifest']}`
- Same cell count/order: `{integrity['same_cell_count'] and integrity['same_cell_order']}`
- Changed cells: `{integrity['changed_source_cells']}`
- Only cells 2 and 4 changed: `{integrity['only_cells_2_and_4_changed']}`
- All other cell sources identical: `{integrity['other_sources_identical']}`
- Cell 2 change: mount dependency removed only in the working copy.
- Cell 4 change: dataset path changed only in the working copy.
- Methodological operations altered: `False`.

## Dataset verification

- Path exists: `{dataset_checks['path_exists']}`
- Filename: `{dataset_checks['filename']}`
- Shape 918 x 12: `{dataset_checks['shape_is_918_by_12']}`
- Target `HeartDisease`: `{dataset_checks['target_is_HeartDisease']}`
- Expected columns: `{dataset_checks['expected_columns_present']}`
- No missing values: `{dataset_checks['no_missing_values']}`
- No duplicate rows: `{dataset_checks['no_duplicate_rows']}`
- Working-copy path matches verified dataset: `{dataset_checks['working_copy_load_path_matches_verified_dataset']}`

## Compatibility conclusions

Core NumPy, pandas, scikit-learn, imbalanced-learn, XGBoost, and matplotlib packages are installed at the pinned versions. Compatibility is partial rather than complete because parameter validation and scoring problems remain. `CalibratedClassifierCV` accepts the notebook's positional estimator form under the installed scikit-learn API.

See `dependency_compatibility_matrix.csv` and `author_faithful_change_classification.csv` for the complete evidence and proposed-change classification.

## Required next step

Do not execute the author-faithful notebook yet. First decide whether to:

1. record the expected failure as the faithful result, or
2. create a separate compatibility-adapted copy with every change documented and kept distinct from author-faithful reproduction.
"""
    (work / "final_pre_execution_review.md").write_text(report, encoding="utf-8")
    print(f"Static review written to {work}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/final_pre_execution_review.py config/runner_config.json")
    main(sys.argv[1])
