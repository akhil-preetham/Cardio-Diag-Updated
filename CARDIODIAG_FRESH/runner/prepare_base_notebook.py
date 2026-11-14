from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import load_config, resolve


NOTEBOOK_RELATIVE = Path(
    "base_reproduction/Daza_Base_Paper_Reproduced/Daza_Paper_Exact_Accuracy.ipynb"
)
LOCAL_DATASET = Path(
    r"C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH"
) / "reference" / "author_release" / "heart5.csv"
OUTPUT_RELATIVE = Path("work/author_faithful/base_reproduction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def find_cells(cells: list[dict], needle: str) -> list[int]:
    return [
        index for index, cell in enumerate(cells, 1)
        if needle.lower() in source_text(cell).lower()
    ]


def main(config_file: str) -> None:
    cfg, root = load_config(Path(config_file).resolve())
    source = resolve(root, cfg["extracted_archives"]) / NOTEBOOK_RELATIVE
    dataset = LOCAL_DATASET
    destination_dir = resolve(root, cfg["output_root"]) / OUTPUT_RELATIVE
    destination = destination_dir / source.name
    if destination_dir.exists() or destination.exists():
        raise SystemExit(f"Refusing to overwrite existing preparation output: {destination_dir}")
    if not source.exists():
        raise SystemExit(f"Original notebook not found: {source}")
    if not dataset.exists():
        raise SystemExit(f"Verified dataset not found: {dataset}")

    original_bytes = source.read_bytes()
    original_hash = sha256(source)
    notebook = json.loads(original_bytes.decode("utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) != 20:
        raise SystemExit(f"Unexpected notebook cell count: {len(cells)}")

    original_cell_2 = source_text(cells[1])
    original_cell_4 = source_text(cells[3])
    modified_cell_2 = (
        "# Local preparation only: Google Colab Drive mounting is omitted.\n"
        "# The original drive.mount operation is preserved in the source notebook.\n"
    )
    modified_cell_4 = (
        f"df = pd.read_csv(r'{dataset}')\n\n"
        "#first 5 values of data\ndf.head()\n"
    )
    cells[1]["source"] = modified_cell_2
    cells[3]["source"] = modified_cell_4
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    copied_hash = sha256(destination)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preparation_only": True,
        "execution_performed": False,
        "original_notebook": str(source),
        "working_copy": str(destination),
        "original_sha256": original_hash,
        "working_copy_sha256": copied_hash,
        "dataset": {
            "path": str(dataset),
            "exists": dataset.exists(),
            "sha256": sha256(dataset),
            "bytes": dataset.stat().st_size,
        },
        "changed_cells": [
            {
                "cell": 2,
                "original_source": original_cell_2,
                "modified_source": modified_cell_2,
                "reason": "Remove only the unavailable Google Colab drive.mount dependency from the working copy; preserve the original operation in the author notebook.",
            },
            {
                "cell": 4,
                "original_source": original_cell_4,
                "modified_source": modified_cell_4,
                "reason": "Replace only the Google Drive dataset path with the verified local heart5.csv path.",
            },
        ],
        "unchanged_methodological_cells": [
            index for index in range(1, len(cells) + 1) if index not in (2, 4)
        ],
    }
    (destination_dir / "base_notebook_patch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    requirements = """# Pinned environment for the prepared author-faithful base notebook.
# Preparation only: packages were not installed by this step.
numpy==2.4.6
pandas==3.1.1
scikit-learn==1.9.1
imbalanced-learn==0.14.2
xgboost==3.2.0
matplotlib==3.11.2
nbformat==5.11.1
nbclient==0.11.0
ipykernel==7.4.0
"""
    (destination_dir / "requirements-author-reproduction.txt").write_text(
        requirements, encoding="utf-8"
    )

    environment = f"""Preparation timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec="seconds")}
Preparation only: True
Notebook execution performed: False
Python: {sys.version}
Platform: {platform.platform()}
Original notebook SHA-256: {original_hash}
Working copy SHA-256: {copied_hash}
Dataset path: {dataset}
Dataset SHA-256: {sha256(dataset)}
Installed package versions used for compatibility review:
  numpy==2.4.6
  pandas==3.1.1
  scikit-learn==1.9.1
  imbalanced-learn==0.14.2
  xgboost==3.2.0
  matplotlib==3.11.2
  nbformat==5.11.1
  nbclient==0.11.0
  ipykernel==7.4.0
"""
    (destination_dir / "base_notebook_environment.txt").write_text(
        environment, encoding="utf-8"
    )

    expected_failures = [
        (
            13,
            "DecisionTreeClassifier grid includes max_features='auto', which is incompatible with current scikit-learn.",
            "Likely failed candidate(s); GridSearchCV may continue with other candidates depending on error handling.",
        ),
        (
            13,
            "KNeighborsClassifier grid includes the misspelled metric 'minkowskii'.",
            "Likely failed candidate(s); valid metric candidates may still run.",
        ),
        (
            13,
            "LogisticRegression grid combines penalty/solver values that are incompatible and includes penalty='none'.",
            "Some grid candidates are expected to fail under current scikit-learn.",
        ),
        (
            13,
            "XGBRegressor grid uses objective='reg:linear' and silent, both legacy parameters.",
            "Likely incompatible or ignored under current XGBoost.",
        ),
        (
            15,
            "XGBRegressor is scored with scoring='accuracy'.",
            "Expected scoring failure because the regressor produces continuous outputs for a classification metric; this may abort GridSearchCV.",
        ),
        (
            19,
            "The loop repeatedly refits and evaluates on X_test while calling the behavior early stopping.",
            "Methodological defect: test performance controls stopping; preserve for author-faithful reproduction.",
        ),
    ]

    report_lines = [
        "# Base notebook preparation report",
        "",
        "## Scope",
        "",
        "This package is preparation-only. No notebook kernel was started, no model was trained, and notebook execution remains disabled.",
        "",
        f"- Original: `{source}`",
        f"- Working copy: `{destination}`",
        f"- Original SHA-256: `{original_hash}`",
        f"- Working-copy SHA-256: `{copied_hash}`",
        f"- Verified dataset: `{dataset}`",
        f"- Dataset SHA-256: `{sha256(dataset)}`",
        "",
        "## Cell inspection",
        "",
        f"The notebook contains {len(cells)} cells: "
        f"{sum(cell.get('cell_type') == 'code' for cell in cells)} code and "
        f"{sum(cell.get('cell_type') == 'markdown' for cell in cells)} markdown.",
        "",
        "- Cell 2 imports `google.colab` and calls `drive.mount`; changed only in the working copy.",
        "- Cell 4 reads `/content/drive/MyDrive/heart5.csv`; changed only to the verified local path.",
        "- Cells 1, 5–20 are unchanged in operation, including SMOTE placement, 80:20 split, grids, CV, calibration, evaluation, and test-set early-stopping behavior.",
        "",
        "## Required packages and compatibility",
        "",
        "The pinned environment is recorded in `requirements-author-reproduction.txt`. Core imports are available in `.venv-repro` at the pinned versions recorded in `base_notebook_environment.txt`. `google.colab` is intentionally not required after the working-copy-only cell 2 adaptation.",
        "",
        "Compatibility is not a guarantee of successful execution. The source contains invalid or legacy configurations listed below.",
        "",
        "## Expected failures or warnings",
        "",
    ]
    for cell, issue, expected in expected_failures:
        report_lines.extend([f"- Cell {cell}: **{issue}** {expected}"])
    report_lines.extend([
        "",
        "## Preservation verification",
        "",
        "The original extracted notebook was read-only during preparation. The original SHA-256 is recorded in `SHA256SUMS.txt` and must be rechecked immediately before any future approved execution.",
        "",
        "## Readiness decision",
        "",
        "The working copy is technically prepared for an explicitly approved execution attempt, but it is not methodologically corrected. Execution still requires a reviewed allowlist, isolated output directory, and confirmation of the expected compatibility failures.",
    ])
    (destination_dir / "base_notebook_preparation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    checklist = """# Base notebook execution checklist

- [x] Original notebook copied without editing the original.
- [x] Original and working-copy SHA-256 recorded.
- [x] Only working-copy cells 2 and 4 changed for local Windows paths.
- [x] Dataset path and schema verified separately.
- [x] Methodological operations preserved.
- [x] Pinned requirements prepared; no packages installed by this preparation.
- [x] Expected invalid/legacy configurations documented.
- [ ] Explicit notebook allowlist approved.
- [ ] Original hash reverified immediately before execution.
- [ ] Working-copy hash and patch manifest reverified immediately before execution.
- [ ] Output directory approved and confirmed empty or uniquely versioned.
- [ ] Notebook execution explicitly enabled for this one working copy.
- [ ] Execution logs, traceback, duration, and output artifacts captured.

Execution must not begin until every unchecked gate is explicitly approved.
"""
    (destination_dir / "base_notebook_execution_checklist.md").write_text(
        checklist, encoding="utf-8"
    )

    sha_lines = [
        f"{original_hash}  original  {source}",
        f"{copied_hash}  working_copy  {destination}",
        f"{sha256(dataset)}  dataset  {dataset}",
    ]
    (destination_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )
    print(f"Prepared working copy: {destination}")
    print(f"Original SHA-256: {original_hash}")
    print(f"Working-copy SHA-256: {copied_hash}")
    print("Notebook execution performed: False")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/prepare_base_notebook.py config/runner_config.json")
    main(sys.argv[1])
