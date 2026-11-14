from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

from common import load_config, resolve, utc_now, write_json


RISK_NAMES = [
    "SMOTE split order",
    "Preprocessing split order",
    "Test-set influence",
    "Cross-validation nesting",
    "Identifier or duplicate leakage",
    "OVEC/OSEC implementation",
    "XAIH-DRC/LoRAS implementation",
    "LIME held-out observations",
    "Paired statistical predictions",
    "Fairness subgroup validity",
    "Clinical utility assumptions",
    "External or destructive operations",
]

MODULE_TO_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "PIL": "Pillow",
    "google": "google-colab",
    "cv2": "opencv-python",
}

BASELINE_REQUIREMENTS = {
    "numpy", "pandas", "sklearn", "imblearn", "xgboost", "matplotlib",
    "nbformat", "nbclient", "ipykernel",
}


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def active_lines(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def excerpt(source: str, pattern: str, context: int = 1) -> str:
    lines = source.splitlines()
    regex = re.compile(pattern, re.I)
    for index, line in enumerate(lines):
        if regex.search(line):
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            return "\n".join(lines[start:end]).strip()
    return source.strip()[:500]


def evidence(notebook: str, cell: int, risk: str, status: str, impact: str,
             code: str, does: str, problem: str, remediation: str) -> dict:
    return {
        "notebook": notebook,
        "cell": cell,
        "risk": risk,
        "status": status,
        "impact": impact,
        "excerpt": code,
        "what_code_does": does,
        "why_problematic": problem,
        "remediation": remediation,
    }


def first_cell(cells: list[dict], pattern: str, active_only: bool = True):
    regex = re.compile(pattern, re.I)
    for index, source in cells:
        haystack = active_lines(source) if active_only else source
        if regex.search(haystack):
            return index, source
    return None, ""


def all_cells(cells: list[dict], pattern: str, active_only: bool = True):
    regex = re.compile(pattern, re.I)
    found = []
    for index, source in cells:
        haystack = active_lines(source) if active_only else source
        if regex.search(haystack):
            found.append((index, source))
    return found


def build_risks(relative: str, cells: list[dict]) -> list[dict]:
    active = [(index, active_lines(source)) for index, source in cells]
    all_source = "\n".join(source for _, source in cells)
    active_source = "\n".join(source for _, source in active)
    risks = []

    smote_cells = all_cells(cells, r"\bSMOTE\s*\(|\bsmote(?:_?\w*)?\.fit_resample\s*\(")
    split_cells = all_cells(cells, r"train_test_split\s*\(")
    if smote_cells and split_cells:
        smote_pos = min(index for index, _ in smote_cells)
        split_pos = min(index for index, _ in split_cells)
        if smote_pos < split_pos:
            status, text = "confirmed", "SMOTE is fitted before the first train/test split."
            cell, source = smote_cells[0]
        elif smote_pos > split_pos:
            status, text = "confirmed", "The first visible SMOTE operation follows the first train/test split."
            cell, source = smote_cells[0]
        else:
            status, text = "suspected", "SMOTE and splitting occur in the same cell; statement order requires review."
            cell, source = smote_cells[0]
        risks.append(evidence(
            relative, cell, "SMOTE split order", status,
            "author-faithful and corrected validation",
            excerpt(source, r"fit_resample|SMOTE"),
            text,
            "Resampling before splitting allows synthetic information derived from eventual test observations to enter training.",
            "For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.",
        ))
    elif first_cell(cells, r"\bSMOTE\b", active_only=False)[0] is not None:
        cell, source = first_cell(cells, r"\bSMOTE\b", active_only=False)
        risks.append(evidence(
            relative, cell, "SMOTE split order", "not established",
            "author-faithful and corrected validation",
            excerpt(source, r"\bSMOTE\b"),
            "SMOTE is mentioned, but no active fit_resample call and split order were both found statically.",
            "A name or import does not prove that resampling was executed.",
            "Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.",
        ))

    prep_cells = all_cells(cells, r"\.fit_transform\s*\(|\.fit\s*\(")
    if prep_cells and split_cells:
        prep_pos = min(index for index, _ in prep_cells)
        split_pos = min(index for index, _ in split_cells)
        if prep_pos < split_pos:
            cell, source = prep_cells[0]
            risks.append(evidence(
                relative, cell, "Preprocessing split order", "confirmed",
                "author-faithful and corrected validation",
                excerpt(source, r"\.fit_transform|\.fit\s*\("),
                "A transformer/model-like fit operation appears before the first train/test split.",
                "Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.",
                "Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.",
            ))
        else:
            risks.append(evidence(
                relative, prep_cells[0][0], "Preprocessing split order", "not established",
                "author-faithful and corrected validation",
                excerpt(prep_cells[0][1], r"\.fit_transform|\.fit\s*\("),
                "Preprocessing fit operations were detected, but static ordering does not prove whether all transformers are fit only on training data.",
                "A later fit may still use the full dataset or a prior transformation may have occurred in another cell.",
                "Inspect each transformer and enforce training-fold pipelines in corrected validation.",
            ))

    grid_cells = all_cells(cells, r"GridSearchCV\s*\(|RandomizedSearchCV\s*\(")
    if grid_cells and split_cells:
        cell, source = grid_cells[0]
        risks.append(evidence(
            relative, cell, "Cross-validation nesting", "confirmed",
            "author-faithful and corrected validation",
            excerpt(source, r"GridSearchCV|RandomizedSearchCV"),
            "An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.",
            "Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.",
            "Use nested CV or reserve an untouched test set and perform all search only on training data.",
        ))
    elif grid_cells:
        cell, source = grid_cells[0]
        risks.append(evidence(
            relative, cell, "Cross-validation nesting", "suspected",
            "author-faithful and corrected validation",
            excerpt(source, r"GridSearchCV|RandomizedSearchCV"),
            "A CV search object is present, but no outer evaluation structure was established statically.",
            "Cross-validation may be valid for selection but cannot by itself establish an unbiased external estimate.",
            "Document the evaluation boundary and use nested CV or a held-out test set.",
        ))

    test_selection_pattern = (
        r"early[\s_-]*stop|"
        r"test[\s_-]+(?:set|data).{0,80}(?:select|tune|criterion)|"
        r"(?:select|tune).{0,80}test[\s_-]+(?:set|data)"
    )
    if re.search(test_selection_pattern, active_source, re.I):
        cell, source = first_cell(cells, test_selection_pattern)
        risks.append(evidence(
            relative, cell, "Test-set influence", "confirmed",
            "author-faithful and corrected validation",
            excerpt(source, r"early.?stop|test.*best|best.*test|X_test|y_test"),
            "The source contains an explicit test-set selection/stopping pattern.",
            "Using test performance to choose a model, threshold, or stopping point invalidates the final test estimate.",
            "Select models and thresholds using training CV only, then evaluate the test set once.",
        ))
    elif grid_cells and split_cells:
        cell, source = grid_cells[0]
        risks.append(evidence(
            relative, cell, "Test-set influence", "not established",
            "author-faithful and corrected validation",
            excerpt(source, r"GridSearchCV|RandomizedSearchCV"),
            "Model-selection code is present, but direct test-set selection was not established by this static rule.",
            "Saved outputs or later cells may still have influenced choices.",
            "Review the complete execution path and keep test data out of all selection decisions.",
        ))

    if re.search(r"patientid|patient_id|subject_id|\bid\b", active_source, re.I):
        cell, source = first_cell(cells, r"patientid|patient_id|subject_id|\bid\b")
        risks.append(evidence(
            relative, cell, "Identifier or duplicate leakage", "suspected",
            "author-faithful and corrected validation",
            excerpt(source, r"patientid|patient_id|subject_id|\bid\b"),
            "An identifier-like field or identifier handling is present in the source.",
            "Identifiers can memorize subjects or permit patient overlap across partitions; duplicate handling may also alter the cohort.",
            "Audit identifier uniqueness and duplicates, exclude identifiers from predictors, and use grouped splitting when repeated patients exist.",
        ))
    elif re.search(r"duplicat", active_source, re.I):
        cell, source = first_cell(cells, r"duplicat")
        risks.append(evidence(
            relative, cell, "Identifier or duplicate leakage", "not established",
            "author-faithful and corrected validation",
            excerpt(source, r"duplicat"),
            "Duplicate checking/removal is referenced, but patient-level independence is not established.",
            "Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.",
            "Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.",
        ))

    for method in ("OVEC", "OSEC"):
        matches = all_cells(cells, rf"\b{method}\b")
        if matches:
            cell, source = matches[0]
            active_method = any(re.search(rf"\b{method}\b", active_lines(s), re.I) for _, s in matches)
            status = "suspected" if active_method else "not established"
            risks.append(evidence(
                relative, cell, "OVEC/OSEC implementation", status,
                "author-faithful and corrected validation",
                excerpt(source, rf"\b{method}\b"),
                f"{method} is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.",
                "A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.",
                f"Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.",
            ))

    for method, pattern in (("XAIH-DRC", r"XAIH|DRC"), ("LoRAS", r"\bloras\b|LoRAS")):
        matches = all_cells(cells, pattern, active_only=False)
        if matches:
            cell, source = matches[0]
            active_method = any(re.search(pattern, active_lines(s), re.I) for _, s in matches)
            risks.append(evidence(
                relative, cell, "XAIH-DRC/LoRAS implementation", "suspected" if active_method else "not established",
                "author-faithful and corrected validation",
                excerpt(source, pattern),
                f"{method} is {'actively referenced' if active_method else 'mentioned only'} in source.",
                "The name alone does not establish a faithful algorithm, parameterization, or execution path.",
                f"Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.",
            ))

    lime_matches = all_cells(cells, r"LimeTabularExplainer|explain_instance")
    if lime_matches:
        cell, source = lime_matches[0]
        heldout = bool(re.search(r"X_test|x_test|test", active_source, re.I))
        risks.append(evidence(
            relative, cell, "LIME held-out observations", "suspected" if heldout else "not established",
            "author-faithful and corrected validation",
            excerpt(source, r"LimeTabularExplainer|explain_instance"),
            "LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.",
            "Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.",
            "Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.",
        ))

    stats_matches = all_cells(cells, r"bootstrap|wilcoxon|delong|roc_curve")
    if stats_matches:
        cell, source = stats_matches[0]
        risks.append(evidence(
            relative, cell, "Paired statistical predictions", "suspected",
            "author-faithful and corrected validation",
            excerpt(source, r"bootstrap|wilcoxon|delong|roc_curve"),
            "Bootstrap or statistical comparison code is present.",
            "Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.",
            "Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.",
        ))

    fairness_matches = all_cells(cells, r"subgroup|groupby|fairness|gender|sex|age")
    if fairness_matches:
        cell, source = fairness_matches[0]
        risks.append(evidence(
            relative, cell, "Fairness subgroup validity", "suspected",
            "author-faithful and corrected validation",
            excerpt(source, r"subgroup|groupby|fairness|gender|sex|age"),
            "Subgroup/fairness analysis code is present.",
            "Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.",
            "Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.",
        ))

    clinical_matches = all_cells(cells, r"decision_curve|net benefit|clinical utility|threshold")
    if clinical_matches:
        cell, source = clinical_matches[0]
        risks.append(evidence(
            relative, cell, "Clinical utility assumptions", "suspected",
            "author-faithful and corrected validation",
            excerpt(source, r"decision_curve|net benefit|clinical utility|threshold"),
            "Decision-curve, threshold, or clinical-utility code is present.",
            "Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.",
            "Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.",
        ))

    destructive = all_cells(cells, r"google\.colab|drive\.mount|!pip|%pip|subprocess|os\.system|shutil\.rmtree|os\.remove|\.unlink\s*\(|wget|curl")
    if destructive:
        cell, source = destructive[0]
        risks.append(evidence(
            relative, cell, "External or destructive operations", "confirmed",
            "author-faithful reproduction only",
            excerpt(source, r"google\.colab|drive\.mount|!pip|%pip|subprocess|os\.system|shutil\.rmtree|os\.remove|\.unlink\s*\(|wget|curl"),
            "The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.",
            "Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.",
            "Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.",
        ))

    return risks


def dependency_rows(notebooks: list[dict]) -> list[dict]:
    rows = []
    for item in notebooks:
        imports = sorted(item["imports"])
        roots = sorted({name.split(".")[0] for name in imports})
        missing = []
        for root in roots:
            if root in {"os", "sys", "json", "random", "time", "warnings", "collections", "shutil"}:
                continue
            if importlib.util.find_spec(root) is None:
                missing.append(root)
        api_issues = []
        source = item.get("_source", "")
        if re.search(r"CalibratedClassifierCV\s*\([^)]*base_estimator\s*=", source, re.S):
            api_issues.append("CalibratedClassifierCV(base_estimator=...) may be incompatible with current scikit-learn.")
        if re.search(r"OneHotEncoder\s*\([^)]*\bsparse\s*=", source, re.S):
            api_issues.append("OneHotEncoder(sparse=...) may be incompatible with current scikit-learn; inspect version-specific API.")
        if ".append(" in source and "DataFrame" in source:
            api_issues.append("DataFrame.append may be removed in current pandas.")
        rows.append({
            "notebook": item["relative_path"],
            "imports": "|".join(imports),
            "missing_in_selected_environment": "|".join(missing),
            "potential_api_incompatibilities": " | ".join(api_issues),
            "external_files": "|".join(item["referenced_paths"]),
            "external_file_status": "requires manual existence/path verification" if item["referenced_paths"] else "none detected",
        })
    return rows


def main(config_file: str) -> None:
    cfg, root = load_config(Path(config_file).resolve())
    extracted = resolve(root, cfg["extracted_archives"])
    output = resolve(root, cfg["output_root"]) / "audit"
    inventory = json.loads((output / "notebook_inventory.json").read_text(encoding="utf-8"))["inventory"]
    evidence_rows = []
    enriched = []
    for item in inventory:
        path = Path(item["path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = [
            (index, source_text(cell))
            for index, cell in enumerate(data.get("cells", []), 1)
            if cell.get("cell_type") == "code"
        ]
        item["_source"] = "\n".join(source for _, source in cells)
        item["evidence_risks"] = build_risks(item["relative_path"], cells)
        evidence_rows.extend(item["evidence_risks"])
        enriched.append(item)

    md = [
        "# Evidence-based second-level risk audit", "",
        f"Generated: `{utc_now()}`", "",
        "This is a source-only audit. Notebook kernels were not started and no notebook cells were executed.",
        "Statuses mean **confirmed** (the cited source establishes the behavior), **suspected** (the source indicates a risk but the full runtime path or invariant is not established), or **not established** (the static evidence is insufficient).",
        "",
    ]
    for row in evidence_rows:
        md.extend([
            f"## {row['notebook']} — cell {row['cell']}: {row['risk']}",
            f"- Status: **{row['status']}**",
            f"- Affects: {row['impact']}",
            "- Source excerpt:",
            "```python",
            row["excerpt"],
            "```",
            f"- What the code does: {row['what_code_does']}",
            f"- Why potentially problematic: {row['why_problematic']}",
            f"- Exact remediation: {row['remediation']}",
            "",
        ])
    (output / "evidence_based_risk_audit.md").write_text("\n".join(md), encoding="utf-8")

    matrix_rows = []
    for row in evidence_rows:
        matrix_rows.append({
            "notebook": row["notebook"],
            "cell": row["cell"],
            "risk": row["risk"],
            "status": row["status"],
            "affects": row["impact"],
            "evidence_excerpt": row["excerpt"].replace("\n", " "),
            "remediation": row["remediation"],
        })
    with (output / "methodology_compliance_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]) if matrix_rows else [
            "notebook", "cell", "risk", "status", "affects", "evidence_excerpt", "remediation"
        ])
        writer.writeheader()
        writer.writerows(matrix_rows)

    dependency_matrix = dependency_rows(enriched)
    with (output / "notebook_dependency_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dependency_matrix[0]))
        writer.writeheader()
        writer.writerows(dependency_matrix)

    blockers = [
        "# Execution blockers",
        "",
        "No notebook execution was authorized or performed.",
        "",
        "## Blockers applying to the collection",
        "- Most notebooks require Google Colab imports and `/content/drive` paths that are unavailable in the local runner.",
        "- Several notebooks install packages with `!pip`, write external artifacts, or contain deletion/shell operations.",
        "- Specialized dependencies such as `loras`, `lime`, LightGBM, CatBoost, ReportLab, image tooling, and plotting extras are not part of the reviewed runner requirements.",
        "- Referenced serialized models, PDFs, CSVs, and Colab Drive files require explicit existence and provenance checks.",
        "- Historical saved outputs cannot be treated as reproduced results.",
        "",
        "## Notebook-specific dependency and file blockers",
    ]
    for row in dependency_matrix:
        if row["missing_in_selected_environment"] or row["external_files"] or row["potential_api_incompatibilities"]:
            blockers.extend([
                f"### `{row['notebook']}`",
                f"- Missing/unavailable imports: `{row['missing_in_selected_environment'] or 'none detected'}`",
                f"- External files requiring verification: `{row['external_files'] or 'none detected'}`",
                f"- Potential API issues: {row['potential_api_incompatibilities'] or 'none detected'}",
                "",
            ])
    blockers.extend([
        "## Required gate before any execution",
        "1. Approve an exact notebook allowlist and execution order.",
        "2. Create and hash working copies; never edit the originals.",
        "3. Resolve every Colab path to a verified local input or mark the notebook non-reproducible.",
        "4. Review every shell, package-install, network, deletion, and file-write operation.",
        "5. Capture package versions and run in an isolated output directory.",
    ])
    (output / "execution_blockers.md").write_text("\n".join(blockers), encoding="utf-8")
    write_json(output / "evidence_based_risk_audit.json", {
        "created_utc": utc_now(), "notebook_count": len(enriched),
        "execution_performed": False, "risks": evidence_rows,
    })
    print(f"Audited {len(enriched)} notebooks without execution.")
    print(f"Evidence report: {output / 'evidence_based_risk_audit.md'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/evidence_audit.py config/runner_config.json")
    main(sys.argv[1])
