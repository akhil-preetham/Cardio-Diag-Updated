"""Reproducible duplicate and partition-overlap audit for CARDIODIAG datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "dataset_quality" / "duplicate_audit_v1"
DATASETS = {
    "heart5": {
        "path": ROOT / "reference" / "author_release" / "heart5.csv",
        "target": "HeartDisease",
    },
    "mendeley": {
        "path": ROOT / "reference" / "author_release" / "MendeleyCardiovascular_Disease_Dataset.csv",
        "target": "target",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return f"{type(value).__name__}:{value!s}"


def row_key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(canonical_value(row[c]) for c in columns)


def grouped_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in df.iterrows():
        groups[row_key(row, columns)].append(int(index))
    return [
        {"key": key, "indices": indices}
        for key, indices in groups.items()
        if len(indices) > 1
    ]


def duplicate_group_summary(
    df: pd.DataFrame, target: str, columns: list[str], comparison: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = grouped_records(df, columns)
    details = []
    for group_number, group in enumerate(groups, 1):
        labels = sorted({canonical_value(df.loc[i, target]) for i in group["indices"]})
        details.append(
            {
                "dataset": str(df.attrs["dataset_name"]),
                "comparison": comparison,
                "group_id": f"{df.attrs['dataset_name']}_{comparison}_{group_number:04d}",
                "row_indices": group["indices"],
                "labels": labels,
                "group_size": len(group["indices"]),
                "first_occurrence": group["indices"][0],
                "subsequent_occurrences": group["indices"][1:],
                "target_consistent": len(labels) == 1,
            }
        )
    duplicate_occurrences = sum(item["group_size"] - 1 for item in details)
    summary = {
        "comparison": comparison,
        "duplicate_groups": len(details),
        "duplicated_records": duplicate_occurrences,
        "duplicated_record_percentage": (
            100.0 * duplicate_occurrences / len(df) if len(df) else 0.0
        ),
        "unique_rows": len(df) - duplicate_occurrences,
        "group_sizes": [item["group_size"] for item in details],
        "conflicting_target_groups": sum(not item["target_consistent"] for item in details),
        "identical_target_groups": sum(item["target_consistent"] for item in details),
    }
    return summary, details


def _numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]


def near_duplicate_candidates(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    numeric_tolerance: float = 0.01,
    categorical_mismatch_limit: int = 0,
) -> list[dict[str, Any]]:
    """Find conservative near pairs: same categorical block, numeric range within 1%.

    Numeric tolerance is a fraction of each column's observed range (with a
    minimum absolute tolerance of 1e-9); all comparable numeric values must
    satisfy it. Missing values compare equal. This is intentionally a candidate
    screen, not an identity claim.
    """
    numeric = _numeric_columns(df, columns)
    categorical = [column for column in columns if column not in numeric]
    tolerances = {}
    numeric_values = {}
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce")
        observed_range = float(values.max() - values.min()) if values.notna().any() else 0.0
        tolerances[column] = max(observed_range * numeric_tolerance, 1e-9)
        numeric_values[column] = values.to_numpy()
    categorical_values = {
        column: df[column].map(canonical_value).to_numpy() for column in categorical
    }
    target_values = df[target].to_numpy()
    blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(df)):
        blocks[tuple(categorical_values[c][index] for c in categorical)].append(index)
    candidates = []
    for indices in blocks.values():
        for left, right in combinations(indices, 2):
            mismatches = 0
            evidence = {}
            comparable = 0
            for column in numeric:
                a, b = numeric_values[column][left], numeric_values[column][right]
                if pd.isna(a) or pd.isna(b):
                    if pd.isna(a) and pd.isna(b):
                        evidence[column] = {"difference": 0.0, "tolerance": tolerances[column]}
                    else:
                        mismatches += 1
                    continue
                comparable += 1
                difference = abs(float(a) - float(b))
                evidence[column] = {"difference": difference, "tolerance": tolerances[column]}
                if difference > tolerances[column]:
                    mismatches += 1
            if mismatches <= categorical_mismatch_limit and comparable == len(numeric):
                candidates.append(
                    {
                        "dataset": str(df.attrs["dataset_name"]),
                        "row_index_a": left,
                        "row_index_b": right,
                        "numeric_columns": len(numeric),
                        "categorical_columns": len(categorical),
                        "numeric_tolerance_fraction": numeric_tolerance,
                        "max_relative_range_difference": max(
                            (
                                evidence[c]["difference"] / tolerances[c]
                                for c in numeric
                            ),
                            default=0.0,
                        ),
                        "target_a": jsonable(target_values[left]),
                        "target_b": jsonable(target_values[right]),
                        "target_conflict": bool(target_values[left] != target_values[right]),
                        "evidence_json": json.dumps(evidence, sort_keys=True),
                    }
                )
    return candidates


def near_groups(candidates: list[dict[str, Any]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for item in candidates:
        a, b = item["row_index_a"], item["row_index_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    groups = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack, group = [start], []
        while stack:
            node = stack.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            group.append(node)
            stack.extend(adjacency[node])
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: group[0])


def partition_overlaps(
    df: pd.DataFrame,
    target: str,
    predictor_columns: list[str],
    seeds: list[int],
    test_size: float = 0.25,
) -> list[dict[str, Any]]:
    rows = []
    keys = [row_key(df.loc[i], predictor_columns) for i in range(len(df))]
    for run, seed in enumerate(seeds, 1):
        train_indices, test_indices = train_test_split(
            np.arange(len(df)), test_size=test_size, stratify=df[target], random_state=seed
        )
        train_set, test_set = set(map(int, train_indices)), set(map(int, test_indices))
        train_full = {row_key(df.loc[i], list(df.columns)) for i in train_set}
        test_full = {row_key(df.loc[i], list(df.columns)) for i in test_set}
        train_predictors = {keys[i] for i in train_set}
        test_predictors = {keys[i] for i in test_set}
        full_overlap = train_full & test_full
        predictor_overlap = train_predictors & test_predictors
        conflicting = 0
        for key in predictor_overlap:
            train_labels = {canonical_value(df.loc[i, target]) for i in train_set if keys[i] == key}
            test_labels = {canonical_value(df.loc[i, target]) for i in test_set if keys[i] == key}
            if train_labels & test_labels == set() and train_labels and test_labels:
                conflicting += 1
        rows.append(
            {
                "dataset": str(df.attrs["dataset_name"]),
                "run": run,
                "seed": seed,
                "n_train": len(train_set),
                "n_test": len(test_set),
                "full_row_overlap_groups": len(full_overlap),
                "full_row_overlap_test_records": sum(
                    1 for i in test_set if row_key(df.loc[i], list(df.columns)) in full_overlap
                ),
                "predictor_overlap_groups": len(predictor_overlap),
                "predictor_overlap_test_records": sum(1 for i in test_set if keys[i] in predictor_overlap),
                "conflicting_label_overlap_groups": conflicting,
            }
        )
    return rows


def inventory_entry(name: str, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    path = config["path"]
    if not path.exists():
        return {"dataset": name, "available": False, "path": str(path)}, pd.DataFrame()
    df = pd.read_csv(path)
    df.attrs["dataset_name"] = name
    target = config["target"]
    return (
        {
            "dataset": name,
            "available": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "target": target,
            "target_class_distribution": {
                str(k): int(v) for k, v in df[target].value_counts(dropna=False).sort_index().items()
            },
            "missing_value_counts": {c: int(v) for c, v in df.isna().sum().items()},
        },
        df,
    )


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# Duplicate record audit",
        "",
        f"Generated UTC: `{result['generated_utc']}`",
        "",
        "## Scope and method",
        "",
        "This diagnostic audit examines only duplicate records and their effect on partition overlap. "
        "Source CSVs are read without modification. Row indices are zero-based indices after CSV loading "
        "(the CSV header is not counted). Exact duplicates are grouped by all columns; predictor-only "
        "groups exclude the target, and Mendeley analyses also report the identifier policy separately.",
        "",
        "Near-duplicate candidates use a conservative deterministic screen: all categorical predictors "
        "must match, every numeric predictor must be non-missing and differ by no more than 1% of that "
        "column's observed range, and candidates are not merged or treated as confirmed duplicates. "
        "The candidate-generation block is the categorical signature, avoiding unrestricted comparisons "
        "across dissimilar records.",
        "",
        "Partition overlap reproduces the corrected-validation holdout protocol (stratified 75/25 "
        "train/test, seeds 42–51). It checks overlap within each train/test partition only; repeated "
        "holdouts naturally reuse observations across different runs.",
        "",
    ]
    for name, item in result["datasets"].items():
        if not item["available"]:
            lines += [f"## {name}", "", "Dataset unavailable; no substitute was downloaded.", ""]
            continue
        inv = item["inventory"]
        exact = item["exact_duplicates"]
        pred = item["predictor_duplicates"]
        lines += [
            f"## {name}",
            "",
            f"- Source: `{inv['path']}`",
            f"- Shape: {inv['rows']} rows × {inv['columns']} columns; target `{inv['target']}`",
            f"- Exact duplicate groups: {exact['duplicate_groups']}; subsequent duplicate occurrences: "
            f"{exact['duplicated_records']} ({exact['duplicated_record_percentage']:.3f}%)",
            f"- Exact groups with conflicting labels: {exact['conflicting_target_groups']}",
            f"- Predictor-only duplicate groups: {pred['duplicate_groups']}; conflicting-label groups: "
            f"{pred['conflicting_target_groups']}",
            f"- Near-duplicate candidate pairs: {item['near_duplicates']['candidate_pairs']}; "
            f"connected candidate groups: {item['near_duplicates']['groups']}; "
            f"conflicting-label pairs: {item['near_duplicates']['conflicting_label_pairs']}",
            "",
            "### Interpretation",
            "",
            "Duplicate groups are diagnostic evidence, not proof of leakage. Cross-partition leakage "
            "requires the same record or predictor profile to appear in both train and test.",
        ]
        if name == "mendeley":
            lines += [
                f"- `patientid` is excluded by the corrected-validation implementation. "
                f"Repeated patient IDs: {item['patient_id']['repeated_ids']}; "
                f"IDs with conflicting observations: {item['patient_id']['conflicting_observation_ids']}.",
            ]
        lines += ["", "### Diagnostic scenarios", ""]
        for scenario, values in item["scenarios"].items():
            lines.append(
                f"- {scenario}: {values['rows_remaining']} rows; class distribution "
                f"{json.dumps(values['class_distribution'], sort_keys=True)}"
            )
        lines += ["", "### Partition overlap", ""]
        overlap = item["partition_overlap"]
        lines += [
            f"- Affected holdouts with full-row overlap: {overlap['affected_full_row_partitions']}/"
            f"{overlap['partitions']}; predictor-only overlap: {overlap['affected_predictor_partitions']}/"
            f"{overlap['partitions']}; conflicting-label overlap: "
            f"{overlap['conflicting_label_partitions']}/{overlap['partitions']}.",
            "",
        ]
    lines += [
        "## Conclusions",
        "",
        "No record was removed from a source or used to produce model results. Exact duplicate rows "
        "with identical labels can inflate apparent sample size, while conflicting labels require "
        "separate review. Identical predictors do not establish patient identity, and near-duplicate "
        "candidates are not confirmed duplicates.",
        "",
        "The unusually high Mendeley performance can be attributed to duplicate leakage only if the "
        "partition table shows train/test overlap under the same protocol. If it does not, duplicates "
        "alone do not explain that performance; dataset separability and other design/data issues remain "
        "outside this Problem 1 audit.",
        "",
        "Limitations: no patient-level ground truth is available beyond the supplied identifier; the "
        "near-duplicate threshold is a conservative screening rule; repeated holdouts are not independent "
        "external cohorts; and the approximately 70,000-row Kaggle dataset was not available locally.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "audit_version": "duplicate_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "near_duplicate_criteria": {
            "categorical_mismatch_limit": 0,
            "numeric_tolerance_fraction_of_observed_range": 0.01,
            "missing_numeric_values": "not candidates unless both values are missing",
        },
        "validation_protocol": {"seeds": list(range(42, 52)), "test_size": 0.25, "stratified": True},
    }
    inventory_rows, group_rows, near_rows, overlap_rows = [], [], [], []
    for name, config in DATASETS.items():
        inv, df = inventory_entry(name, config)
        inventory_rows.append(inv)
        if not inv["available"]:
            result["datasets"][name] = {"available": False, "inventory": inv}
            continue
        target = config["target"]
        all_columns = list(df.columns)
        predictor_columns = [c for c in all_columns if c != target]
        identifier_columns = [
            c for c in predictor_columns
            if c.lower() in {"id", "patientid", "patient_id", "subject_id"} or c.lower().endswith("_id")
        ]
        predictor_without_ids = [c for c in predictor_columns if c not in identifier_columns]
        exact, exact_groups = duplicate_group_summary(df, target, all_columns, "exact")
        pred, pred_groups = duplicate_group_summary(df, target, predictor_columns, "predictor")
        for group in exact_groups + pred_groups:
            group_rows.append(group)
        near = near_duplicate_candidates(df, target, predictor_without_ids)
        near_group_list = near_groups(near)
        for item in near:
            near_rows.append(item)
        split_rows = partition_overlaps(df, target, predictor_without_ids, list(range(42, 52)))
        overlap_rows.extend(split_rows)
        patient = {
            "identifier_columns": identifier_columns,
            "repeated_ids": 0,
            "conflicting_observation_ids": 0,
            "duplicate_records_including_patientid": exact if name == "mendeley" else None,
        }
        if "patientid" in df.columns:
            id_groups = df.groupby("patientid", dropna=False, sort=True)
            repeated = id_groups.filter(lambda group: len(group) > 1)
            patient["repeated_ids"] = int(sum(len(group) > 1 for _, group in id_groups))
            patient["conflicting_observation_ids"] = int(
                sum(group.drop(columns=["patientid"]).drop_duplicates().shape[0] > 1 for _, group in id_groups if len(group) > 1)
            )
            patient["predictor_duplicates_excluding_patientid"] = pred
        exact_subsequent = {
            index
            for group in exact_groups
            for index in group["subsequent_occurrences"]
        }
        predictor_subsequent = {
            index
            for group in pred_groups
            for index in group["subsequent_occurrences"]
        }
        scenario_groups = {
            "original_dataset": df.index,
            "exact_duplicate_rows_excluded": [index for index in df.index if index not in exact_subsequent],
            "identical_predictor_groups_conservatively_deduplicated": [
                index for index in df.index if index not in predictor_subsequent
            ],
            "conflicting_label_groups_flagged_not_removed": df.index,
        }
        scenarios = {}
        for scenario, indices in scenario_groups.items():
            subset = df.loc[indices]
            scenarios[scenario] = {
                "rows_remaining": len(subset),
                "class_distribution": {
                    str(k): int(v) for k, v in subset[target].value_counts(dropna=False).sort_index().items()
                },
            }
        result["datasets"][name] = {
            "available": True,
            "inventory": inv,
            "identifier_columns": identifier_columns,
            "exact_duplicates": exact,
            "predictor_duplicates": pred,
            "near_duplicates": {
                "candidate_pairs": len(near),
                "groups": len(near_group_list),
                "conflicting_label_pairs": sum(item["target_conflict"] for item in near),
                "examples": near[:10],
                "groups_row_indices": near_group_list[:10],
            },
            "patient_id": patient,
            "scenarios": scenarios,
            "partition_overlap": {
                "partitions": len(split_rows),
                "affected_full_row_partitions": sum(r["full_row_overlap_groups"] > 0 for r in split_rows),
                "affected_predictor_partitions": sum(r["predictor_overlap_groups"] > 0 for r in split_rows),
                "conflicting_label_partitions": sum(r["conflicting_label_overlap_groups"] > 0 for r in split_rows),
                "total_full_row_overlap_test_records": sum(r["full_row_overlap_test_records"] for r in split_rows),
                "total_predictor_overlap_test_records": sum(r["predictor_overlap_test_records"] for r in split_rows),
            },
        }
    result["unavailable_kaggle_dataset"] = True
    pd.DataFrame(inventory_rows).to_csv(output_dir / "dataset_inventory.csv", index=False)
    group_columns = [
        "dataset", "comparison", "group_id", "row_indices", "labels", "group_size",
        "first_occurrence", "subsequent_occurrences", "target_consistent",
    ]
    near_columns = [
        "dataset", "row_index_a", "row_index_b", "numeric_columns", "categorical_columns",
        "numeric_tolerance_fraction", "max_relative_range_difference", "target_a", "target_b",
        "target_conflict", "evidence_json",
    ]
    overlap_columns = [
        "dataset", "run", "seed", "n_train", "n_test", "full_row_overlap_groups",
        "full_row_overlap_test_records", "predictor_overlap_groups",
        "predictor_overlap_test_records", "conflicting_label_overlap_groups",
    ]
    pd.DataFrame(group_rows, columns=group_columns).to_csv(
        output_dir / "duplicate_groups.csv", index=False
    )
    pd.DataFrame(near_rows, columns=near_columns).to_csv(
        output_dir / "near_duplicate_candidates.csv", index=False
    )
    pd.DataFrame(overlap_rows, columns=overlap_columns).to_csv(
        output_dir / "partition_overlap.csv", index=False
    )
    (output_dir / "duplicate_audit.json").write_text(json.dumps(result, indent=2, default=jsonable), encoding="utf-8")
    (output_dir / "duplicate_audit_report.md").write_text(make_report(result), encoding="utf-8")
    source_hashes = {
        name: item["inventory"].get("sha256")
        for name, item in result["datasets"].items()
        if item.get("available")
    }
    manifest = {
        "audit_version": result["audit_version"],
        "generated_utc": result["generated_utc"],
        "script": str(Path(__file__).resolve()),
        "source_files": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "configuration": result["validation_protocol"],
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.name != "audit_manifest.json"
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_audit(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": list(result["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
