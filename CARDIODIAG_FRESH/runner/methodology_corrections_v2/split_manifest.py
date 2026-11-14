"""Serialize and validate exact nested-CV assignments without fitting models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _rows_for_split(
    level: str,
    repeat: int,
    fold: int,
    seed: int,
    train: Iterable[int],
    test: Iterable[int],
    source_row_ids: Sequence[str],
) -> list[dict]:
    rows = []
    for role, indices in (("train", train), ("test", test)):
        for index in indices:
            rows.append(
                {
                    "level": level,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "seed": int(seed),
                    "role": role,
                    "row_index": int(index),
                    "source_row_id": str(source_row_ids[index]),
                }
            )
    return rows


def create_nested_split_manifest(
    y: Sequence[int],
    outer_seeds: Sequence[int],
    outer_folds: int = 5,
    inner_folds: int = 4,
    source_row_ids: Sequence[str] | None = None,
) -> dict:
    """Create exact outer and inner assignments using the primary seed convention.

    ``source_row_ids`` are stable source-row labels only. They are never presented as
    patient identifiers; callers must provide legitimate patient IDs separately if they
    exist.
    """
    labels = np.asarray(y)
    if source_row_ids is None:
        source_row_ids = [f"source_row:{index}" for index in range(len(labels))]
    if len(source_row_ids) != len(labels):
        raise ValueError("source_row_ids must have one entry per target row")
    if len(set(map(str, source_row_ids))) != len(labels):
        raise ValueError("source_row_ids must be unique")

    outer_rows: list[dict] = []
    inner_rows: list[dict] = []
    for repeat, seed in enumerate(outer_seeds, start=1):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=int(seed))
        for fold, (train_index, test_index) in enumerate(outer.split(np.zeros(len(labels)), labels), start=1):
            outer_rows.extend(
                _rows_for_split("outer", repeat, fold, int(seed), train_index, test_index, source_row_ids)
            )
            for row in outer_rows[-(len(train_index) + len(test_index)) :]:
                row["outer_fold"] = fold
            inner_seed = int(seed) + repeat * 100 + fold
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
            outer_train_labels = labels[train_index]
            for inner_fold, (inner_train_local, inner_test_local) in enumerate(
                inner.split(np.zeros(len(train_index)), outer_train_labels), start=1
            ):
                inner_train = train_index[inner_train_local]
                inner_test = train_index[inner_test_local]
                new_inner_rows = _rows_for_split(
                    "inner",
                    repeat,
                    inner_fold,
                    inner_seed,
                    inner_train,
                    inner_test,
                    source_row_ids,
                )
                for row in new_inner_rows:
                    row["outer_fold"] = fold
                inner_rows.extend(
                    new_inner_rows
                )
    return {
        "format_version": "nested_split_manifest_v2",
        "source_row_identifier_policy": "stable source-row identifier; not a patient identifier",
        "n_rows": len(labels),
        "outer_seeds": [int(seed) for seed in outer_seeds],
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "outer_assignments": outer_rows,
        "inner_assignments": inner_rows,
    }


def validate_nested_split_manifest(manifest: dict) -> dict:
    """Validate coverage, disjointness, and inner-within-outer invariants."""
    n_rows = int(manifest["n_rows"])
    outer = manifest["outer_assignments"]
    inner = manifest["inner_assignments"]
    errors: list[str] = []
    outer_groups = {}
    for row in outer:
        outer_groups.setdefault((row["repeat"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(
            row["row_index"]
        )
    for key, roles in outer_groups.items():
        if roles["train"] & roles["test"]:
            errors.append(f"outer train/test overlap: {key}")
        if len(roles["train"] | roles["test"]) != n_rows:
            errors.append(f"outer incomplete coverage: {key}")

    for repeat_number, seed in enumerate(manifest["outer_seeds"], start=1):
        repeat_tests = {}
        for row in outer:
            if row["repeat"] == repeat_number and row["role"] == "test":
                repeat_tests.setdefault(row["row_index"], 0)
                repeat_tests[row["row_index"]] += 1
        if any(count != 1 for count in repeat_tests.values()) or len(repeat_tests) != n_rows:
            errors.append(f"outer test coverage violation: repeat {repeat_number}, seed {seed}")

    inner_groups = {}
    for row in inner:
        inner_groups.setdefault(
            (row["repeat"], row["outer_fold"], row["fold"]), {"train": set(), "test": set()}
        )[row["role"]].add(row["row_index"])
    for key, roles in inner_groups.items():
        outer_train = outer_groups.get((key[0], key[1]), {}).get("train", set())
        if roles["train"] & roles["test"]:
            errors.append(f"inner train/test overlap: {key}")
        if not roles["train"] | roles["test"] <= outer_train:
            errors.append(f"inner rows outside outer training set: {key}")
        if roles["train"] | roles["test"] != outer_train:
            errors.append(f"inner incomplete outer-training coverage: {key}")
    return {
        "valid": not errors,
        "errors": errors,
        "outer_groups": len(outer_groups),
        "inner_groups": len(inner_groups),
        "n_rows": n_rows,
    }


def save_nested_split_manifest(path: Path, manifest: dict) -> None:
    validation = validate_nested_split_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(json.dumps(validation))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**manifest, "validation": validation}, indent=2), encoding="utf-8")


def load_and_validate_nested_split_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_nested_split_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(json.dumps(validation))
    return manifest
