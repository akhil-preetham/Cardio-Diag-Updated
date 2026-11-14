from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetSchemaError(ValueError):
    """Raised when a dataset does not satisfy the requested schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_schema(
    frame: pd.DataFrame,
    *,
    target: str,
    required_columns: set[str] | None = None,
) -> dict[str, Any]:
    if target not in frame.columns:
        raise DatasetSchemaError(f"Target column {target!r} is missing")
    required = required_columns or set()
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetSchemaError(f"Required columns are missing: {missing}")
    if frame.empty:
        raise DatasetSchemaError("Dataset is empty")
    if frame[target].isna().any():
        raise DatasetSchemaError("Target contains missing values")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "target": target,
        "target_values": frame[target].value_counts(dropna=False).to_dict(),
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "identifier_columns": [
            column
            for column in frame.columns
            if column.lower() in {"id", "patientid", "patient_id", "subject_id"}
            or column.lower().endswith("_id")
        ],
    }


def load_dataset(
    path: str | Path,
    *,
    target: str,
    required_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    metadata = validate_schema(frame, target=target, required_columns=required_columns)
    metadata["path"] = str(source.resolve())
    metadata["sha256"] = sha256_file(source)
    return frame, metadata
