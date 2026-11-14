from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

try:
    from .common import load_config, resolve, utc_now, write_json
except ImportError:
    from common import load_config, resolve, utc_now, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(name: str, path: Path, target: str) -> dict:
    if not path.exists():
        return {"dataset": name, "path": str(path), "status": "missing"}
    frame = pd.read_csv(path)
    identifier_columns = [
        column for column in frame.columns
        if column.lower() in {"id", "patientid", "patient_id", "subject_id"}
        or column.lower().endswith("_id")
    ]
    numeric_ranges = {}
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        numeric_ranges[column] = {
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "unique": int(values.nunique()),
        }
    return {
        "dataset": name,
        "path": str(path),
        "sha256": file_sha256(path),
        "status": "validated",
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "target": target,
        "target_exists": target in frame.columns,
        "target_values": frame[target].value_counts(dropna=False).to_dict()
        if target in frame.columns else {},
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "identifier_columns": identifier_columns,
        "numeric_ranges": numeric_ranges,
        "invalid_target_values": (
            sorted(set(pd.to_numeric(frame[target], errors="coerce").dropna()) - {0, 1})
            if target in frame.columns else []
        ),
    }


def main(config_file: str) -> None:
    cfg, root = load_config(Path(config_file).resolve())
    output = resolve(root, cfg["output_root"]) / "audit"
    output.mkdir(parents=True, exist_ok=True)
    targets = {"five_states": "HeartDisease", "mendeley": "target"}
    results = [
        inspect_dataset(name, resolve(root, cfg["datasets"][name]), target)
        for name, target in targets.items()
    ]
    report = {"created_utc": utc_now(), "datasets": results}
    write_json(output / "dataset_validation.json", report)
    rows = []
    for item in results:
        rows.append({
            "dataset": item["dataset"], "path": item["path"], "status": item["status"],
            "sha256": item.get("sha256", ""), "rows": item.get("rows", ""),
            "columns": item.get("columns", ""), "target": item["target"],
            "target_exists": item.get("target_exists", False),
            "duplicate_rows": item.get("duplicate_rows", ""),
            "identifier_columns": "|".join(item.get("identifier_columns", [])),
        })
    pd.DataFrame(rows).to_csv(output / "dataset_validation.csv", index=False)
    print(f"Validated {len(results)} datasets: {output / 'dataset_validation.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/dataset_validation.py config/runner_config.json")
    main(sys.argv[1])
