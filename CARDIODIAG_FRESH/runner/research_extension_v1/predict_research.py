"""Validated batch scoring for the trusted local five-core RESEARCH models.

No clinical decisions or diagnosis labels are produced. The model bundle must
match the completed local transport experiment's prediction freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from runner.research_extension_v1.transport import (
    ARMS,
    CORE,
    OUT,
    SOURCE,
    normalize_core,
)


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CORE) - set(frame.columns)
    extra = set(frame.columns) - set(CORE)
    if missing or extra:
        raise ValueError(f"Expected exactly five core fields; missing={sorted(missing)}, extra={sorted(extra)}")
    if frame.empty:
        raise ValueError("Input contains no records")
    normalized = normalize_core(frame, source=False)
    if normalized.isna().all(axis=1).any():
        raise ValueError("A record with every predictor missing cannot be scored")
    if np.isinf(normalized.to_numpy()).any():
        raise ValueError("Infinite predictor values are not supported")
    return normalized


def load_trusted_models() -> dict:
    freeze_path = OUT / "PREDICTION_FREEZE.json"
    models_path = OUT / "SOURCE_MODELS.joblib"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(models_path.read_bytes()).hexdigest()
    if actual != freeze["models_sha256"]:
        raise ValueError("Model bundle hash differs from its completed experiment freeze")
    # Never load arbitrary user-supplied pickle/joblib paths.
    return joblib.load(models_path)


def score(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = validate_input(frame)
    models = load_trusted_models()
    source = normalize_core(pd.read_csv(SOURCE), source=True)
    low, high = source.Age.min(), source.Age.max()
    result = pd.DataFrame({"input_row_index": np.arange(len(frame))})
    for arm in ARMS:
        result[arm] = models[arm].predict_proba(normalized)[:, 1]
    result["missing_predictor_count"] = normalized.isna().sum(axis=1).to_numpy()
    result["age_outside_source_range"] = (~normalized.Age.between(low, high) & normalized.Age.notna()).to_numpy()
    result["scope"] = "retrospective research only; no patient-specific reliability or clinical-use validation"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing scoring output")
    result = score(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} research records; output={args.output.resolve()}")


if __name__ == "__main__":
    main()
