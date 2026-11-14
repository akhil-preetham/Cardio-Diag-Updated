"""Strict-schema research scoring for the portable, pinned local release."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from portable_release import PROJECT, assert_frozen_runtime, discover_workspace, load_trusted_source_models, read_json, rebase_path, sha256


def score(frame, workspace: Path):
    # The gate precedes joblib loading and calls into the frozen source code.
    assert_frozen_runtime(workspace)
    sys.path.insert(0, str(workspace / PROJECT))
    from runner.research_extension_v1.predict_research import validate_input
    from runner.research_extension_v1.transport import ARMS, normalize_core
    import numpy as np
    import pandas as pd

    if frame.columns.duplicated().any():
        raise ValueError("Duplicate predictor column names are not supported")
    values = validate_input(frame)
    source_path = workspace / PROJECT / 'data/interim/validated/z_alizadeh_sani.csv'
    manifest_path = workspace / PROJECT / 'results/research_extension_v1/transport/TRANSPORT_MANIFEST.json'
    manifest = read_json(manifest_path)
    expected = [digest for recorded, digest in manifest['input_hashes'].items()
                if rebase_path(recorded, workspace, manifest_path) == source_path.resolve()]
    if len(expected) != 1 or sha256(source_path) != expected[0]:
        raise ValueError('Source reference hash mismatch in this checkout')
    source = normalize_core(pd.read_csv(source_path), source=True)
    models = load_trusted_source_models(workspace)
    result = pd.DataFrame({"input_row_index": np.arange(len(values))})
    for arm in ARMS:
        result[arm] = models[arm].predict_proba(values)[:, 1]
    result["missing_predictor_count"] = values.isna().sum(axis=1).to_numpy()
    result["age_outside_source_range"] = (values.Age.notna() & ~values.Age.between(source.Age.min(), source.Age.max())).to_numpy()
    result["scope"] = "retrospective research only; no patient-specific reliability or clinical-use validation"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.root.resolve() if args.root else discover_workspace()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing scoring output")
    import pandas as pd
    predictions = score(pd.read_csv(args.input), workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        predictions.to_csv(stream, index=False)
    print(f"Scored {len(predictions)} research records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
