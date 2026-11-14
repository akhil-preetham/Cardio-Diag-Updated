"""Read-only hash and primary-metric verification of the completed extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent.resolve()
RESULTS = ROOT / "results/research_extension_v1"


def resolve_input(key: str, manifest_path: Path) -> Path:
    path = Path(key)
    if not path.is_absolute():
        candidates = [ROOT / path, manifest_path.parent / path]
        path = next((p for p in candidates if p.is_file()), candidates[0])
    path = path.resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"Manifest target leaves the intended workspace: {path}")
    return path


def verify_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked = []
    input_section = "inputs" if "inputs" in manifest else "input_hashes"
    for section in (input_section, "outputs"):
        for key, expected in manifest.get(section, {}).items():
            if not isinstance(expected, str) or len(expected) != 64:
                continue
            target = resolve_input(key, path)
            if not target.is_file():
                raise FileNotFoundError(target)
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Hash mismatch: {target}")
            checked.append({"section": section, "path": str(target), "sha256": actual})
    if not checked:
        raise ValueError(f"No hash entries found: {path}")
    return {"manifest": str(path.relative_to(ROOT)), "hash_entries_checked": len(checked), "all_match": True}


def simple_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {"roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier": np.mean((p - y) ** 2), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])}


def verify() -> dict:
    manifests = [verify_manifest(RESULTS / "zas/MANIFEST.json"), verify_manifest(RESULTS / "transport/TRANSPORT_MANIFEST.json")]
    pred = pd.read_csv(RESULTS / "zas/OOF_PREDICTIONS.csv")
    saved = pd.read_csv(RESULTS / "zas/REPEAT_METRICS.csv")
    source = pd.read_csv(ROOT / "data/interim/validated/z_alizadeh_sani.csv")
    source_y = source.Cath.map({"Cad": 1, "Normal": 0}).to_numpy()
    groups = 0
    for (repeat, model), g in pred.groupby(["repeat", "model_id"]):
        if len(g) != 303 or set(g.row_index) != set(range(303)):
            raise ValueError("Incorrect source OOF coverage")
        if not np.array_equal(g.y_true.to_numpy(), source_y[g.row_index.to_numpy()]):
            raise ValueError("Source labels disagree")
        row = saved[(saved.repeat == repeat) & (saved.model_id == model)].iloc[0]
        for metric, value in simple_metrics(g.y_true, g.probability).items():
            if not np.isclose(value, row[metric], rtol=0, atol=1e-10):
                raise ValueError(f"Source metric mismatch: {repeat}/{model}/{metric}")
        groups += 1
    if groups != 24:
        raise ValueError("Expected 8 source procedures across 3 repeats")
    target = pd.read_csv(RESULTS / "transport/TARGET_PREDICTIONS.csv")
    metrics = pd.read_csv(RESULTS / "transport/TARGET_METRICS.csv")
    for row in metrics.itertuples():
        keep = ~target.boundary_ambiguous26_50 if row.endpoint == "exclude_ambiguous26_50" else np.ones(len(target), dtype=bool)
        y = target.y_base_gt50.copy()
        if row.endpoint == "upper_assignment26_50_positive":
            y[target.boundary_ambiguous26_50] = 1
        expected = simple_metrics(y[keep], target.loc[keep, row.arm])
        for name, value in expected.items():
            saved_name = "brier_score" if name == "brier" else name
            if not np.isclose(value, getattr(row, saved_name), rtol=0, atol=1e-10):
                raise ValueError(f"Target metric mismatch: {row.endpoint}/{row.arm}/{name}")
    return {"ok": True, "manifests": manifests, "source_repeat_procedure_groups": groups,
            "source_prediction_rows_including_historical": len(pred), "target_cases": len(target),
            "target_endpoint_procedure_groups": len(metrics), "new_predictive_fits": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
