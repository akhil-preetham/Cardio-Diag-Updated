"""Portable read-only verification for preserved CARDIODIAG research artifacts.

Historical manifests are byte-preserved. Recorded Windows paths are rebased into
the current checkout, never opened at their original absolute locations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys


PROJECT = "CARDIODIAG_FRESH"
SIBLING = "research_publication_plan"


def discover_workspace(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / PROJECT).is_dir():
            return candidate
    raise FileNotFoundError("Pass --root for a checkout containing CARDIODIAG_FRESH")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase_path(recorded: str, workspace: Path, manifest: Path) -> Path:
    """Resolve only inside this checkout, including on non-Windows hosts."""
    workspace = workspace.resolve()
    manifest = manifest.resolve()
    text = recorded.replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    folded = [part.casefold() for part in parts]
    candidates = []
    for anchor in (PROJECT, SIBLING):
        if anchor.casefold() in folded:
            index = folded.index(anchor.casefold())
            candidates = [workspace / anchor / Path(*parts[index + 1:])]
            break
    if not candidates:
        if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
            raise ValueError(f"Unrecognized recorded absolute path: {recorded}")
        relative = Path(*parts)
        candidates = [workspace / PROJECT / relative, manifest.parent / relative]
        if len(parts) == 1:
            candidates.reverse()
    for candidate in candidates:
        if not candidate.resolve().is_relative_to(workspace):
            raise ValueError(f"Manifest path escapes checkout: {recorded}")
    # Existence selection considers only paths under the new checkout.
    return next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())


def verify_manifest(path: Path, workspace: Path, allow_missing: tuple[str, ...] = ()) -> dict:
    manifest = read_json(path)
    checked, omissions = [], []
    input_key = "inputs" if isinstance(manifest.get("inputs"), dict) else "input_hashes"
    for section in (input_key, "outputs"):
        for recorded, expected in manifest.get(section, {}).items():
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                continue
            current = rebase_path(recorded, workspace, path)
            relative = current.relative_to(workspace.resolve()).as_posix()
            if not current.is_file():
                if any(relative == prefix.rstrip("/") or relative.startswith(prefix.rstrip("/") + "/") for prefix in allow_missing):
                    omissions.append(relative)
                    continue
                raise FileNotFoundError(f"Missing manifest artifact in this checkout: {relative}")
            if sha256(current) != expected.lower():
                raise ValueError(f"Hash mismatch in this checkout: {relative}")
            checked.append({"section": section, "path": relative})
    if not checked:
        raise ValueError(f"No supported hash entries in {path.name}")
    return {"manifest": path.relative_to(workspace).as_posix(), "hash_entries_checked": len(checked),
            "all_present_entries_match": True, "explicitly_omitted": omissions}


def assert_frozen_runtime(workspace: Path) -> dict:
    """Do this before loading the trusted bundle or resuming any checkpoints."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Frozen runtime requires Python 3.11')
    frozen = read_json(workspace / PROJECT / "results/research_extension_v1/zas/FROZEN_INPUTS.json")
    expected = frozen["versions"]
    actual, mismatches = {}, {}
    for distribution, recorded in expected.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = "not installed"
        actual[distribution] = observed
        if observed != recorded:
            mismatches[distribution] = {"expected": recorded, "actual": observed}
    if mismatches:
        raise RuntimeError("Frozen runtime differs; install the pinned release requirements before loading models or resuming checkpoints: " + json.dumps(mismatches, sort_keys=True))
    return actual


def load_trusted_source_models(workspace: Path) -> dict:
    """No user-controlled pickle path; validate environment and bytes first."""
    importlib.metadata.version("scikit-learn")
    assert_frozen_runtime(workspace)
    directory = workspace / PROJECT / "results/research_extension_v1/transport"
    transport = read_json(directory / "TRANSPORT_MANIFEST.json")
    expected_sklearn = transport["environment"]["sklearn"]
    observed_sklearn = importlib.metadata.version("scikit-learn")
    if observed_sklearn != expected_sklearn:
        raise RuntimeError(f"Saved model needs scikit-learn {expected_sklearn}; observed {observed_sklearn}")
    freeze = read_json(directory / "PREDICTION_FREEZE.json")
    path = directory / "SOURCE_MODELS.joblib"
    digest = sha256(path)
    if digest != freeze["models_sha256"] or digest != transport["outputs"][path.name]:
        raise ValueError("Trusted source model bundle hash mismatch")
    import joblib
    return joblib.load(path)


def verify_metrics(workspace: Path) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    project = workspace / PROJECT
    result = project / "results/research_extension_v1"
    source = pd.read_csv(project / "data/interim/validated/z_alizadeh_sani.csv")
    source_y = source.Cath.map({"Cad": 1, "Normal": 0}).to_numpy()
    predictions = pd.read_csv(result / "zas/OOF_PREDICTIONS.csv")
    summary = pd.read_csv(result / "zas/REPEAT_METRICS.csv")

    def metrics(y, probability):
        y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Invalid probability")
        return {"roc_auc": roc_auc_score(y, probability), "average_precision": average_precision_score(y, probability),
                "brier": float(np.mean((probability - y)**2)), "log_loss": log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1])}

    group_count = 0
    for (repeat, model), group in predictions.groupby(["repeat", "model_id"]):
        if len(group) != 303 or set(group.row_index) != set(range(303)):
            raise ValueError("OOF source coverage/uniqueness mismatch")
        if not np.array_equal(group.y_true.to_numpy(), source_y[group.row_index.to_numpy()]):
            raise ValueError("Source labels differ")
        saved = summary[(summary.repeat == repeat) & (summary.model_id == model)]
        if len(saved) != 1:
            raise ValueError("Expected exactly one saved complete-repeat summary")
        for key, actual in metrics(group.y_true, group.probability).items():
            if not np.isclose(actual, saved.iloc[0][key], atol=1e-10, rtol=0):
                raise ValueError(f"Source metric differs: {model}/{repeat}/{key}")
        group_count += 1
    if group_count != 24 or len(predictions) != 7272:
        raise ValueError("Expected eight procedures by three repeats")

    target = pd.read_csv(result / "transport/TARGET_PREDICTIONS.csv")
    target_summary = pd.read_csv(result / "transport/TARGET_METRICS.csv")
    if len(target) != 377 or target.target_case_id.nunique() != 377:
        raise ValueError("Target case coverage mismatch")
    for saved in target_summary.itertuples():
        y = target.y_base_gt50.copy()
        keep = np.ones(len(y), dtype=bool)
        if saved.endpoint == "exclude_ambiguous26_50":
            keep = ~target.boundary_ambiguous26_50.to_numpy(dtype=bool)
        elif saved.endpoint == "upper_assignment26_50_positive":
            y[target.boundary_ambiguous26_50] = 1
        elif saved.endpoint != "base_gt50":
            raise ValueError("Unknown target endpoint")
        if len(y[keep]) != saved.n or int(y[keep].sum()) != saved.events:
            raise ValueError("Target endpoint size/count mismatch")
        for key, actual in metrics(y[keep], target.loc[keep, saved.arm]).items():
            field = "brier_score" if key == "brier" else key
            if not np.isclose(actual, getattr(saved, field), atol=1e-10, rtol=0):
                raise ValueError(f"Target metric differs: {saved.arm}/{saved.endpoint}/{key}")
    if len(target_summary) != 6:
        raise ValueError("Expected six procedure/endpoint summaries")
    return {"source_repeat_procedure_groups": group_count, "source_prediction_rows": len(predictions),
            "target_cases": len(target), "target_endpoint_procedure_groups": len(target_summary), "new_predictive_fits": 0}


def verify_release(workspace: Path, allow_missing: tuple[str, ...] = (), check_runtime: bool = False, metrics: bool = True) -> dict:
    workspace = workspace.resolve()
    result = workspace / PROJECT / "results/research_extension_v1"
    records = [verify_manifest(result / "zas/MANIFEST.json", workspace, allow_missing),
               verify_manifest(result / "transport/TRANSPORT_MANIFEST.json", workspace, allow_missing),
               verify_manifest(result / "transport/FROZEN_PROTOCOL.json", workspace, allow_missing)]
    response = {"ok": True, "workspace": str(workspace), "manifests": records,
                "scope": "Listed preserved manifest mappings; relocation does not alter recorded historical bytes"}
    if check_runtime:
        response["runtime"] = assert_frozen_runtime(workspace)
    if metrics:
        response["metrics"] = verify_metrics(workspace)
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--allow-missing-prefix", action="append", default=[], help="Explicit checkout-relative legacy prefix; never suppresses mismatching existing files")
    parser.add_argument("--hashes-only", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    workspace = args.root.resolve() if args.root else discover_workspace()
    print(json.dumps(verify_release(workspace, tuple(args.allow_missing_prefix), args.check_runtime, not args.hashes_only), indent=2))


if __name__ == "__main__":
    main()
