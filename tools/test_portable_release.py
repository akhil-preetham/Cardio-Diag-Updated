from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import shutil

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portable_release as release


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_windows_recorded_paths_rebase_only_into_new_checkout(tmp_path):
    workspace = tmp_path / "new checkout"
    source = workspace / "CARDIODIAG_FRESH/data/source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("new bytes")
    manifest = workspace / "CARDIODIAG_FRESH/results/manifest.json"
    recorded = r"C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\data\source.csv"
    assert release.rebase_path(recorded, workspace, manifest) == source.resolve()
    sibling = workspace / "research_publication_plan/data_screening/example.xlsx"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("new workbook")
    recorded = r"C:\old\workspace\research_publication_plan\data_screening\example.xlsx"
    assert release.rebase_path(recorded, workspace, manifest) == sibling.resolve()


def test_path_traversal_and_unrecognized_absolute_paths_are_rejected(tmp_path):
    manifest = tmp_path / "CARDIODIAG_FRESH/results/m.json"
    for recorded in [r"C:\unknown\secret.txt", "/unrelated/secret.txt", r"CARDIODIAG_FRESH\..\..\outside.txt"]:
        with pytest.raises(ValueError):
            release.rebase_path(recorded, tmp_path, manifest)


def test_windows_relative_manifest_paths_are_platform_neutral(tmp_path):
    expected = tmp_path / "CARDIODIAG_FRESH/results/research_extension_v1/zas/scores.csv"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"frozen data")
    manifest = expected.parent / "MANIFEST.json"
    recorded = r"results\research_extension_v1\zas\scores.csv"
    assert release.rebase_path(recorded, tmp_path, manifest) == expected.resolve()


def test_manifest_relocation_preserves_hashes_and_has_explicit_missing_allowlist(tmp_path):
    directory = tmp_path / "CARDIODIAG_FRESH/results"
    artifact = directory / "saved.csv"
    directory.mkdir(parents=True)
    artifact.write_bytes(b"frozen\r\nbytes\r\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = directory / "m.json"
    write_json(manifest, {"input_hashes": {r"C:\old\CARDIODIAG_FRESH\results\saved.csv": digest}, "outputs": {"saved.csv": digest, "optional/missing.csv": digest}})
    before = manifest.read_bytes()
    with pytest.raises(FileNotFoundError):
        release.verify_manifest(manifest, tmp_path)
    result = release.verify_manifest(manifest, tmp_path, ("CARDIODIAG_FRESH/optional",))
    assert result["hash_entries_checked"] == 2
    assert result["explicitly_omitted"] == ["CARDIODIAG_FRESH/optional/missing.csv"]
    assert manifest.read_bytes() == before
    artifact.write_text("modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        release.verify_manifest(manifest, tmp_path, ("CARDIODIAG_FRESH",))


def test_model_environment_mismatch_prevents_joblib_loading(tmp_path, monkeypatch):
    frozen = tmp_path / "CARDIODIAG_FRESH/results/research_extension_v1/zas/FROZEN_INPUTS.json"
    write_json(frozen, {"versions": {"scikit-learn": "expected-version"}})
    monkeypatch.setattr(release.importlib.metadata, "version", lambda _: "different-version")
    called = []
    import joblib
    monkeypatch.setattr(joblib, "load", lambda _: called.append(True))
    with pytest.raises(RuntimeError, match="Frozen runtime differs"):
        release.load_trusted_source_models(tmp_path)
    assert not called


def test_bundle_hash_mismatch_prevents_unpickling(tmp_path, monkeypatch):
    result = tmp_path / "CARDIODIAG_FRESH/results/research_extension_v1"
    write_json(result / "zas/FROZEN_INPUTS.json", {"versions": {"scikit-learn": "same"}})
    write_json(result / "transport/TRANSPORT_MANIFEST.json", {"environment": {"sklearn": "same"}, "outputs": {"SOURCE_MODELS.joblib": "0" * 64}})
    write_json(result / "transport/PREDICTION_FREEZE.json", {"models_sha256": "0" * 64})
    (result / "transport/SOURCE_MODELS.joblib").write_bytes(b"untrusted payload")
    monkeypatch.setattr(release.importlib.metadata, "version", lambda _: "same")
    called = []
    import joblib
    monkeypatch.setattr(joblib, "load", lambda _: called.append(True))
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        release.load_trusted_source_models(tmp_path)
    assert not called


def test_actual_saved_release_verifies_after_relocation_and_never_falls_back(tmp_path):
    original = release.discover_workspace(Path(__file__))
    result = original / "CARDIODIAG_FRESH/results/research_extension_v1"
    if not (result / "zas/MANIFEST.json").is_file():
        pytest.skip("Actual research artifacts are not included in this test checkout")
    relocated = tmp_path / "relocated checkout"
    manifests = [result / "zas/MANIFEST.json", result / "transport/TRANSPORT_MANIFEST.json", result / "transport/FROZEN_PROTOCOL.json"]
    for path in manifests:
        value = release.read_json(path)
        section = "inputs" if "inputs" in value else "input_hashes"
        paths = [path]
        for key in (section, "outputs"):
            paths.extend(release.rebase_path(name, original, path) for name, digest in value.get(key, {}).items() if isinstance(digest, str) and len(digest) == 64)
        for current in paths:
            destination = relocated / current.relative_to(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, destination)
    checked = release.verify_release(relocated, check_runtime=True)
    assert checked["ok"]
    assert checked["metrics"]["source_prediction_rows"] == 7272
    assert checked["metrics"]["target_cases"] == 377
    assert sum(row["hash_entries_checked"] for row in checked["manifests"]) == 163
    current_source = relocated / "CARDIODIAG_FRESH/data/interim/validated/z_alizadeh_sani.csv"
    current_source.write_bytes(current_source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Hash mismatch in this checkout"):
        release.verify_release(relocated)


def test_scoring_reference_uses_requested_checkout_and_checks_before_models(tmp_path, monkeypatch):
    import portable_predict
    import pandas as pd
    original = release.discover_workspace(Path(__file__))
    relative = Path('CARDIODIAG_FRESH/data/interim/validated/z_alizadeh_sani.csv')
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes((original / relative).read_bytes())
    manifest = tmp_path / 'CARDIODIAG_FRESH/results/research_extension_v1/transport/TRANSPORT_MANIFEST.json'
    write_json(manifest, {'input_hashes': {relative.as_posix(): release.sha256(source)}})
    source.write_bytes(source.read_bytes() + b'\n')
    monkeypatch.setattr(portable_predict, 'assert_frozen_runtime', lambda _: {})
    loaded = []
    monkeypatch.setattr(portable_predict, 'load_trusted_source_models', lambda _: loaded.append(True))
    values = pd.DataFrame([{'Age': 60, 'Sex': 'Male', 'DM': 0, 'HTN': 1, 'Current Smoker': 0}])
    with pytest.raises(ValueError, match='Source reference hash mismatch'):
        portable_predict.score(values, tmp_path)
    assert not loaded


def test_wrong_python_is_rejected_before_reading_files(tmp_path, monkeypatch):
    monkeypatch.setattr(release.sys, 'version_info', (3, 12, 0))
    with pytest.raises(RuntimeError, match='Python 3.11'):
        release.assert_frozen_runtime(tmp_path)
