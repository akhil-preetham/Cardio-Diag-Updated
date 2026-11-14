"""Verify Phase 7.4.1 artifact hashes and the Phase 7.3 preservation manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "results/phase7_4_1_qa/PHASE7_4_1_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["artifacts_sha256"].items():
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            failures.append(f"path escapes repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    historical_path = ROOT / manifest["historical_preservation"]["phase7_3_manifest"]
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_count = 0
    for section in ("inputs", "outputs"):
        for relative, expected in historical.get(section, {}).items():
            historical_count += 1
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"historical missing: {relative}")
            elif sha256(path) != expected:
                failures.append(f"historical hash mismatch: {relative}")

    return {
        "ok": not failures,
        "artifact_hashes_checked": len(manifest["artifacts_sha256"]),
        "historical_hashes_checked": historical_count,
        "failures": failures,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
