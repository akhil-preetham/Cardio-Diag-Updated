"""Generate centre-specific calibration plots for the Phase 4.3 audit."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4_validation_audit"
FIGURES = OUT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(OUT / "CENTRE_CALIBRATION_CURVES.csv")
    for (stage, centre), group in curves.groupby(["stage", "held_out_centre"]):
        figure, axis = plt.subplots(figsize=(6, 5))
        for bins, binned in group.groupby("bins"):
            axis.plot(binned.predicted, binned.observed, marker="o", label=f"{bins} bins")
        axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed fraction positive")
        axis.set_title(f"{stage} - {centre}")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(FIGURES / f"{stage}__{centre.replace(' ', '_')}.png", dpi=160)
        plt.close(figure)
    manifest_path = OUT / "PHASE4_3_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generated_utc"] = datetime.now(UTC).isoformat()
    manifest["outputs"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in OUT.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
