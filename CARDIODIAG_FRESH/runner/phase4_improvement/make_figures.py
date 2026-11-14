"""Create staged ROC and calibration figures from saved IECV predictions."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4_improvement"
FIGURES = OUT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(OUT / "IECV_FINAL_PREDICTIONS.csv")
    plt.figure(figsize=(7, 6))
    for stage, group in predictions.groupby("stage"):
        fpr, tpr, _ = roc_curve(group.y_true, group.probability)
        plt.plot(fpr, tpr, label=stage)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Staged pooled out-of-centre ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "staged_pooled_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    for stage, group in predictions.groupby("stage"):
        observed, predicted = calibration_curve(group.y_true, group.probability, n_bins=5, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=stage)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Staged pooled calibration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "staged_pooled_calibration.png", dpi=160)
    plt.close()

    manifest_path = OUT / "PHASE4_IMPROVEMENT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generated_utc"] = datetime.now(UTC).isoformat()
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != manifest_path.name]
    manifest["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
