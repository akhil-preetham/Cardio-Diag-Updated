"""Create Phase 4 figures from saved predictions and refresh the output manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4_uci_multicentre"
FIGURES = OUT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(OUT / "OUT_OF_CENTRE_PREDICTIONS.csv")
    strict = predictions[predictions.feature_set == "strict"]
    plt.figure(figsize=(7, 6))
    for model, group in strict.groupby("model"):
        fpr, tpr, _ = roc_curve(group.y_true, group.probability)
        plt.plot(fpr, tpr, label=model)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("UCI four-centre pooled out-of-centre ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "strict_pooled_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    for centre, group in strict[strict.model == "logistic"].groupby("held_out_centre"):
        observed, predicted = calibration_curve(group.y_true, group.probability, n_bins=5, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=centre)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Strict logistic calibration by held-out centre")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "strict_logistic_calibration_by_centre.png", dpi=160)
    plt.close()

    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "PHASE4_REPRODUCIBILITY_MANIFEST.json"]
    manifest = json.loads((OUT / "PHASE4_REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["generated_utc"] = datetime.now(UTC).isoformat()
    manifest["environment"] = {"python": sys.version, "platform": platform.platform()}
    manifest["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    (OUT / "PHASE4_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
