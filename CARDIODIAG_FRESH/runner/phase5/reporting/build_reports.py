"""Build Phase 5 protocol, manuscript evidence and reproducibility reports."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    reports = OUT / "reports"
    manifests = OUT / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    text = {
        "PHASE5_RESEARCH_PROTOCOL.md": "# Phase 5 research protocol\n\nPhase 5 evaluates the frozen ZAS V3 five-predictor model, bounded ZAS feature-group/model alternatives, UCI Stage 2 and Stage 4 centre-held-out alternatives, and reliability diagnostics. Historical outputs remain immutable. Model selection is prespecified around discrimination, calibration, centre consistency, timing, missingness, duplicate sensitivity and reproducibility.\n",
        "PHASE5_METHODS.md": "# Phase 5 methods\n\nAll preprocessing, imputation, encoding, scaling, spline transformation and model tuning occur inside development folds. ZAS uses repeated stratified nested validation. UCI uses leave-one-centre-out outer validation with development-only inner selection. Saved Phase 4.3 predictions are used for independent staged reliability comparisons. Record-level bootstrap intervals do not represent patient-cluster or centre-sampling uncertainty.\n",
        "PHASE5_LIMITATIONS.md": "# Phase 5 limitations\n\nZAS has 303 patients, incomplete feature timing and no patient identifier. UCI centre composition and missingness differ substantially, and Switzerland has sparse negatives. Optional GAM, EBM, CatBoost and Bayesian packages are unavailable. Public candidates have endpoint, modality, access or independence limitations. No direct independent external validation cohort was established.\n",
        "TRIPOD_AI_CHECKLIST.md": "# TRIPOD+AI evidence checklist\n\nThis package documents target population, outcome definitions, prediction time, predictors, missingness, preprocessing, nested validation, calibration, discrimination, centre-specific results, uncertainty and limitations. Items requiring external validation, patient-level linkage, clinical utility and deployment safety remain unresolved rather than being marked complete.\n",
        "PHASE5_MANUSCRIPT_EVIDENCE.md": "# Phase 5 manuscript evidence\n\nHistorical, independently reproduced and newly generated results are separated by namespace. The strongest new evidence is the modest paired Stage 3 to Stage 4 improvement and the larger Stage 1 to Stage 2 symptom-stage gain. No result supports clinical readiness or universal transportability. The UCI study is not external validation of ZAS V3.\n",
        "PHASE5_RESULTS.md": "# Phase 5 results\n\nResults are stored in `results/phase5/zas_track`, `results/phase5/uci_track`, `results/phase5/reliability` and `results/phase5/data_registry`. Model tables include out-of-fold predictions, metrics, calibration, stability and compatibility decisions. Optional unavailable model families are explicitly recorded.\n",
        "PHASE5_FINAL_DECISION.md": "# Phase 5 final decision: CONDITIONAL GO\n\nProceed with research and manuscript development, but do not claim a clinically selected model, direct external validation, patient-level safety or universal transportability. The frozen ZAS V3 baseline remains the reference. Stage 2 is the most defensible expanded early-presentation UCI task; Stage 4 is a separate exercise-test task. A compatible independent cohort remains required.\n",
    }
    for name, body in text.items():
        (reports / name).write_text(body, encoding="utf-8")
    (reports / "PHASE5_SUMMARY.csv").write_text(
        "track,reference,decision,evidence_status\\n"
        "ZAS,V3 five predictors,reference preserved,new bounded nested results\\n"
        "UCI,Stage 2 and Stage 4,conditional research only,centre-held-out results\\n"
        "external_validation,None compatible direct cohort,not established,provenance audit only\\n",
        encoding="utf-8",
    )
    outputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in OUT.rglob("*")
        if path.is_file() and path.name != "PHASE5_MANIFEST.json"
    }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": 5107,
        "historical_namespaces_preserved": True,
        "outputs": outputs,
    }
    (manifests / "PHASE5_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
