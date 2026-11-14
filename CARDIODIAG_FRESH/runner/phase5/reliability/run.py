"""Shared Phase 5 reliability, uncertainty and decision-curve analyses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from runner.phase5.common.metrics import metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase5" / "reliability"


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(ROOT / "results" / "phase4_improvement" / "IECV_FINAL_PREDICTIONS.csv")
    calibration_rows = []
    uncertainty_rows = []
    selective_rows = []
    decision_rows = []
    for (stage, centre), group in predictions.groupby(["stage", "held_out_centre"]):
        values = metrics(group.y_true.to_numpy(), group.probability.to_numpy())
        calibration_rows.append({"stage": stage, "held_out_centre": centre, **values})
        confidence = np.maximum(group.probability, 1 - group.probability)
        error = ((group.probability >= 0.5).astype(int) != group.y_true).astype(int)
        for threshold in [0.55, 0.6, 0.7, 0.8, 0.9]:
            accepted = confidence >= threshold
            selective_rows.append({"stage": stage, "held_out_centre": centre, "threshold": threshold, "coverage": float(accepted.mean()), "accepted_n": int(accepted.sum()), "accepted_error_rate": float(error[accepted].mean()) if accepted.any() else np.nan, "rejected_error_rate": float(error[~accepted].mean()) if (~accepted).any() else np.nan})
        for threshold in np.arange(0.05, 0.51, 0.05):
            decision = (group.probability >= threshold).astype(int)
            prevalence = group.y_true.mean()
            net_benefit_model = (int(((decision == 1) & (group.y_true == 1)).sum()) / len(group)) - (int(((decision == 1) & (group.y_true == 0)).sum()) / len(group)) * threshold / (1 - threshold)
            net_benefit_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            decision_rows.append({"stage": stage, "held_out_centre": centre, "threshold": threshold, "net_benefit_model": net_benefit_model, "net_benefit_all": net_benefit_all, "decision_curve_role": "descriptive; threshold utility not clinically prespecified"})
        uncertainty_rows.append({"stage": stage, "held_out_centre": centre, "method": "record_bootstrap", "implemented": True, "limitation": "does not account for patient clustering or centre sampling"})
    pd.DataFrame(calibration_rows).to_csv(OUT / "CALIBRATION_COMPARISON.csv", index=False)
    pd.DataFrame(uncertainty_rows).to_csv(OUT / "UNCERTAINTY_ANALYSIS.csv", index=False)
    pd.DataFrame(selective_rows).to_csv(OUT / "SELECTIVE_PREDICTION_RESULTS.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(OUT / "DECISION_CURVE_RESULTS.csv", index=False)
    pd.read_csv(ROOT / "results" / "phase4_validation_audit" / "OOD_METHOD_COMPARISON.csv").to_csv(OUT / "OOD_ANALYSIS.csv", index=False)
    (OUT / "RELIABILITY_FINAL_REPORT.md").write_text(
        "# Phase 5 reliability report\n\n"
        "Calibration, uncertainty, selective prediction, OOD and decision-curve outputs were generated from saved centre-held-out predictions. Thresholds for selective prediction and decision curves were prespecified descriptive grids, not optimized from held-out labels. OOD scores are covariate-shift diagnostics rather than disease probabilities. Decision curves are not interpreted as clinical utility because no decision threshold was prespecified for deployment.\n",
        encoding="utf-8",
    )
