from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cardiodiag.calibration import calibration_summary
from cardiodiag.clinical_utility import net_benefit
from cardiodiag.statistics import delong_auc_test, paired_bootstrap_difference


def main(root: Path):
    base = root / "results/ensemble_experiments"
    output = root / "results/ensemble_postprocessing"
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    calibration = []
    utility = []
    p_values = []
    for dataset in ["five_states", "mendeley"]:
        for track in ["author_faithful", "compatibility_adapted", "corrected"]:
            paths = {}
            for method in ["ovec", "osec"]:
                path = base / dataset / track / method / "predictions.csv"
                if path.exists():
                    paths[method] = pd.read_csv(path)
                    row = paths[method]
                    calibration.append({"dataset": dataset, "track": track, "method": method, **calibration_summary(row.y_true, row.probability)})
                    for threshold in np.arange(0.05, 1.0, 0.05):
                        utility.append({"dataset": dataset, "track": track, "method": method, "threshold": round(float(threshold), 2), "net_benefit": net_benefit(row.y_true, row.probability, float(threshold))})
            if set(paths) != {"ovec", "osec"}:
                continue
            first, second = paths["ovec"], paths["osec"]
            if not np.array_equal(first.y_true.to_numpy(), second.y_true.to_numpy()):
                raise RuntimeError(f"Unpaired prediction vectors for {dataset}/{track}")
            difference = paired_bootstrap_difference(
                first.y_true.to_numpy(),
                first.prediction.to_numpy(),
                second.prediction.to_numpy(),
                metric=lambda y, p: np.mean(y == p),
                n_bootstrap=2000,
            )
            statistic, p_value = wilcoxon(
                first.y_true.to_numpy() == first.prediction.to_numpy(),
                second.y_true.to_numpy() == second.prediction.to_numpy(),
                zero_method="wilcox",
            )
            delong = delong_auc_test(first.y_true.to_numpy(), first.probability.to_numpy(), second.probability.to_numpy())
            comparisons.append({"dataset": dataset, "track": track, "comparison": "OVEC_minus_OSEC", "wilcoxon_statistic": float(statistic), "wilcoxon_p_value": float(p_value), **difference, **{f"delong_{key}": value for key, value in delong.items()}})
            p_values.append(float(p_value))
    if p_values:
        order = np.argsort(p_values)
        adjusted = np.empty(len(p_values))
        for rank, index in enumerate(order):
            adjusted[index] = min(1.0, (len(p_values) - rank) * p_values[index])
        adjusted = np.maximum.accumulate(adjusted[order])[np.argsort(order)]
        for row, value in zip(comparisons, adjusted):
            row["holm_adjusted_p_value"] = float(value)
    pd.DataFrame(comparisons).to_csv(output / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration_summary.csv", index=False)
    pd.DataFrame(utility).to_csv(output / "decision_curve_net_benefit.csv", index=False)
    (output / "analysis_manifest.json").write_text(json.dumps({"paired_tests": "identical persisted test vectors only", "multiple_comparison_correction": "Holm applied to Wilcoxon p-values", "de_long": "paired implementation executed", "clinical_utility": "net benefit threshold sweep from 0.05 to 0.95"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
