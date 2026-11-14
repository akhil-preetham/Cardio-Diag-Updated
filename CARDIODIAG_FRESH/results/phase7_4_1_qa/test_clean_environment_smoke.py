import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def test_pinned_core_versions_import():
    expected = {
        "numpy": "2.4.6", "pandas": "3.1.1", "scipy": "1.17.1",
        "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "pytest": "9.2.0",
    }
    assert {name: version(name) for name in expected} == expected


def test_saved_repeat_metrics_readable_without_fitting():
    table = pd.read_csv(ROOT / "results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv")
    selection = table[table.model_id == "selection_aware_procedure"]
    assert len(selection) == 3
    assert selection["n"].eq(303).all()
    assert selection["events"].eq(216).all()


def test_corrected_figure_files_are_valid():
    figure_dir = ROOT / "paper/phase7_4_1_final_manuscript/FIGURES"
    png = Image.open(figure_dir / "FIGURE2_PHASE6_REPEAT_AUC.png")
    assert png.width >= 1800 and png.height >= 1000
    pdf = (figure_dir / "FIGURE2_PHASE6_REPEAT_AUC.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-")


def test_phase7_4_1_and_phase7_3_hash_manifests_verify():
    verifier_path = ROOT / "results/phase7_4_1_qa/verify_manifest.py"
    spec = importlib.util.spec_from_file_location("phase7_4_1_verify_manifest", verifier_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = verifier.verify()
    assert result["ok"], json.dumps(result, indent=2)
    assert result["historical_hashes_checked"] == 74
