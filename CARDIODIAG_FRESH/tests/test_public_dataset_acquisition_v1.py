import pandas as pd

from runner.public_dataset_acquisition_v1.audit import (
    angiocad_patient_label,
    binary_num_target,
    duplicate_summary,
    numeric_stenosis,
)


def test_uci_binary_target_maps_zero_vs_one_to_four():
    result = binary_num_target(pd.Series([0, 1, 2, 4, None], dtype="Float64"))
    assert result.iloc[:4].tolist() == [0, 1, 1, 1]
    assert pd.isna(result.iloc[4])


def test_numeric_stenosis_handles_normal_and_ranges():
    assert numeric_stenosis("NL") == 0
    assert numeric_stenosis("26-50") == 26
    assert numeric_stenosis("100") == 100


def test_angiocad_patient_label_uses_lower_bound_and_not_gensini():
    labels = pd.DataFrame({"ID": [1, 2], "LM": ["NL", "51-75"], **{c: ["NL", "NL"] for c in ["Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig", "Prox LCX", "Mid LCX", "Dist LCX", "OM", "Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"]}})
    result = angiocad_patient_label(labels)
    assert result["binary_stenosis_ge_50_lower_bound"].tolist() == [0, 1]


def test_duplicate_summary_counts_subsequent_occurrences():
    frame = pd.DataFrame({"patient": [1, 1, 2], "target": [0, 0, 1]})
    assert duplicate_summary(frame, ["patient", "target"]) == {"rows": 3, "unique_rows": 2, "duplicate_occurrences": 1, "duplicate_groups": 1}
