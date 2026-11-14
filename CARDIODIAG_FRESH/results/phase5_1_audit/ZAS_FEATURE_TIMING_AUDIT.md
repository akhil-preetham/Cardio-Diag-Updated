# ZAS eligible_combined feature-timing audit

## Result traced

The reported logistic/ridge row is reproduced in `results/phase5/zas_track/ZAS_MODEL_BENCHMARK.csv` with 2 matching rows. The exact feature list is hard-coded at `runner/phase5/zas_track/run.py:20-27`; it contains 38 columns and is listed in `ZAS_FEATURE_TIMING_TABLE.csv`. No feature list is inferred from column names.

The included variables are demographics, medical history, symptoms/examination, resting ECG and laboratory values. The code does **not** include EF-TTE, Region RWMA, VHD, Cath, angiographic measurements or vessel-count fields. Those excluded fields are specifically identified by the prior clinical feature audit as downstream or target-derived risks.

## Classification

- **A:** Age and Sex are clearly available before the diagnostic outcome.
- **B:** All other included variables remain timing-uncertain. They are clinically plausible presentation/history/ECG/laboratory variables, but the local source documentation does not establish that they were recorded before diagnostic decisions or before angiographic work-up.
- **C:** No included variable is conclusively shown by the available documentation to be post-diagnostic, but Function Class and some ECG/laboratory fields could become C if source acquisition records show diagnostic-work-up timing.
- **D:** No direct target-derived field is included in `eligible_combined`; Cath and downstream echo/angiographic fields are excluded.

The complete feature-by-feature evidence is in `ZAS_FEATURE_TIMING_TABLE.csv`. Prior repository evidence explicitly classifies EF-TTE and Region RWMA as diagnostic/downstream and Cath as the target; it does not provide sufficient timing evidence to upgrade the included B features to A.
