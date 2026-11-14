# Table 5. Paired UCI Stage 3-to-Stage 4 metric differences

| Comparison | Metric | Difference (later−earlier) | 95% interval low | 95% interval high | Bootstrap draws | Uncertainty unit |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 3 → Stage 4 | ROC-AUC | 0.0235 | 0.0061 | 0.0443 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Average precision | 0.0454 | 0.0196 | 0.0706 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Brier | -0.0099 | -0.0186 | -0.0015 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Log loss | -0.0278 | -0.0514 | -0.0056 | 400 | Paired record bootstrap; centre/source-row paired |

Intervals are percentile intervals from 400 paired source-record bootstrap draws using the same sampled rows for both stages; they are not patient-cluster or centre-sampling intervals.
