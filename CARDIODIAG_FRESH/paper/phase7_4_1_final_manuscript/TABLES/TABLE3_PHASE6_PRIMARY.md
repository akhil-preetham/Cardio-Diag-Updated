# Table 3. Phase 6 primary ZAS results by complete-repeat out-of-fold set

| Repeat | Rows each | Events | Prevalence | Selection ROC-AUC | V3-style ROC-AUC | Paired delta ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 303 | 216 | 0.7129 | 0.9134 | 0.7370 | 0.1764 |
| 2 | 303 | 216 | 0.7129 | 0.8999 | 0.7420 | 0.1579 |
| 3 | 303 | 216 | 0.7129 | 0.9176 | 0.7300 | 0.1877 |

Paired mean AUC difference = 0.1740; descriptive split-variation t interval 0.1367–0.2113 (three overlapping repeat summaries; not patient/population uncertainty).

Secondary performance and calibration summaries (each repeat includes all 303 source rows):

| Repeat | Procedure | Average precision | Brier | Log loss | Sensitivity | Specificity | Calibration intercept | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | V3-style comparator | 0.8528 | 0.1741 | 0.5257 | 0.6713 | 0.7126 | -0.0616 | 1.0781 |
| 1 | Selection-aware | 0.9602 | 0.1075 | 0.3447 | 0.8657 | 0.7931 | 0.1218 | 0.8693 |
| 2 | V3-style comparator | 0.8519 | 0.1734 | 0.5251 | 0.6852 | 0.7126 | 0.0222 | 0.9781 |
| 2 | Selection-aware | 0.9537 | 0.1137 | 0.3666 | 0.8148 | 0.8161 | 0.2055 | 0.8097 |
| 3 | V3-style comparator | 0.8359 | 0.1763 | 0.5349 | 0.7130 | 0.7011 | 0.0173 | 0.9575 |
| 3 | Selection-aware | 0.9580 | 0.1016 | 0.3289 | 0.8148 | 0.8851 | 0.1927 | 0.9247 |

AP reference prevalence = 0.7129 for each repeat. Calibration estimates are limited; thresholds were selected on training OOF predictions using Youden J and have no clinical interpretation.
