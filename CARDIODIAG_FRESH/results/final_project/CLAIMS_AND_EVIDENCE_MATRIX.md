# Claims and evidence matrix

| Proposed claim | Evidence | Status / permitted wording |
|---|---|---|
| Phase 5 eligible-combined ZAS ROC-AUC is 0.928 | Saved Phase 5 prediction CSV independently aggregated and benchmark reconciled | Supported as exploratory descriptive result; specify 303 source rows and post-hoc selection among 36 combinations |
| No direct target field is in the selected predictor group | Selected group membership and feature audit | Supported for direct target leakage; does not rule out temporal/proxy leakage |
| Logistic and Ridge independently confirm the result | Constructor and saved probability identity | Not supported; same LogisticRegression pipeline/C grid and exact same predictions |
| The selected predictors are available before angiography | No row-level timestamps; symptoms/ECG/labs unresolved | Unverified; do not claim prospective or pre-test availability |
| V3 is a validated clinical prediction model | Frozen config and internal nested predictions only | Not established; call internal reference/baseline |
| UCI results externally validate ZAS | Different cohort variables/task/endpoint | Not supported |
| UCI staged performance improves across stages | Phase 4.3 saved IECV predictions; paired stage audit | Stage 4 is descriptively better than Stage 3 on paired records; not all stages improve monotonically; centre and cohort limitations apply |
| Duplicate exclusion sensitivity confirms robustness | Historical scenario used row-only keys and misreported exclusions | Historical claim invalid; corrected sensitivity available separately at n=918 |
| Model is clinically useful or ready for deployment | No prospective utility study, timing verification, calibration-in-practice or impact analysis | Not established |
| Independent external validation is complete | Dataset registry and compatibility audit | No compatible, verified independent cohort established |
