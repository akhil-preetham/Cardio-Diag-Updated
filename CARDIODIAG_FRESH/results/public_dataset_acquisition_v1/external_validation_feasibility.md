# External-validation feasibility

## Directly suitable

None currently. No downloaded public cohort simultaneously verifies independence from Z-Alizadeh, the same angiographic >=50% endpoint, all five frozen predictors, predictor timing and adequate binary outcome support.

## Conditionally suitable

- **UCI four-centre:** suitable for a separate four-centre site-aware experiment and common-subset models. It is not a clean external validation of frozen V3 because former smoking and family history are absent from the processed files, and Cleveland is part of the development lineage.
- **AngioCAD:** suitable for a separate patient-level clinical or imaging study after the clinical dictionary and outcome mapping are verified. It is not currently a frozen V3 validation cohort.

## Related-task research datasets

Mendeley stenosis detection and CADICA support image/lesion research. AngioCAD supports image/video and artery-level stenosis research. MultiD4CAD supports restricted multimodal CCTA research.

## Unsuitable for the requested claim

Statlog is non-independent from Cleveland; Mendeley cardiovascular, Kaggle CVD 70K, MIMIC-IV and eICU do not establish the required angiographic endpoint; the 100-patient Mendeley stenosis cohort is all-positive.

## Recommended next experiment

First run a frozen-schema, site-aware UCI four-centre audit using only predictors actually present and timed before the target, with complete-case and training-fold-only imputation. In parallel, prepare an AngioCAD patient-level imaging/clinical audit, keeping all frames from a patient in one partition. Do not label either result external validation of frozen V3 until the predictor and endpoint gaps are resolved.
