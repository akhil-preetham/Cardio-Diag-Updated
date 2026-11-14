# External Validation Action Plan

1. Obtain a cohort with the same five predictors and a compatible angiographic CAD endpoint.
2. Obtain patient-level independence evidence and acquisition-time documentation.
3. Harmonize coding and units before opening the external test labels.
4. Freeze the V3 model and calibration parameters before evaluation.
5. Report discrimination, calibration, missingness and domain shift without refitting on the external cohort.

MultiD4CAD is a restricted-access 118-sample multimodal CCTA/clinical dataset (Zenodo record 15148653; 78 CAD and 40 without CAD) and may be useful for a separate multimodal CAD track, but strict predictor equivalence and access are unresolved. CADICA is a 42-patient invasive coronary angiography image/lesion dataset (Mendeley DOI 10.17632/p9bpx9ctcv.4; CC BY 4.0), but it is primarily an image/lesion task and does not provide verified equivalence to the five-feature tabular model. Neither supports strict external validation in this phase.
