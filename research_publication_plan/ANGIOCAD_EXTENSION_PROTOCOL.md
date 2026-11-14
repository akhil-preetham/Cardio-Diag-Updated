# Optional extension: a frozen shared-feature cross-cohort CAD study

Status: design proposal following aggregate source screening, 14 November 2026. No model has been trained on AngioCAD and no transport score exists. This extends, rather than replaces, the existing ZAS research question. The urgent evidence-only manuscript does not depend on completion of this extension.

## Why this is worth adding

An independent clinical source addresses a more meaningful gap than increasing the number of classifier trials on 303 previously analyzed rows. AngioCAD provides a real, openly licensed opportunity, but with missing source fields and different outcome/referral definitions. An honest assessment of whether a compact ZAS-developed model transports, including failure, could strengthen the empirical contribution. It does not make nested validation or logistic regression a new algorithm.

## Data already acquired

The unchanged feature and label workbooks, official Zenodo metadata and provider/local hashes are in data_screening/angiocad/. The record has 413 label IDs, 377 feature IDs and 377 matched cases. The 36 label cases without clinical rows require an exclusion table; their absence may be selective. Case identifiers permit joins but do not independently verify that no patient has repeated visits.

AngioCAD has 298 matched cases with a segment above 50% and 79 without such a segment. Ten of those 79 include a 26-50 segment, so their exact >=50 status is unresolved. STEMI/NSTEMI/unstable-angina/stable-angina case composition is not equivalent to healthy versus diseased population sampling. These aggregates are feasibility findings, not classifier performance.

## Freeze before fitting or examining predictor-outcome associations

1. **Primary task:** retrospective discrimination of recorded obstructive coronary status. Preserve source-specific label definitions. ZAS Cath >=50 versus AngioCAD >50 is a documented transport/applicability discrepancy; do not call endpoints identical. Retain the 10 boundary-ambiguous cases as a prespecified sensitivity set. Possible sensitivity assignments include excluding them, classifying them all negative, and an upper-bound all-positive assignment; none reconstructs their true exact threshold status. Report each changed estimand.
2. **Fields:** propose Age, Sex, DM, HTN and Current Smoker as a compact shared core, subject to exact source definitions/codes. Laboratory extension fields are permitted only after units, assay definitions, missingness and acquisition context are verified. Do not substitute total cholesterol for LDL, absent former-smoker for no former smoking, or case-ID for patient identity. A failed mapping removes a field with an explanation before any score comparison.
3. **Excluded predictors:** Cath and all stenosis/outcome labels, Gensini Score, clinical Diagnosis, patient/case ID, procedure series, and diagnostic/downstream or treatment variables lacking an appropriate timing justification. No clinical early-prediction claim is permitted for the retained fields without source evidence of timing.
4. **Model:** fix a simple regularized logistic procedure first. Choose C and preprocessing only within ZAS development validation. If a nonlinear comparator is included, freeze its search budget and report both. Do not select models by AngioCAD results. A shared-feature procedure is a new model, not unchanged validation of the historical V3 or 38-field model.
5. **Source evidence:** evaluate the shared-feature procedure with training-local preprocessing and nested source validation. Historical exposure to ZAS makes this a retrospective development process. Use no AngioCAD rows to fit imputation, scaling, feature selection or source calibration in the primary locked transport analysis.
6. **Transport evaluation:** fit the prespecified procedure on ZAS development data and apply to the matched AngioCAD cohort once mapping is frozen. Report discrimination, AP with prevalence, Brier/log loss and descriptive calibration. Show source/target case mix and missingness before interpreting any difference. A drop in calibration is plausible under acute/stable referral shift; report it rather than tuning it away.
7. **Uncertainty:** provide source-record-based uncertainty only where patient clustering cannot be established, and state that limitation. Bootstrap evaluation of a fixed transported model conditions on its training fit. Do not treat development/target cohort differences as randomized causal comparisons. Include negative-class precision concerns with only 79 cases under the provisional non-obstructive definition.
8. **Adaptation:** omit target-label recalibration from the first urgent analysis. If later studied, allocate separate adaptation and evaluation records, fix the split and method in advance, and report untouched transport and adapted results separately. Do not recalibrate and evaluate on the same target labels.
9. **Stopping:** pilot runtime/RAM, not predictive success. Complete the frozen analysis even if results are worse or inconclusive. No model family or feature mapping may be added after examining target performance without labeling a new exploratory analysis.

## Required outputs before an external result enters the paper

- Source/target dictionary and units/codes/timing matrix, plus unresolved facts.
- Patient/case independence and cross-source overlap assessment; documented inclusion/exclusion flow.
- Dated model/endpoint/protocol freeze with hashes, including boundary sensitivities.
- Full source and target predictions, task-specific metrics, class counts and limitations, published only within applicable terms.
- New manifests and checks for training/evaluation disjointness and no target-outcome-driven selection.
- A revised manuscript that calls this a new shared-feature cross-cohort evaluation with outcome/spectrum limitations; no claim of validation of exact historical ZAS weights or universal clinical reliability.

If fields or outcomes fail compatibility, report the feasibility finding and retain the existing bounded paper. An online dataset's clinical origin alone is not sufficient to justify a transportability claim.
