# PROBAST+AI appraisal and applicability assessment

## Scope and interpretation

PROBAST+AI separately appraises model development (quality) and model evaluation (risk of bias), then concerns about applicability. Its four domains are participants/data, predictors, outcome and analysis; it contains 16 development and 18 evaluation signalling questions and six applicability items. This is a structured evidence review, not a substitute for independent dual-reviewer scoring or a claim of “low risk.” Question wording is paraphrased from the [official PROBAST+AI article](https://www.bmj.com/content/388/bmj-2024-082505) and the [official supplementary signaling-question table](https://bmjdigitalhealth.bmj.com/content/bmjdhai/1/1/e000154/DC1/embed/inline-supplementary-material-1.pdf).

Status labels below describe whether evidence is available in this manuscript/repository: **Addressed**, **Partially addressed**, **Missing**, **Not applicable**, or **Requires author confirmation**. They are not PROBAST yes/no votes. “Concern” is explicitly separated from status.

## Model-development quality evidence

| Domain/item | Question focus | Status | Evidence and concern |
|---|---|---|---|
| 1.1 | Appropriate data sources? | Partially addressed | Public historical ZAS source and UCI sources identified; local ZAS 56-column vs catalog 54-feature discrepancy unresolved. |
| 1.2 | Appropriate design? | Partially addressed | Retrospective nested source-row procedure appropriate to the stated task; patient-level independence and cohort selection mechanism unverified. |
| 1.3 | Inclusion/exclusion representative? | Partially addressed | Source sample and counts disclosed, but angiography-referred source population is not demonstrated representative of a prediagnostic target. |
| 2.1 | Predictors consistently defined/assessed? | Partially addressed | Fixed columns/groups; exact units/collection methods and timing are incomplete. |
| 2.2 | Preprocessing similar across participants? | Addressed | Training-partition pipelines and consistent preprocessing described and code inspected. |
| 2.3 | Predictor assessments independent of outcome knowledge? | Requires author confirmation | No source record establishes assessor blinding or whether recording was influenced by known diagnostic results. |
| 2.4 | Predictors available at intended use time? | Requires author confirmation | No predictor time is source-confirmed before the diagnostic decision; prediagnostic applicability is therefore not supported. |
| 3.1 | Outcome defined/assessed appropriately? | Partially addressed | Cath Cad/Normal and official ≥50% context described; local schema discrepancy and assessor/source mapping need confirmation. |
| 3.2 | Outcome assessed similarly for all? | Partially addressed | One dataset target field used; no source-level adjudication consistency data reviewed. |
| 3.3 | Outcome assessment independent of predictor data? | Requires author confirmation | Historical source does not establish outcome-assessor blinding. Direct target fields were excluded as predictors. |
| 3.4 | Predictor-to-outcome interval appropriate? | Requires author confirmation | No acquisition timestamps; interval cannot be established. |
| 4.1 | Sample size reasonable? | Partially addressed | 303 rows, 216 positive labels, 30 candidates and 15 outer fits; paper acknowledges limited size/repeats and no patient-level precision. Material concern. |
| 4.2 | Continuous/categorical predictors handled appropriately? | Partially addressed | Imputation/encoding/scaling/splines are pipeline-contained; field-level definitions, units and category details are incomplete. |
| 4.3 | Missing/censored records handled appropriately? | Partially addressed | Training-only imputation used; feature-level missingness summaries/reasons are incomplete. No time-to-event censoring task. |
| 4.4 | Imbalance treatment followed by appropriate recalibration? | Partially addressed | Class weighting/candidate handling and sigmoid calibration are described; calibration scores are selection-dependent and fit-size mismatched. |
| 4.5 | Overfitting addressed? | Addressed | Nested feature-group/model selection with outer evaluation directly addresses selection optimism; three repeats and small sample leave residual uncertainty. |

## Model-evaluation risk-of-bias evidence

| Domain/item | Question focus | Status | Evidence and concern |
|---|---|---|---|
| 1.1 | Appropriate evaluation data sources? | Partially addressed | Evaluation uses frozen source dataset; no independent ZAS cohort. |
| 1.2 | Appropriate evaluation design? | Addressed | Paired frozen outer partitions and inner selection; outer-test labels are untouched until evaluation. |
| 1.3 | Evaluation sample representative? | Partially addressed | Source rows represent the assembled ZAS file, not a verified independent patient cohort or broad clinical spectrum. |
| 2.1 | Predictors defined/assessed consistently? | Partially addressed | Same fixed data columns in folds; underlying timing/measurement provenance remains uncertain. |
| 2.2 | Preprocessing consistent? | Addressed | Pipeline steps fit within training partitions; independent Phase 6.1 review verified saved partitions and predictions. |
| 2.3 | Predictor assessment blind to outcomes? | Requires author confirmation | Not established by available source documentation; this is distinct from model-training label leakage. |
| 2.4 | Predictors available at intended use? | Requires author confirmation | Timing remains unresolved; prediagnostic application would have high applicability concern. |
| 3.1 | Outcome defined/assessed appropriately? | Partially addressed | Target labels are independently matched to source rows; local source schema/derivation discrepancy remains. |
| 3.2 | Outcome consistent across rows? | Partially addressed | Label reconciliation passes; source-level outcome ascertainment consistency was not independently documented. |
| 3.3 | Outcome assessment independent of predictors? | Requires author confirmation | Assessor blinding/source process not documented; Cath/angiographic direct fields were excluded. |
| 3.4 | Predictor/outcome time interval appropriate? | Requires author confirmation | Not knowable from available timestamps. |
| 4.1 | Avoid apparent performance only? | Addressed | Outer-test predictions estimate the complete inner-selected procedure; Phase 6.1 reconstructed results. |
| 4.2 | Evaluation sample size reasonable? | Partially addressed | 303 rows/87 negative rows per repeat; folds/repeats are overlapping source rows, not independent samples. |
| 4.3 | Missing/censored data handled appropriately? | Partially addressed | Preprocessing is training-only; detailed missingness evidence is limited. |
| 4.4 | Evaluate on data without imbalance correction? | Addressed | Outer test-fold prevalence is not resampled; model-side class weighting does not alter test labels. |
| 4.5 | Avoid data leakage in splits? | Partially addressed | Row-level preprocessing/selection leakage is controlled and independently audited. Patient duplicates cannot be excluded because unique IDs are unavailable. |
| 4.6 | Replicate all development steps in resampling? | Partially addressed | Candidate selection/hyperparameter tuning nested in outer training. Calibration cross-fit reuses labels for selection and is selection-dependent; this limits calibration/probability interpretation, not the verified outer AUC. |
| 4.7 | Evaluate discrimination, calibration and net benefit appropriately? | Partially addressed | AUC/AP/Brier/log loss and calibration intercept/slope reported; calibration has construction limits and no decision-curve/net-benefit analysis. No clinical utility claim. |

## Applicability items

The tool considers participants/data, predictors and outcome for both development and evaluation.

| Assessment | Development | Evaluation | Rationale |
|---|---|---|---|
| Participants/data source match intended population | Partially addressed | Partially addressed | Valid only for a narrow retrospective source-row description; referral/selection and patient identity are unresolved. For prediagnostic clinical patients, concern is high. |
| Predictors match intended use and measurement conditions | Requires author confirmation | Requires author confirmation | No acquisition timestamps confirm availability before diagnostic decision. This is a major applicability barrier to a prediagnostic claim. |
| Outcome matches intended condition/use | Partially addressed | Partially addressed | Appropriate to the stated angiography-defined CAD source outcome if mapping is confirmed; local schema/label provenance remains unresolved. |

## Overall appraisal

- **Retrospective source-row discrimination:** evaluation design/outer discrimination is strongly supported by the independent Phase 6.1 audit. Main concerns remain small sample, unverified patient independence, source population selection, local schema discrepancy, assessor processes and limited calibration inference.
- **Prediagnostic clinical application:** applicability concern is **high/unresolved** because no predictor is confirmed available before the diagnostic decision. Do not present the manuscript as prediagnostic prediction.
- **Calibration and clinical utility:** calibration evidence is limited; decision utility was not studied. No deployment claim is supported.

This report does not assign a single “low/high” overall PROBAST+AI score because the target-use population and source-level predictor/outcome timing are not fully established and a formal independent appraisal is outside this audit. Authors/reviewers should complete the official form independently before journal submission.
