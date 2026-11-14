# TRIPOD+AI itemized reporting checklist

This checklist maps the Phase 7.4.1 manuscript to the official 27-item TRIPOD+AI statement and its lettered subitems. “Addressed” means the draft reports the item; it does not certify methodological validity. Where details need confirmation, the draft must retain an explicit limitation until authors provide evidence. Source: [official TRIPOD+AI statement and expanded checklist](https://www.bmj.com/content/385/bmj-2023-078378), [expanded checklist PDF](https://www.bmj.com/content/suppl/2024/04/25/bmj-2023-078378.DC8/colg078378.wt1.pdf).

| Item | Reporting element | Status | Manuscript location / action |
|---|---|---|---|
| 1 | Identify study as prediction-model development/evaluation, target population and outcome | Partially addressed | Title states retrospective ZAS discrimination/angiography-defined CAD; model’s target population is source dataset rows, not a defined clinical-use population. |
| 2 | Abstract addressing TRIPOD+AI Abstracts checklist | Partially addressed | Structured abstract reports objective, data, methods, results and conclusion; journal-specific 13-item abstract mapping and exact abstract checklist not completed. |
| 3a | Healthcare context/rationale and existing models | Addressed | Introduction frames retrospective CAD discrimination and selection optimism; cites original ZAS/V3 context. |
| 3b | Target population, intended purpose, care-pathway point and users | Partially addressed | Retrospective dataset population and question defined; no clinical use point/user is claimed, appropriately. |
| 3c | Known health inequalities in target population | Missing | No health-inequality review or subgroup representation analysis. Avoid claiming absence of inequality. |
| 4 | Explicit study objectives and development/evaluation type | Addressed | Objective and central question identify reanalysis/evaluation of selection procedure, not new classifier development. |
| 5a | Data source, rationale and representativeness for development/evaluation | Partially addressed | Public historical datasets and separate UCI track described; intended-population representativeness and some provenance remain limited. |
| 5b | Data collection/accrual dates | Requires author confirmation | Dataset collection date range not established in local evidence. Confirm from source documentation or state unavailable. |
| 6a | Setting, centre count and location | Partially addressed | Tehran Rajaei and UCI centres named; confirm exact country/location/context/source data setting. |
| 6b | Eligibility criteria | Requires author confirmation | Local manuscript does not establish original cohort inclusion/exclusion criteria beyond available dataset rows. Seek source documentation. |
| 6c | Treatments received and handling, if relevant | Requires author confirmation | No treatment variables/handling are described; authors should confirm whether treatment before recorded features/outcome is relevant. |
| 7 | Preprocessing and data-quality checking | Partially addressed | Pipeline-based imputation/encoding/scaling and leakage exclusions described; full feature-level missingness/data-quality summaries and source harmonization details are incomplete. |
| 8a | Outcome definition/time horizon, assessment, rationale, consistency | Partially addressed | Cath Cad/Normal and UCI `num` outcome defined; angiography-based endpoint described. Local ZAS schema discrepancy and assessment details remain unresolved; prospective horizon is not applicable to the retrospective framing. |
| 8b | Outcome assessor qualifications, if subjective | Requires author confirmation | Assessors/diagnostic adjudication metadata are not available in reviewed sources. |
| 8c | Blinding of outcome assessment to predictors | Requires author confirmation | No source evidence on outcome-assessor blinding. |
| 9a | Initial predictor choice and prespecification/preselection | Addressed | Six frozen groups, five unique families, excluded leakage fields and Phase 5 separation are described. |
| 9b | Predictor definitions, units, timing and blinding | Partially addressed | Fields/groups and timing uncertainty are listed; exact units/measurement definitions and acquisition times are incomplete. Do not infer prediagnostic availability. |
| 9c | Predictor assessor qualifications, if subjective | Requires author confirmation | Subjective symptom/examination/ECG interpretation assessor details are not in the local artifacts. |
| 10 | How sample size was determined and sufficiency | Partially addressed | 303 rows/216 events, 30 candidates, only three completed repeats and runtime reduction disclosed; no formal sample-size calculation or precision-based justification. |
| 11 | Missing-data handling and omissions | Partially addressed | Training-only imputation is described; feature-wise missingness counts/reasons and omitted-row details are not fully provided. |
| 12a | Data use, partitions, sample-size considerations and leakage | Addressed | Frozen nested outer/inner structure, complete-repeat predictions and outer-test isolation described and independently audited; source-row independence limitation explicit. |
| 12b | Predictor functional form, scaling, transformation and categories | Partially addressed | Pipeline transformations and spline/logistic candidates described; complete operational details rely on repository code, not public archive. |
| 12c | Model rationale, all steps/hyperparameters and internal validation | Partially addressed | Candidate pool/inner selection/families and code location described; full public executable code/lockfile availability remains incomplete. |
| 12d | Heterogeneity across clusters/centres | Partially addressed | UCI held-centre results shown; ZAS has no verified patient/centre clustering and source-row evaluation cannot resolve it. |
| 12e | Performance measures/plots and comparison rationale | Partially addressed | AUC/AP/Brier/log loss/sensitivity/specificity/calibration and paired comparisons reported; no decision curve/net benefit and no clinical-utility claim. |
| 12f | Model updating | Not applicable | No post-evaluation model update/recalibration is proposed. |
| 12g | How evaluated predictions were calculated | Partially addressed | Reproduction code/artifacts are referenced locally; a stable public code archive and persistent locator are not yet available. |
| 13 | Class imbalance rationale and recalibration | Partially addressed | Outcome prevalence and balanced V3-style logistic/calibration limitations reported; calibration construction is selection-dependent and remains limited. |
| 14 | Fairness approaches and rationale | Missing | No fairness approach or subgroup performance analysis was performed. |
| 15 | Model output, classifications and thresholds | Partially addressed | Probability scores and training-selected Youden thresholds described; no clinical threshold rationale, fixed deployed model, or complete threshold table is provided. |
| 16 | Development/evaluation differences in setting/eligibility/outcome/predictors | Addressed | ZAS internal procedure and separate UCI task are explicitly distinguished; UCI is not external validation. |
| 17 | Ethics committee approval/waiver | Requires author confirmation | Explicit placeholder; no approval/exemption asserted. Obtain institutional determination. |
| 18a | Funding source and funder role | Requires author confirmation | Placeholder retained. |
| 18b | Conflicts and financial disclosures | Requires author confirmation | Each author must confirm. |
| 18c | Protocol access or no protocol; deviations | Partially addressed | Frozen protocol available in repository; no stable public archive, registration, or finalized deviation statement. |
| 18d | Registration information or not registered | Requires author confirmation | Registration status not known. |
| 18e | Data availability/restrictions | Partially addressed | UCI source locations cited; local derived data, terms, version and persistent access details need author confirmation. |
| 18f | Analytical code availability/version/license | Partially addressed | Local runner/environment files and exact commands exist; persistent public archive/version/license not yet supplied. |
| 19 | Patient/public involvement | Requires author confirmation | No involvement statement inferred; authors must confirm. |
| 20a | Flow and outcome counts | Partially addressed | Cohort counts and class totals provided; detailed eligibility/row-flow and missing-data flow not available. |
| 20b | Dataset characteristics, dates, predictors, missingness, treatments | Partially addressed | Counts/centres/outcomes reported; predictor distributions, missingness proportions, treatments and dates are incomplete. |
| 20c | Compare evaluation data to development data | Partially addressed | The same ZAS source forms nested training/test partitions; predictor distributions are not tabulated by partition. UCI is a different task and is not used as an external comparison. |
| 21 | Participants and outcome events for each analysis | Partially addressed | 303/216 per ZAS repeat and UCI centre/total counts given; detailed inner partition row/event counts not in manuscript. |
| 22 | Full model specification for prediction in new people | Not applicable | Study evaluates a selection procedure, not a fixed clinically deployable final model; no individual-level implementation is claimed. |
| 23a | Performance estimates, uncertainty and key subgroups | Partially addressed | Complete-repeat and centre metrics reported; ZAS interval is descriptive split variation, not a valid patient/population CI; subgroup CIs unavailable. |
| 23b | Heterogeneity across clusters | Partially addressed | UCI centre variation presented; ZAS centre/patient heterogeneity and cluster-level uncertainty cannot be estimated from the available IDs. |
| 24 | Model updating results | Not applicable | No model updating is reported. |
| 25 | Overall interpretation/fairness/context | Addressed | Restrained retrospective conclusion; competing explanations and no-utility limitation discussed. Fairness evidence remains absent (see item 14). |
| 26 | Limitations and implications | Addressed | Sample size, selection, timing, identity, calibration, uncertainty, schema and transport limitations disclosed. |
| 27a | Handling poor/unavailable inputs in implementation | Not applicable | No clinical implementation is proposed; if the work is repositioned for use, this item must be completed. |
| 27b | User interaction/expertise | Not applicable | No intended model user or implementation workflow is claimed. |
| 27c | Next research steps for applicability/generalizability | Addressed | Need for timing evidence and independent compatible evaluation is discussed. |

## Conclusion

This mapping identifies reporting completeness and author actions. It does not establish TRIPOD+AI compliance, methodological validity, clinical utility or low risk of bias. Items marked “requires author confirmation” must remain open until documentary evidence or approved declarations are supplied.
