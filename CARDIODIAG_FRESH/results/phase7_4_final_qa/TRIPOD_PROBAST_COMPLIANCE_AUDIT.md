# Phase 7.4 — TRIPOD+AI / PROBAST+AI audit

This is a reporting review, not a compliance certificate or low-risk determination. TRIPOD+AI has 27 reporting items; PROBAST+AI separately assesses development/evaluation across participants/data, predictors, outcome and analysis. See the [official TRIPOD+AI statement](https://www.bmj.com/content/385/bmj-2023-078378) and [PROBAST+AI statement](https://www.bmj.com/content/388/bmj-2024-082505).

## TRIPOD+AI item map

| Items | Topic | Status | Audit finding |
|---|---|---|---|
| 1 | Title | Addressed | Retrospective ZAS analysis and separate UCI analyses identified. |
| 2 | Abstract | Partially addressed | Structured and numerically reconciled; final journal-specific length/format unknown. |
| 3–4 | Background/objectives | Addressed | Motivation, narrow contributions and explicit question present. |
| 5–7 | Data source, setting, participants | Partially addressed | Datasets, centres and source-row counts given; exact eligibility/study dates and patient identities unavailable/incomplete. |
| 8–9 | Outcome/predictors | Partially addressed | Coding, groups, exclusions and timing limits stated; local schema difference and source timing unresolved. |
| 10–13 | Sample size, missingness, development | Partially addressed | Candidate pool, nested selection and preprocessing described; small sample rationale, detailed missingness and executable specification merit author/source review. |
| 14–17 | Evaluation and analysis | Partially addressed | Metrics, thresholds, calibration caveats and split uncertainty reported; patient/centre uncertainty not available; calibration limited. |
| 18 | Open science | Partially addressed | Commands and local code described; no persistent archive/version, registration status or clean install verification. |
| 19 | Patient/public involvement | Requires author confirmation | No statement can be inferred. |
| 20–24 | Results | Partially addressed | Cohorts, metrics, selection and separate centre results provided; clarify no fixed deployable model is proposed. |
| 25–27 | Discussion | Addressed | Limitations, interpretation and scope restrained; no utility or prospective claims. |

## PROBAST+AI domain flags

No formal signalling-question scores were assigned. Qualified authors/reviewers should complete the official instrument.

| Domain | Evidence / concern | Status |
|---|---|---|
| Participants and data sources | Public historical cohorts; ZAS patient independence unverified; provenance/schema discrepancy; four UCI sources reported. | Partially addressed; representativeness concern remains. |
| Predictors | Groups and leakage exclusions reported; acquisition timestamps absent and some variables could follow work-up. | Partially addressed; prediagnostic applicability unestablished. |
| Outcome | ZAS Cath and UCI `num` coding specified; local ZAS schema and outcome provenance require confirmation. | Partially addressed. |
| Analysis | Nested source-row evaluation and preprocessing independently reconciled; small n, three repeats, selection-dependent calibration and limited uncertainty remain. | Partially addressed; calibration/patient inference limited. |

The manuscript is transparent about important limitations. Checklist completion cannot remedy missing identities/timestamps or establish transportability. Complete the official TRIPOD+AI checklist and formal PROBAST+AI appraisal before submission; do not call this audit a low-risk assessment.
