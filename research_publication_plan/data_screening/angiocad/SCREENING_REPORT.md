# AngioCAD tabular screening

Official source: https://zenodo.org/records/15826856 (DOI 10.5281/zenodo.15826856).
Provider metadata reports open access under CC BY 4.0. Downloaded only the two small
clinical/label workbooks; all image archives were excluded. File sizes, provider MD5
and local SHA-256 are recorded in PROVENANCE_MANIFEST.json. Source data are preserved
unchanged; all reports are aggregate. No model was fitted.

| Screening quantity | Count |
|---|---:|
| Nonblank clinical-feature IDs | 377 |
| Nonblank label IDs | 413 |
| Matched unique IDs | 377 |
| Matched provisional >50% cases | 298 |
| Matched no-segment->50% cases | 79 |
| Matched nonpositive cases containing a 26-50 segment | 10 |

## Endpoint sensitivity

The provisional positive rule is any segment in 51-75, 76-90, 91-99 or 100.
It operationalizes **>50%** recorded segment stenosis, not ZAS's **>=50%** outcome.
The 26-50 bucket combines exact-50 with lower stenosis and cannot distinguish them.
Cases without a >50 segment but with a 26-50 segment are boundary-ambiguous for >=50.
Do not label all 26-50 cases positive as if this represented >=50; that is an upper-bound
sensitivity assignment, not the true endpoint. Present lower/upper assignments only
as sensitivity analyses with their changed definitions. No-segment->50 controls are
non-obstructive under this rule and are not necessarily normal/healthy.

## Clinical and feature compatibility

The clinical table's Diagnosis counts (aggregate_screening.json) include STEMI,
NSTEMI, unstable angina and stable angina. This coronary referral spectrum differs
from population screening and may differ substantially from ZAS. FH, EX-Smoker and
the ZAS symptom/ECG groups are absent from the header. An unchanged five-feature V3
or 38-field model therefore cannot be externally evaluated with these fields.

A new shared-feature development procedure could be planned using Age, Sex, DM,
HTN, Current Smoker and verified common laboratory variables. Check unit/encoding,
measurement timing, missingness, eligibility and outcome mapping before fitting.
Exclude outcome-defining artery labels, Gensini Score, Diagnosis and diagnostic
series fields. LVEF/echo and treatment fields require explicit timing restrictions.
Do not select mappings, predictors or model settings by external-cohort performance.

The 413 label cases do not imply 413 usable tabular cases: only 377 have feature IDs.
Reasons for the 36 missing clinical cases remain unresolved and may introduce selection.
Unique IDs are case keys in this release; this alone does not establish every case
represents an independent patient with no repeated visits.

## Reproduce

Run `python screen_angiocad.py` in this directory. Existing downloads are verified
and reused; the script prints only aggregate counts. It writes aggregate JSON,
this report and provenance metadata. It does not train or generate a paper result.
