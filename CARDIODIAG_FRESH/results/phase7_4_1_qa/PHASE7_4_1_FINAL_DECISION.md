# Phase 7.4.1 final decision

## Decision: READY WITH SPECIFIED REMAINING ISSUES

The requested technical corrections and review rendering are complete. A clean Windows/Python 3.11.16 environment installed the frozen 126-package snapshot with no dependency conflicts; focused smoke tests passed. Figure 2 was corrected to show the full 0–1 ROC-AUC range, both PNG and PDF were regenerated from the same saved metrics, and the original Phase 7.3 assets remain intact. The full corrected manuscript renders as a 12-page PDF; all pages were rasterized and visually reviewed, with no detected missing content, glyph substitution, clipping or overflow.

This decision permits journal selection after considering the limits below. It is not a submission-readiness, checklist-compliance, clinical-utility, prospective-prediction, or external-validation claim.

## Remaining issues before submission

1. **Author declarations and source facts:** ethics approval/waiver, funding, conflicts, registration, patient/public involvement, authorship and some data-collection, eligibility, treatment, outcome-assessment and assessor-blinding details require author confirmation. The checklist does not infer these facts.
2. **Feature timing and target use:** symptoms, ECG and laboratory acquisition times remain unresolved. No prediagnostic availability is confirmed. The defensible framing remains retrospective discrimination of angiography-defined CAD using recorded clinical features.
3. **Applicability and data limitations:** source-row evaluation does not establish patient-level independence; local ZAS schema provenance remains unresolved; sample size and three repeats limit precision. PROBAST+AI applicability for prediagnostic use is high concern/unresolved. No fairness analysis or clinical decision-utility analysis is available.
4. **Rendering scope:** the PDF is a visually reviewed Markdown-to-PDF review artifact made with a local ReportLab runtime. No LaTeX compile or journal-specific template/layout validation was possible. The local rendering dependencies are not part of the project’s pinned 126-package snapshot.
5. **Repository provenance:** no Git metadata was present, so a working-tree diff could not be established. The existing Phase 7.3 manifest’s 46 input and 28 output hashes were verified unchanged.

The itemized TRIPOD+AI checklist and PROBAST+AI assessment distinguish addressed reporting from methodological quality and author-dependent information. Populating these checklists is not evidence of compliance or low risk of bias.

## Scope confirmation

No Phase 1–7.3 historical output, Phase 6 experiment, UCI track, endpoint or saved metric was changed. No model fitting, benchmark, new candidate search, journal selection or submission occurred. The Phase 7.3 source manuscript remains unchanged; the corrected copy is under `paper/phase7_4_1_final_manuscript/`.
