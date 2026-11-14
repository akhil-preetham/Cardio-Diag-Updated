# CARDIODIAG project completion report — initial state and audit entry

Date: 2026-11-13 (Asia/Calcutta)
Repository path: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH`

## Repository and preservation state

The project is nested under the user-supplied parent path. Neither the project folder nor its parent contains Git metadata. A Git working-tree status therefore cannot be established; no clean-tree claim is made. Existing historical result namespaces are retained. Phase 5's manifest lists 40 output paths; all listed files were found and their SHA-256 hashes matched the manifest during the initial read-only reconciliation.

The project includes prior reports and results for research protocol versions 1–4, Phases 2–5, independent audits, public-dataset acquisition, reliability, explanation, and manuscript development. These are historical evidence; their presence does not imply that every claim has been independently reverified in this completion pass.

## Current authority by track

### Z-Alizadeh Sani

- Frozen reference: `results/primary_research_v3/`, configured for Age, Sex, Current Smoker, EX-Smoker, and FH; 5 outer folds and 4 inner folds with fold-specific nested sigmoid calibration. The recorded highest descriptive calibrated ROC-AUC is 0.735 ± 0.060. The design is internal validation and lacks a patient identifier.
- Phase 5 candidate study: `results/phase5/zas_track/` with 6 feature groups and 6 model labels, 288 outer selection records, and 21,816 fold-level prediction rows.
- Phase 5.1 audit: `runner/phase5_1_audit.py` and `results/phase5_1_audit/`. No separately named Phase 5.1.1 implementation or output namespace was found. The Phase 5.1 runner writes to fixed output paths; rerunning it can overwrite the existing audit artifacts.

### UCI four-centre CAD

- Phase 4 and Phase 4.1–4.3 contain centre-aware protocols, saved predictions, audits and decisions.
- Phase 5 separates stage/centre results and retains Stage 2 early-presentation and Stage 4 exercise-test tasks. Existing decisions say Stage 2 is the more defensible expanded early-presentation task and Stage 4 is a later, separate task. Centre heterogeneity and small/unstable strata remain limitations.
- The repository states that the UCI endpoint is related to, but not equivalent to, the ZAS angiographic endpoint. It is not external validation of the ZAS model.

### External datasets

The registry and acquisition reports cover multiple candidates. Existing reports identify compatibility, licensing, overlap, access and predictor-timing constraints. No compatible independent external validation cohort is established in the current claims. This completion pass will retain that status unless a dataset's provenance and endpoint compatibility can be demonstrated from local evidence.

## Phase 5.1.1 questions checked from existing artifacts

Saved eligible-combined predictions were independently aggregated by source row and averaged over the two repeats. For both `logistic` and `ridge`, the audit obtained 606 fold predictions, 303 unique source rows, 216 positive outcomes, two predictions per row, and no inconsistent labels. Recomputed ROC-AUC was 0.928214, PR-AUC 0.966830, Brier score 0.096424, and log loss 0.314108, matching the saved benchmark. The logistic and ridge prediction vectors were exactly identical (maximum absolute difference 0).

The model constructor maps both labels to the same L2 `LogisticRegression` pipeline and same C grid. Thus the 36 benchmark rows are not 36 distinct estimator implementations; at most five distinct estimator families are represented by the labels, before accounting for overlap among other configurations. The 0.928 candidate is a result row in a table; no explicit winning-row selection implementation was found. The row was identified after examining 36 benchmark candidates, and feature-group/model choice was not nested. It remains exploratory. Timing evidence for most included features is unresolved; obvious echo and angiographic fields are excluded.

## Known state inconsistencies and risks

- Phase 5.1 report text describes “source patients” although the dataset has no verified patient identifier; completion documents will use “source rows.”
- The Phase 5.1 feature table labels Age and Sex as confirmed pre-outcome, but local source documentation does not prove acquisition timestamps. The final eligibility matrix will classify timing only to the level supported by available evidence.
- The Phase 5.1 runner is described as read-only while writing/overwriting outputs.
- An earlier comprehensive audit notes a contradiction between `docs/EXPERIMENT_RESULTS.md` and the final verification/statistical artifacts regarding whether paired tests were completed. The authoritative evidence will be resolved from files before manuscript claims are updated.
- Existing reports and manifests disagree across phases in scope and evidentiary status in places; historical artifacts will not be edited to force consistency.
- The project's checked-in virtual environments refer to an unavailable or permission-blocked interpreter in this sandbox. A separate local Python environment exists outside the project, but initial execution was denied by the sandbox. Lightweight saved-prediction and hash checks have been performed without model fitting; Python QA will be attempted only through the normal approval path if needed.

## Initial completion assessment

The repository already contains substantial implementation and historical evidence. The work is not yet established as submission-ready: the main remaining scientific checks are phase-to-phase evidence reconciliation, definitive task-specific feature eligibility, selection-aware interpretation of ZAS, centre-aware reconciliation of UCI staged results, and careful manuscript/reference validation. No model has been retrained during this initial inspection.

This report is the baseline record for the completion pass. Follow-on audit deliverables will be added under `results/final_project/` and manuscript artifacts under `paper/` without replacing historical phase outputs.

## Completion-pass update (2026-11-13)

The independent saved-artifact reconciliation is complete. No expensive experiment was rerun and no historical result path was overwritten. Added a corrected UCI duplicate-sensitivity reconstruction keyed by centre and source row, plus final decision, results, claims/evidence, QA, and human-action handover files under `results/final_project/`. The focused manuscript, supplementary material, and cover-letter draft are under `paper/`; the prior manuscript was retained under `paper/legacy_cardiodiag_reproduction/`.

Automated verification completed: 81 tests passed and one optional-dependency test skipped; 115 Python sources parsed; new audit runner passes Ruff; repository-wide Ruff still has 15 pre-existing findings. Phase 5, Phase 4 multicentre, and Phase 4.3 manifests all matched (40/40, 34/34, 43/43). The LaTeX source citation keys match the bibliography, but compilation could not be performed because `pdflatex`, `latexmk`, and `bibtex` are unavailable. No model was refit.

The final research decision is **further experimental and reproducibility work required before submission**. Remaining scientific and author-dependent actions are enumerated in `results/final_project/REMAINING_HUMAN_ACTIONS.md` and `paper/publication_readiness_report.md`. Git status remains unavailable because the parent and project contain no Git metadata.
