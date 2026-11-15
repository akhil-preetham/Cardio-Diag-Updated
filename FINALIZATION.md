# Finalized retrospective research release

The reference-style six-page editable Word paper and matching PDF are in `CARDIODIAG_FRESH/paper/reference_format/`. They follow the supplied reference layout with Times New Roman 10-point text, three editable equations and 65 independently verified table values. All six rendered pages were inspected. This changes presentation only; frozen experiments and results remain unchanged. See the folderâ€™s `CONFERENCE_FORMAT_README.md` for layout deviations and rebuilding.

The detailed research report is `CARDIODIAG_FRESH/paper/final/CARDIODIAG_FINAL_PAPER.pdf`: eight pages, two columns, actual embedded Times New Roman at 10 TeX points (9.96264 PDF points). Title, body, tables and references use the requested 10-point text. Mathematical notation retains TeX symbol/superscript fonts. Plot labels use Times New Roman at nominal 10 points. The previous papers and all historical research outputs remain unchanged.

## Completed scope

The reviewed submission-upgrade draft has been selectively integrated into the current paper with its latest evidence retained. New supplementary methods and an author-review audit use readable wrapped paths and the requested typography. Figure values are loaded from frozen CSV evidence; `tools/paper_evidence.py` reconstructs 732 metric values across 180 prediction groups and verifies input/artifact hashes. Its scope is computational/editorial verification, not independent investigator or clinical replication. The unsupported additional bootstrap from the reviewed ZIP is excluded.

- Original information-matched controls, selection-independent calibration and shared-core public-cohort transport.
- Version-two calibration alternatives, explicit feature typing, training-only missingness augmentation, retrospective target adaptation, subgroup diagnostics and paired/refitting uncertainty. Worse candidates remain unpromoted.
- Final sensitivity-focused threshold evaluation, decision curves, source learning curves, target-adaptation learning curves, coefficient stability and numeric input perturbations. Protocol/code/input hashes were frozen before experiments, with all variants retained.
- Portable scoring, pinned runtime, source/model hash checks and Windows/Linux release checks.

The final source-core threshold targets 90% sensitivity using training-only nested predictions. Actual held-out mean sensitivity is **0.888889**, specificity **0.490421**, PPV **0.812574** and NPV **0.640008**. This quantifies the missed-case/false-positive trade-off rather than claiming a guaranteed screening target.

Source-core development learning curves improve mean AUC from **0.769024** with 25% of outer training records to **0.805884** with all training records. Target-offset adaptation yields mean Brier **0.152489** at 10% of the adaptation records and **0.150723** at 100%; one deterministic stratified sample per fraction/fold limits interpretation. Five-percent binary-code flips reduce source AUC to **0.782842**. Fifty bootstrap refits describe coefficient signs, not causal effects. Decision curves retain 627 values and explicit all/none baselines; no clinical threshold is recommended.

## Checks

```sh
python tools/portable_release.py --root . --check-runtime
python tools/verify_reliability_v2.py
python tools/verify_final_analyses.py
python tools/verify_publication.py --root .
python -m pytest -q
```

The final analysis verifier reconstructs 57 metric groups and checks 627 decision-curve points, threshold/adaptation separation, sensitivity diagnostic counts and exact full-size baseline reproduction. The release suite passed **48 tests**, with one optional historical LoRAS test skipped. Final experiment warnings: none.

## Rebuild the paper

Install the pinned research environment, provide a licensed Times New Roman installation, then run:

```sh
python tools/plot_final_analyses.py
python tools/build_final_paper.py
tectonic CARDIODIAG_FRESH/paper/final/CARDIODIAG_FINAL_PAPER.tex --keep-logs
```

On Windows the default font directory is `C:/Windows/Fonts`. Elsewhere set `CARDIODIAG_TIMES_DIR` to a directory containing `times.ttf`, `timesbd.ttf`, `timesi.ttf`, `timesbi.ttf`, and set `CARDIODIAG_TIMES_FONT` to the regular font path for plotting. The builder makes ignored local font copies because the Windows compiler requires relative paths. Font files are never published. The compiled PDF embeds font subsets. With the document runtime (pdfplumber, pypdf, pypdfium2 and Pillow), run `python tools/qa_final_paper.py` and review all rendered pages. The QA file records embedded fonts, sizes, page count, section/numeric checks and PDF hash.

## Evidence that remains unavailable

Completion is bounded by the released public records. No additional compatible independent clinical cohort has been established in this release. Patient identity/grouping, acquisition timing, exact boundary stenosis, authoritative code equivalence, prospective validation and institutional declarations remain unresolved. Existing historical multi-dataset files with different endpoints or feature definitions cannot be merged merely to increase sample size. An independent research team has not reproduced the full study; automated verification and matched baseline refits are the reproducibility evidence provided here.

No journal submission, acceptance, ethics approval, patient-specific reliability or clinical deployment readiness is asserted. Corresponding author, contributions, funding/conflicts and institution-specific ethics determination require author confirmation before submission. Further model searches solely to increase the reported score are outside this finalized retrospective study.
