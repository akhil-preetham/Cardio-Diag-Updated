# Publication release checks

The scientific experiment and eight-page report were completed before repository packaging. This release adds relocation-safe verification, pinned runtime setup and a default primary test profile without rewriting historical experiment bytes or scores.

## Verified before publication

- Source/credential-pattern scan found no candidate credentials in the prepared source tree.
- Environments, caches, nested Git metadata, compiler binaries and multi-gigabyte data archives are excluded.
- Notebook output cells and execution state are cleared in the publication copy. `SOURCE_PUBLICATION_MAP.csv` records that transformation.
- Both primary experiment manifests and the frozen transport input protocol verify in a relocated tree: 163 supported hash entries, no missing allowances.
- Metrics reconstruct from 7,272 source prediction rows, 24 source repeat/procedure groups, 377 target cases and six target endpoint/procedure groups.
- The portable tests move real artifacts into a new temporary checkout and detect corruption there even while the original workstation files still exist.
- Saved model loading fails before unpickling if core runtime versions or the trusted local model bytes differ.
- `.gitattributes` preserves exact frozen file bytes across operating systems.

The original workstation suite result is **116 passed, one optional historical LoRAS test skipped**. The default publication profile intentionally targets the included primary datasets/evidence. A separately created Python 3.11 environment installed the release requirements successfully, passed dependency consistency, verified all **163 hash entries** without missing allowances, and passed **39 tests with one optional historical LoRAS test skipped**. All **153 published Python modules** passed syntax parsing. The exact transitive package versions are retained in `requirements-release.lock.txt`. Full legacy tracks remain optional because their data/environments are not all distributed in this repository.

## Scientific boundaries

Finalization adds 57 verified complete-repeat groups and 627 reconstructed decision-curve points, training-only threshold selection, source/target learning curves, coefficient stability and perturbation scenarios. The updated suite passed **48 tests, one optional historical LoRAS test skipped**. The final eight-page paper uses embedded Times New Roman at 10-point text size; its font, numeric and rendering checks are recorded in `CARDIODIAG_FRESH/paper/final/PAPER_QA.json`. Prior papers and original experimental bytes remain unchanged. See [FINALIZATION.md](FINALIZATION.md).

The additive v2 follow-up completed all prespecified analyses with no captured warnings. Independent verification recomputed **84 complete-repeat metric groups**, checked 45-component development/calibration/test separation, target adaptation/evaluation separation and 50 bootstrap out-of-bag memberships. The clean ordinary-core rerun reproduced all 909 established predictions to 1e-12. The updated release suite passed **45 tests with one optional LoRAS test skipped**, and dependency consistency passed. The original 163 primary hash entries remain valid. Follow-up results and rejected variants are documented in [RELIABILITY_V2.md](RELIABILITY_V2.md).

The results do not establish a new universally superior algorithm. Equal-feature logistic outperformed historical selection; calibration was mixed across repeats; cohort transport and input loss degraded performance. Predictor timing, unique patients, target binary ascertainment and endpoint equivalence remain unresolved. Ethics/funding/author declarations and journal submission requirements remain human responsibilities. This is a tested research release, not a claim that no future improvement is possible.
