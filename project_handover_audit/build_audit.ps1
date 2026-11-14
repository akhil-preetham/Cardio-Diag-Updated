$ErrorActionPreference = 'Stop'
$project = Join-Path (Get-Location) 'CARDIODIAG_FRESH'
$audit = $PSScriptRoot
$exclude = '[\\/](\.venv[^\\/]*|\.pytest_cache|\.ruff_cache|__pycache__|\.sf)[\\/]'
function Get-ProjectFiles($directory) {
 foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
  if ($item.PSIsContainer) {
   if ($item.Name -notmatch '^(\.venv.*|\.pytest_cache|\.ruff_cache|__pycache__|\.sf)$') { Get-ProjectFiles $item.FullName }
  } else { $item }
 }
}
$files = @(Get-ProjectFiles $project)
$inventory = @($files | ForEach-Object { [pscustomobject]@{path=[IO.Path]::GetRelativePath($project,$_.FullName);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
$inventory | Export-Csv (Join-Path $audit 'FILE_INVENTORY.csv') -NoTypeInformation
$checks = @()
foreach ($namespace in @('results/phase6_zas_selection/PHASE6_MANIFEST.json','results/phase6_1_audit/PHASE6_1_MANIFEST.json','results/phase7_1_evidence/PHASE7_1_MANIFEST.json','results/phase7_2_methodology/PHASE7_2_MANIFEST.json','paper/phase7_3_manuscript/PHASE7_3_MANIFEST.json','results/phase7_4_1_qa/PHASE7_4_1_MANIFEST.json','paper/final_college_report/FINAL_REPORT_MANIFEST.json')) {
 $manifestPath = Join-Path $project $namespace
 if (!(Test-Path -LiteralPath $manifestPath)) { continue }
 $m = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
 foreach ($section in @('inputs','outputs')) {
  if (!$m.$section) { continue }
  foreach ($entry in $m.$section.PSObject.Properties) {
   if ($entry.Value -isnot [string] -or $entry.Value -notmatch '^[a-fA-F0-9]{64}$') { continue }
   $target = Join-Path $project $entry.Name
   $exists = Test-Path -LiteralPath $target -PathType Leaf
   $actual = if ($exists) {(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()} else {''}
   $checks += [pscustomobject]@{manifest=$namespace;section=$section;path=$entry.Name;exists=$exists;matches=($actual -eq $entry.Value);expected=$entry.Value;actual=$actual}
  }
 }
}
$checks | Export-Csv (Join-Path $audit 'MANIFEST_CHECKS.csv') -NoTypeInformation
$datasetRows = foreach ($name in @('z_alizadeh_sani','mendeley','five_states','kaggle_cvd_70000','uci_heart_disease_cleveland','statlog_heart','uci_heart_failure')) {
 $path = Join-Path $project "data/interim/validated/$name.csv"
 if (Test-Path -LiteralPath $path) {
  $rows = @(Import-Csv -LiteralPath $path)
  $cols = @($rows[0].PSObject.Properties.Name)
  [pscustomobject]@{dataset=$name;path=[IO.Path]::GetRelativePath($project,$path);rows=$rows.Count;columns=$cols.Count;exact_duplicate_rows=(@($rows | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1).Count - @($rows | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Sort-Object -Unique).Count);target_candidates=(@($cols | Where-Object {$_ -match 'Cath|target|HeartDisease|cardio|DEATH_EVENT|disease|num'}) -join ';')}
 }
}
$datasetRows | Export-Csv (Join-Path $audit 'DATASET_DIMENSIONS.csv') -NoTypeInformation
$metrics = @(Import-Csv (Join-Path $project 'results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv'))
$register = foreach ($row in $metrics) {
 foreach ($metric in @('roc_auc','pr_auc_average_precision','brier_score','log_loss','sensitivity','specificity','calibration_intercept','calibration_slope')) {
  [pscustomobject]@{experiment='Phase6';dataset='ZAS';procedure=$row.model_id;repeat=$row.repeat;metric=$metric;value=$row.$metric;protocol='3 repeats x 5 outer folds; 3 inner folds; complete-repeat OOF';source='results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv';producer='runner/phase6_zas_selection/run.py';status='saved-artifact reconciliation passed focused tests';limitations='source rows; overlapping repeats; selection-dependent calibration; no external validation'}
 }
}
$register | Export-Csv (Join-Path $audit 'EXPERIMENT_AND_METRIC_REGISTER.csv') -NoTypeInformation
function Write-Report($name,$body) { Set-Content -LiteralPath (Join-Path $audit $name) -Value $body -Encoding utf8 }
Write-Report 'COMPLETE_PROJECT_HISTORY.md' @'
# CARDIODIAG project history — evidence-led reconstruction

Audit date: 14 November 2026 (Asia/Kolkata). This is an initial reconstruction, not an assertion that every historical notebook or numerical claim has been independently audited. Phase sequence is supported by namespaces and decision reports; filesystem times are not research-event dates.

| Stage | Objective, implementation and evidence | Current interpretation |
|---|---|---|
| Original reproduction | Author notebooks and reference material under reference/; README describes Five States author-faithful and corrected Mendeley/Five States experiments | Historical reproduction, separate from ZAS; bibliography relationships not freshly verified |
| Restructuring | src/cardiodiag modules, runner scripts, tests, requirements files | Implemented research software; current root README describes early work and is not a complete current-state index |
| Dataset expansion | data/raw, data/interim/validated, metadata; collection and clinical audit runners | Multiple distinct datasets and endpoints; do not combine performance estimates |
| Research V1–V4 | runner/primary_research*, corresponding results and independent audit namespaces | Predecessor protocols retained; exact supersession requires detailed historical reconciliation |
| V3 comparator | Five risk/demographic features; primary_research_v3 outputs and independent audit | Historical V3 differs from Phase 6 re-evaluated V3-style procedure |
| Phase 2 | Staged feature, calibration and uncertainty outputs under results/phase2 | Historical staged investigation, not Phase 6 |
| Phase 3 | Cohort discovery and external-validation feasibility; PHASE3_FINAL_DECISION.md | Conditional go for acquisition; external validation not feasible with verified resources |
| Phase 4 / improvement / validation audit | Four-centre UCI reconstruction, centre-held-out evaluation and staged analyses | Separate UCI investigation; not external validation of ZAS |
| Phase 5 | ZAS feature-group benchmark; runner/phase5/zas_track/run.py | Winning 0.928 row post hoc; exploratory |
| Phase 5.1 | results/phase5_1_audit; saved-score and feature-timing review | Reproduction does not remove selection optimism |
| Final project consolidation | results/final_project and runner/final_project | Historical synthesis predates selection-aware primary result |
| Phase 6 | runner/phase6_zas_selection/run.py; 30 group-family candidates, 15 outer partitions | Completed source-row procedure evaluation; primary discrimination comparison |
| Phase 6.1 | runner/phase6_1_audit/audit.py; five complete audit reports read | Completed audit; confirms discrimination, documents unresolved calibration limitations; corrected metadata is a separate CSV |
| Phase 7.1 | results/phase7_1_evidence | Evidence consolidation; decision supports bounded retrospective framing |
| Phase 7.2 | results/phase7_2_methodology | Methodological review; no population superiority or significance claim supported |
| Phase 7.3 | paper/phase7_3_manuscript | Manuscript, traceability, tables and figures prepared |
| Phase 7.4 and 7.4.1 | results/phase7_4_final_qa, results/phase7_4_1_qa; corrected manuscript copy | Historical 126-package environment verification and 12-page ReportLab rendering documented; no LaTeX compile claim |
| Subsequent college package | paper/final_college_report: DOCX, PDF, builders and QA reports | Exists beyond handover's earlier draft; content audit describes six-page two-column paper style, not the requested full chapter-based college structure |

No historical prediction, manuscript or result was edited by this audit. New complete nested-model runs were not launched.
'@
Write-Report 'TECHNICAL_IMPLEMENTATION_INVENTORY.md' @'
# Technical inventory

Active research workspace: CARDIODIAG_FRESH. Parent workspace also contains extracted source material, archives and original CSVs. No Git worktree was established (git status reported not a repository). A nested CARDIODIAG_FRESH entry also exists; do not assume it is the active root.

FILE_INVENTORY.csv records paths, sizes and SHA-256 for project files excluding virtual environments and caches. This inventories artifacts; it does not certify every file's semantics.

| Area | Role |
|---|---|
| src/cardiodiag | Data quality/loading, preprocessing pipelines, imbalance sampling, feature selection, ensembles, optimization, evaluation, calibration, explainability, fairness, decision curves and paired statistics |
| runner | Historical reproduction, dataset acquisition/validation, corrected validation, research versions, phased experiments and independent audits |
| runner/phase6_zas_selection/run.py | Groups/grids, fold-local preprocessing, inner choice, selected-score crossfit, sigmoid and threshold, outer prediction, repeat aggregation |
| runner/phase6_zas_selection/finalize.py | Phase 6 finalization; separate from training runner |
| runner/phase6_1_audit/audit.py | Saved-prediction reconciliation and manifest verification |
| tests | Historical unit/structural tests plus Phase 6 and Phase 6.1 focused tests |
| data/raw, data/interim, data/processed, data/metadata | Original acquisitions, validated files, task-specific derivatives and provenance metadata |
| results | Distinct experimental/audit namespaces; names do not alone prove completion |
| reference, notebooks | Author-faithful material and notebook workflows; not executed in this audit |
| paper | Legacy manuscript, research versions, Phase 7 manuscript/corrected copy and later college package |
| requirements.txt, requirements-runner.txt | Ranged/unpinned working requirements |
| results/phase7_4_final_qa/requirements-pinned-working-env.txt | Historical pinned working-environment snapshot; not a general cross-platform lock |
| .venv, .venv-repro, .venv-phase7_4_1 | Different local environments; commands must identify which one is used |

No root pyproject.toml/uv.lock was established from the inspected active-root listing. Do not carry these forward as verified current dependency files merely because the handover mentions them historically.
'@
Write-Report 'DATASET_AND_PROVENANCE_AUDIT.md' @'
# Dataset and provenance audit

DATASET_DIMENSIONS.csv contains newly counted local CSV dimensions and exact serialized-row duplicates for validated datasets. Missingness normalization, near duplicates, cross-dataset overlap, licenses and patient identity are not independently resolved by these counts.

| Track | Evidence and restrictions |
|---|---|
| ZAS | Local validated CSV; Phase 6.1 tests reconcile source labels, partitions and predictions. Cath Cad=1/Normal=0; 303 source rows, 216 positive, 87 negative. Local 56 columns including target imply 55 other columns; source catalogue's 54-feature description remains unresolved. No verified independent-patient identifiers or feature acquisition times |
| Mendeley | Separate validated/raw CSVs and metadata; patientid excluded in corrected historical workflow according to README. Historical 1,000-row claim must be paired with current counted dimensions and original provenance |
| Five States/Heart5 | Separate historical reproduction track; parent heart5.csv and validated five_states.csv. File equivalence is not assumed solely from similar dimensions |
| Kaggle 70k | Separate cardiovascular endpoint and computationally reduced historical workflow; no transfer of its scores into Phase 6 |
| UCI four-centre | 920 records documented in manuscript/audits; 509 positive/411 negative, num 0 versus 1–4. Four-centre source reconstruction is distinct from validated Cleveland-only CSV |
| Other resources | Statlog, heart failure and ECG/clinical metadata exist; distinct outcomes and modalities. Their presence does not make them compatible external cohorts |

Original paper/repository/Mendeley/Zenodo relationships and bibliography were not freshly checked against authoritative online sources in this initial local pass. They remain unresolved rather than accepted as final citations. Technical local verification is separate from publication permission or ethics determinations.
'@
Write-Report 'PHASE_COMPLETION_MATRIX.md' @'
# Phase completion matrix

| Work | Evidence status | Remaining |
|---|---|---|
| Historical implementation/reproduction | Source/results exist; README and namespaces inspected | Full original-notebook and all-score reconciliation not performed |
| Phase 3 | Final decision read | Compatible independent cohort acquisition |
| Phase 4 | Runners, outputs and later manuscript summaries exist | Independent repeat of all UCI reconstruction/metric checks in this audit |
| Phase 5 / 5.1 | Benchmark/audit artifacts exist | Preserve exploratory status; complete all-score register |
| Phase 6 | Implementation inspected; focused tests passed | Generalization, timing and independent patient evidence absent |
| Phase 6.1 | All five named Markdown audit reports read; focused reconciliation tests passed | Calibration design is documented, not scientifically corrected by new predictions |
| Phase 7.1 | Decision and manifest inspected | Recount all 75 historical evidence entries and verify every association |
| Phase 7.2 | Decision read | Human/source uncertainties persist |
| Phase 7.3 / 7.4.1 | Corrected manuscript inspected; historical QA and rendering decisions read | Journal-specific export, author declarations, fresh visual QA; LaTeX not verified |
| Later college report | DOCX/PDF/builders and content audit exist | Confirm institutional format; six-page paper format does not satisfy chapter-based report by itself |
| This handover audit | Initial local reconstruction, inventory, hashes and primary tests | Not a completed exhaustive audit of every historical number, claim, notebook, citation and rendered page |
'@
Write-Report 'SCIENTIFIC_VALIDITY_AUDIT.md' @'
# Scientific validity audit

The defensible question remains retrospective discrimination of angiography-defined CAD using recorded clinical features.

Phase 6 source inspection finds candidate selection and tuning in GridSearchCV within outer training. Pipeline-local imputation, encoding, scaling and optional splines are refit within training folds. Six groups and five family labels yield 30 candidates. The first three repeats of frozen five-fold outer assignments are used; operational inner folds are generated anew with 61051 + 100*repeat + fold and n_splits=3, despite the older source split manifest containing four-fold inner assignments. Reusing the outer assignments does not mean reusing its old inner folds.

Selection precedes crossfit_selected_scores. Held-out calibration-fold labels have influenced group/family/hyperparameter selection. Scores are estimator-out-of-fold, not selection-independent. Sigmoid fitting and Youden threshold selection share these training scores. Inner scoring fits use about two-thirds of outer training, while final prediction uses the full outer-training refit. These limitations remain in the historical procedure. The metadata range(1,5) produces four labels for three operational folds; Phase 6.1 supplies a separate corrected display, not a new calibration experiment.

No outer-test-label path into selection/calibrator/threshold was identified in the inspected runner. Outer-test permutation importance is post-fit descriptive analysis. The independent focused tests confirm saved coverage, labels, candidate ranking metadata, fold/repeat metrics and hashes. This supports the recorded outer discrimination estimate of the implemented procedure; it does not independently reproduce model fitting from scratch.

Mean complete-repeat AUC is 0.910316 for selection-aware versus 0.736315 for the re-evaluated V3-style procedure; mean difference 0.174001. Comparator features/family are fixed, but C is inner-tuned and calibration is refitted. It is not an untouched original fitted model. Averaging fold AUCs would give different values and is not used for the primary repeat result.

Three overlapping repeats do not justify independent-replicate t inference. The reported interval is descriptive split variation; do not claim population significance, patient-level superiority or clinical benefit. Pooling repeat OOF scores also compares scores from different fitted fold models, so interpret this aggregation as the recorded procedure estimand, not one fixed-model ROC curve.

Timing, patient independence, schema provenance, compatible external validation and clinical utility remain unresolved. Phase 5's post-hoc winning row is exploratory. UCI is separate. Existing checklists do not establish TRIPOD+AI compliance or low PROBAST+AI risk.
'@
Write-Report 'MANUSCRIPT_CONSISTENCY_AUDIT.md' @'
# Manuscript consistency audit — initial pass

Inspected paper/phase7_4_1_final_manuscript/MANUSCRIPT.md against Phase 6/6.1 code, tables, decisions and focused tests. This is not an exhaustive claim-by-claim external bibliographic verification or a fresh rendered-page inspection.

| Claim/issue | Assessment |
|---|---|
| Repeat AUCs and paired differences | Consistent with inspected saved table and passing metric-reconciliation tests |
| Calibration selection dependence and training-size mismatch | Explicitly disclosed; no claim of correction |
| V3 comparator | Correctly described as matched re-evaluation with inner C tuning |
| Patient unit, uncertainty and timing | Correctly bounded to source rows, descriptive split variation and uncertain timing |
| UCI | Clearly separate; no ZAS external-validation claim |
| Phase 5 0.928 | Explicitly exploratory/post hoc |
| Author list | Corrected manuscript lists only Ch Hari Krishna and G Vennela as names awaiting confirmation; user handover additionally lists Akhil Preetham Erukudinnela. Human reconciliation required; do not infer final order |
| Environment wording | Manuscript says working requirements are not fully pinned; later QA documents a frozen snapshot. Explain distinction in a future isolated revision, without implying original requirements were locked |
| Candidate prespecification | Source asserts frozen pool, and outer tests remained outside operational selection; independence from all prior dataset exploration is not established by within-run nesting alone |
| Bibliography and source relationships | Not freshly verified online in this pass; no new claim of citation correctness |
| Layout | Historical ReportLab 12-page QA report exists; LaTeX compilation unverified |
| College report | Current content audit describes a six-page paper-style output. Full chapter-based institutional report requirements remain a separate task |

Do not rewrite historical manuscript files during audit. Any later corrections should use a new version and traceable manifest.
'@
Write-Report 'REPRODUCIBILITY_AUDIT.md' @'
# Reproducibility audit

New checks: eight focused Phase 6/6.1 tests passed in 15.29 seconds using .venv, PYTHONPATH=src;., PYTHONDONTWRITEBYTECODE=1 and pytest cache disabled. The sandboxed interpreter launcher initially could not create its base Python process; the approved outside-sandbox retry succeeded. No full nested experiment or benchmark was launched.

Command from CARDIODIAG_FRESH:

    .\.venv\Scripts\python.exe -m pytest tests\test_phase6_1_audit.py tests\test_phase6_zas_selection.py -q -p no:cacheprovider

MANIFEST_CHECKS.csv independently checks supported direct path-to-SHA256 inputs/outputs mappings in selected manifests. Unsupported nested/list manifest schemas are not silently counted as verified. FILE_INVENTORY.csv inventories project artifacts outside environments/caches. A matching hash proves current byte consistency with a listed hash, not scientific correctness or provenance authenticity.

Historical Phase 7.4.1 reports a 126-package clean install, pip check and four smoke tests; those installation checks were not rerun here. Working requirements are ranged/unpinned; a frozen working-environment snapshot exists separately. Full Ruff, all tests, all notebooks, all manifests and manuscript rendering were not rerun in this pass. No Git worktree exists at the inspected workspace root, so no branch/commit/diff assertion is possible.

The historical 12-page review PDF used ReportLab and a separate local document runtime, not a LaTeX compile or journal template. Its rendering toolchain is not included in the 126-package project snapshot. Fresh document visual verification remains outstanding.
'@
Write-Report 'UNRESOLVED_ISSUES.md' @'
# Unresolved issues and prioritized actions

| Priority | Issue | Owner/type | Next reviewable deliverable |
|---|---|---|---|
| P0 | Author list omits Akhil in corrected manuscript; author order/contributions unapproved | Human | Approved names, affiliations, contributions and corresponding author |
| P0 | Ethics, permissions, funding/conflicts and registration/PPI statements | Human/institution | Evidence-backed declaration sheet; do not invent none/exempt |
| P0 | ZAS 54 versus 55 predictor discrepancy and source lineage | Technical/source | Column-by-column source-version mapping and verified dictionary |
| P0 | Feature acquisition timing and patient identity | Source/institution | Source evidence, or explicit retained limitations |
| P1 | Calibration is selection-dependent; no corrected predictive experiment established | Methodological | Prospective design memo for selection-independent calibration; compute estimate before any expensive rerun |
| P1 | Remaining historical numbers and all manuscript claims not freshly checked | Technical | Expanded experiment/claim register with recalculations and authoritative citations |
| P1 | Original paper/repository/dataset/Zenodo references unverified in this pass | Technical/web | Authoritative bibliographic relationship audit |
| P1 | Journal and institutional report format unconfirmed | Human | Chosen journal/template and approved full report structure |
| P1 | College package currently described as six-page paper style | Human/document | Chapter-based report mapped to institutional sample, if required |
| P2 | Environment portability, full lint/tests and notebook validation | Technical | Isolated reproducibility report; preserve historical outputs |
| P2 | Journal-specific typesetting and fresh PDF visual QA | Technical/document | Compiled/rendered package and page-by-page verification |
| P2 | External generalization and clinical utility absent | New evidence | Separate compatible cohort protocol; not required to fabricate a positive publication conclusion |

No new model is warranted simply to raise AUC. Calibration corrections may change probabilities/threshold metrics and sometimes ranking across fold-model score scales; preserve existing predictions and save corrections under a new namespace. Before proposing implementation, specify affected scripts/protocol/manuscript copy, costs, new artifacts and checks for disjointness, coverage, metadata and unchanged historical hashes.
'@
$mismatches = @($checks | Where-Object { !$_.matches })
Write-Report 'FINAL_RESEARCH_HANDOVER.md' @"
# CARDIODIAG research handover — initial verified state

The active workspace is CARDIODIAG_FRESH. Preserve the retrospective angiography-defined CAD research direction and all historical outputs.

The Phase 6 discrimination comparison is supported by saved artifacts and eight focused tests passed now: mean repeat ROC-AUC approximately 0.9103 versus 0.7363, paired difference 0.1740. This evaluates procedures on source rows, not verified independent patients. Calibration is selection-dependent and training-size mismatched; Phase 6.1 documented it and corrected metadata separately, without generating a new independently calibrated result.

Phase 7 evidence/methodology/manuscript/QA packages exist. Historical QA reports a clean pinned snapshot and 12-page ReportLab rendering; neither final submission readiness nor LaTeX compilation follows. A later college DOCX/PDF package also exists, but its content audit describes six-page paper formatting. Author/institution details and the desired full college-report format need reconciliation. The corrected manuscript currently omits Akhil from its provisional name list.

New artifact inventory: $($inventory.Count) files outside environments/caches. Supported selected-manifest hash checks: $($checks.Count); mismatches or missing targets: $($mismatches.Count). See MANIFEST_CHECKS.csv for individual results and coverage; unmatched historical hashes require investigation before claiming intact packages.

This initial audit has not completed exhaustive verification of every historical score, all 75 evidence associations, external references, all source/protocol details, notebooks, or every rendered page. The metric register currently covers 48 primary Phase 6 repeat/metric records; expand it before claiming every major experiment is registered. Publication-ready status is therefore not established. The Phase 6 source-row discrimination evidence is the strongest currently checked contribution.

Next priorities: resolve author/governance facts; finish source/schema and claim/citation reconciliation; decide whether to retain bounded calibration limitations or commission a separately designed correction; confirm college/journal templates; then prepare isolated final document versions and visual QA. No historical manuscript or experiment output was overwritten.
"@
$outputs = @(Get-ChildItem -LiteralPath $audit -File | Where-Object {$_.Name -ne 'FINAL_AUDIT_MANIFEST.json'} | ForEach-Object {[pscustomobject]@{path=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}})
@{audit_date_local='2026-11-14';timezone='Asia/Kolkata';scope='Initial local reconstruction; exhaustive audit incomplete';project_root=$project;focused_tests='8 passed in 15.29s';inventory_count=$inventory.Count;manifest_checks=$checks.Count;manifest_mismatches=$mismatches.Count;predictive_training_runs=0;historical_outputs_intentionally_modified=$false;outputs=$outputs} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $audit 'FINAL_AUDIT_MANIFEST.json') -Encoding utf8
Write-Output "Inventory=$($inventory.Count); hash checks=$($checks.Count); mismatches=$($mismatches.Count)"
