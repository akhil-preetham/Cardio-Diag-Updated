"""Audit the paper-style report and write Phase 8 audit/manifest files.

This script reads saved results and the rendered report only. It does not fit a
model, run a benchmark, or modify historical experiment namespaces.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import docx
from docx import Document
from pypdf import PdfReader
import pypdf


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOCX = OUT / "CARDIODIAG_FINAL_PROJECT_REPORT.docx"
PDF = OUT / "CARDIODIAG_FINAL_PROJECT_REPORT.pdf"
QA_PAGES = OUT / "qa" / "word_pages_final_paper"
STYLE_REFERENCE = Path(r"C:\Users\akhil\Downloads\CARDIODIAG_6_8_Page_Research_Paper.pdf")


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_report() -> tuple[dict, Document, PdfReader, list[str], str]:
    assert DOCX.is_file() and PDF.is_file(), "Build DOCX and export PDF first."
    doc = Document(DOCX)
    pdf = PdfReader(PDF)
    pages = len(pdf.pages)
    assert 6 <= pages <= 8, f"Expected 6–8 pages, found {pages}."
    assert len(doc.sections) >= 3, f"Expected title, full-width figure and two-column sections; found {len(doc.sections)}."
    assert round(doc.sections[0].page_width.inches, 2) == 8.50
    assert round(doc.sections[0].page_height.inches, 2) == 11.00
    assert len(doc.tables) == 9, f"Expected 9 report tables, found {len(doc.tables)}."
    assert len(doc.inline_shapes) == 2, f"Expected two report figures, found {len(doc.inline_shapes)}."
    assert doc.styles["Normal"].font.name == "Times New Roman"
    assert round(doc.styles["Normal"].font.size.pt, 1) == 10.0

    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    for p in paragraphs:
        for run in p.runs:
            if not run.text.strip():
                continue
            assert run.font.name == "Times New Roman", f"Unexpected font: {run.font.name!r}"
            assert run.font.size is not None, f"Missing explicit size: {run.text[:40]!r}"
            color = run.font.color.rgb
            assert color is not None and str(color).upper() == "000000", f"Non-black text: {run.text[:40]!r}"

    body = "\n".join(p.text for p in doc.paragraphs)
    table_text = "\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    full_text = body + "\n" + table_text
    required = [
        "I. INTRODUCTION", "II. DATA AND STUDY DESIGN", "III. METHODS",
        "IV. RESULTS", "V. DISCUSSION", "VI. CONCLUSION", "DECLARATIONS", "REFERENCES",
        "Retrospective discrimination of angiography-defined CAD using recorded clinical features",
        "[AUTHOR NAMES AND ORDER REQUIRE CONFIRMATION]",
        "No predictor has source-confirmed availability before the diagnostic decision",
        "The Phase 5 0.928 result was not used to alter the pool",
        "selection-dependent", "Age, Sex, Current Smoker, EX-Smoker and FH",
        "TRIPOD+AI reporting checklist and a PROBAST+AI applicability assessment",
        "the evidence supports a retrospective discrimination report only",
    ]
    for phrase in required:
        assert phrase.casefold() in full_text.casefold(), f"Missing report element: {phrase}"
    table_captions = re.findall(r"TABLE\s+([IVX]+)\.", body)
    assert len(table_captions) == 9, f"Expected 9 table captions, found {len(table_captions)}."
    assert set(table_captions) == {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
    assert "Fig. 1." in body and "Fig. 2." in body

    refs_start = body.rfind("REFERENCES")
    assert refs_start >= 0
    citations = {
        number
        for citation in re.findall(r"\[(\d+(?:,\d+)*)\]", body[:refs_start])
        for number in citation.split(",")
    }
    references = set(re.findall(r"^\[(\d+)\]", body[refs_start:], flags=re.M))
    assert citations == {str(i) for i in range(1, 9)}, citations
    assert references == citations, (references, citations)

    zas = read_csv("data/interim/validated/z_alizadeh_sani.csv")
    counts: dict[str, int] = {}
    for row in zas:
        counts[row["Cath"]] = counts.get(row["Cath"], 0) + 1
    assert len(zas) == 303 and counts == {"Cad": 216, "Normal": 87}
    assert len(zas[0]) == 56
    timing = read_csv("results/phase5_1_audit/ZAS_FEATURE_TIMING_TABLE.csv")
    assert len(timing) == 38
    assert sum(r["classification"] == "A" for r in timing) == 2
    assert sum(r["classification"] == "B" for r in timing) == 36

    repeat_rows = read_csv("results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv")
    selection = [r for r in repeat_rows if r["model_id"] == "selection_aware_procedure"]
    v3 = [r for r in repeat_rows if r["model_id"] == "frozen_v3_logistic_balanced"]
    assert len(selection) == len(v3) == 3
    selection_auc = [float(r["roc_auc"]) for r in selection]
    v3_auc = [float(r["roc_auc"]) for r in v3]
    paired = [a - b for a, b in zip(selection_auc, v3_auc)]
    assert [round(x, 4) for x in selection_auc] == [0.9134, 0.8999, 0.9176]
    assert [round(x, 4) for x in v3_auc] == [0.7370, 0.7420, 0.7300]
    assert round(mean(paired), 3) == 0.174
    assert round(min(paired), 3) == 0.158 and round(max(paired), 3) == 0.188
    split_ci = (
        mean(paired) - 4.302652729911275 * stdev(paired) / math.sqrt(len(paired)),
        mean(paired) + 4.302652729911275 * stdev(paired) / math.sqrt(len(paired)),
    )
    assert tuple(round(x, 3) for x in split_ci) == (0.137, 0.211), split_ci
    secondary = {
        model: {field: round(mean(float(r[field]) for r in rows), 4)
                for field in ("pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity")}
        for model, rows in (("selection", selection), ("v3", v3))
    }
    assert secondary["selection"] == {
        "pr_auc_average_precision": 0.9573, "brier_score": 0.1076, "log_loss": 0.3467,
        "sensitivity": 0.8318, "specificity": 0.8314,
    }
    assert secondary["v3"] == {
        "pr_auc_average_precision": 0.8469, "brier_score": 0.1746, "log_loss": 0.5286,
        "sensitivity": 0.6898, "specificity": 0.7088,
    }

    calibration = {
        name: {
            "intercept": round(mean(float(r["calibration_intercept"]) for r in rows), 3),
            "slope": round(mean(float(r["calibration_slope"]) for r in rows), 3),
        }
        for name, rows in (("selection", selection), ("v3", v3))
    }
    assert calibration == {"selection": {"intercept": 0.173, "slope": 0.868}, "v3": {"intercept": -0.007, "slope": 1.005}}
    family_rows = read_csv("results/phase6_zas_selection/MODEL_SELECTION_FREQUENCY.csv")
    assert sum(int(r["selection_count"]) for r in family_rows) == 15
    assert {r["selected_model_family"]: int(r["selection_count"]) for r in family_rows} == {
        "elastic_net": 4, "logistic_l2": 9, "spline_logistic": 2,
    }
    fold_rows = read_csv("results/phase6_zas_selection/PERFORMANCE_COMPARISON.csv")
    fold_auc = {
        model: [float(r["roc_auc"]) for r in fold_rows if r["model_id"] == model]
        for model in ("selection_aware_procedure", "frozen_v3_logistic_balanced")
    }
    assert round(min(fold_auc["selection_aware_procedure"]), 4) == 0.8372
    assert round(max(fold_auc["selection_aware_procedure"]), 4) == 0.9799
    assert round(min(fold_auc["frozen_v3_logistic_balanced"]), 4) == 0.6375
    assert round(max(fold_auc["frozen_v3_logistic_balanced"]), 4) == 0.8579

    uci = read_csv("results/phase4_uci_multicentre/POOLED_OUT_OF_CENTRE_METRICS.csv")
    strict = next(r for r in uci if r["feature_set"] == "strict" and r["model"] == "logistic")
    assert round(float(strict["roc_auc"]), 4) == 0.6753
    assert round(float(strict["pr_auc"]), 4) == 0.6809
    assert round(float(strict["brier"]), 4) == 0.2254
    centres = read_csv("results/phase4_uci_multicentre/CENTRE_VALIDATION_METRICS.csv")
    strict_centres = [r for r in centres if r["feature_set"] == "strict" and r["model"] == "logistic"]
    assert len(strict_centres) == 4
    stage_rows = read_csv("paper/phase7_4_1_final_manuscript/TABLES/TABLE4_UCI_STAGES.csv")
    assert len(stage_rows) == 4
    stage_auc = [round(float(r["Pooled ROC-AUC"]), 4) for r in stage_rows]
    assert stage_auc == [0.7179, 0.8225, 0.8215, 0.8450]
    abstract = next((p.text for p in doc.paragraphs if p.text.startswith("Abstract")), "")
    abstract_values = ("0.9134", "0.8999", "0.9176", "0.7370", "0.7420", "0.7300", "0.174", "0.137–0.211")
    assert abstract, "Structured abstract is missing."
    assert all(value in abstract for value in abstract_values), "Abstract repeat metrics or paired interval do not match saved results."
    assert round(float(strict["roc_auc"]), 3) == 0.675
    assert "0.675" in abstract, "Abstract UCI AUC does not match the rounded strict logistic result."

    page_texts = [page.extract_text() or "" for page in pdf.pages]
    assert all(len(text.strip()) > 200 for text in page_texts), "An almost-empty PDF page was found."
    pdf_text = "\n".join(page_texts)
    for value in ("0.9134", "0.8999", "0.9176", "0.7370", "0.7420", "0.7300", "0.174", "0.137", "0.211", "0.6753", "0.9602", "0.1016", "0.173", "0.868"):
        assert value in pdf_text, f"Expected numerical claim missing from PDF: {value}"
    visual_pages = sorted(QA_PAGES.glob("page-*.png"), key=lambda p: int(re.search(r"page-(\d+)", p.stem).group(1)))
    assert len(visual_pages) == pages, f"Expected {pages} page renders, found {len(visual_pages)}."

    checks = {
        "pdf_pages": pages,
        "page_size": "US Letter",
        "sections": len(doc.sections),
        "figures": len(doc.inline_shapes),
        "tables": len(doc.tables),
        "inline_citations": sorted(int(x) for x in citations),
        "zas_rows": len(zas),
        "zas_labels": counts,
        "selection_auc_by_repeat": [round(x, 4) for x in selection_auc],
        "v3_style_auc_by_repeat": [round(x, 4) for x in v3_auc],
        "mean_paired_repeat_difference": round(mean(paired), 4),
        "paired_split_variation_interval": [round(x, 3) for x in split_ci],
        "abstract_numeric_claims": {
            "selection_auc": [round(x, 4) for x in selection_auc],
            "v3_style_auc": [round(x, 4) for x in v3_auc],
            "mean_paired_difference": round(mean(paired), 3),
            "split_variation_interval": [round(x, 3) for x in split_ci],
            "uci_strict_logistic_auc_rounded": round(float(strict["roc_auc"]), 3),
            "status": "checked against complete-repeat source metrics and strict UCI pooled metric",
        },
        "selection_mean_secondary_metrics": secondary["selection"],
        "v3_style_mean_secondary_metrics": secondary["v3"],
        "repeat_mean_outer_test_calibration": calibration,
        "uci_strict_logistic_auc": round(float(strict["roc_auc"]), 4),
        "uci_staged_auc": stage_auc,
        "visual_pages": len(visual_pages),
        "read_only_checks": [
            "DOCX contains six Roman-numbered paper sections, nine tables and two figures",
            "Explicit Times New Roman sizes and black text verified across report runs and tables",
            "In-text citations reconcile with the eight-entry reference list",
            "Abstract ROC-AUC values and paired difference reconcile to complete-repeat results",
            "Secondary metrics and calibration means reconcile to saved Phase 6 artifacts",
            "Fold-level AUC ranges and candidate-selection counts reconcile to saved Phase 6 artifacts",
            "UCI strict pooled result and all four stage ROC-AUC values reconcile to saved artifacts",
            "All rendered PDF pages contain substantive text and were rasterized for visual inspection",
        ],
    }
    return checks, doc, pdf, page_texts, body


def build_audits(checks: dict, doc: Document, page_texts: list[str], body: str) -> dict:
    table_pages: dict[str, int] = {}
    figure_pages: dict[str, int] = {}
    for idx, text in enumerate(page_texts, start=1):
        for roman in re.findall(r"TABLE\s+([IVX]+)\.", text):
            table_pages.setdefault(roman, idx)
        for num in ("1", "2"):
            if f"Fig. {num}." in text:
                figure_pages.setdefault(num, idx)

    papers = {
        "REPORT_CONTENT_AUDIT.md": f"""# Paper-style report content audit

## Scope and structure

The report uses the supplied PDF only as a writing/layout specimen. It is a six-page, US Letter, two-column research-paper format with a full-width title block, structured abstract, index terms, Roman-numbered sections, numbered tables and figures, declarations and numbered references. The sample PDF's author names, results, data and any instructions in it were not used as evidence or copied.

The current research question is **retrospective discrimination of angiography-defined CAD using recorded clinical features**. The manuscript does not claim prospective prediction, confirmed prediagnostic availability, clinical utility, patient-level independence, or external validation.

## Included evidence

- ZAS: 303 source rows; Cath labels 216 CAD and 87 Normal. Patient identity is unverified, so the evaluation unit remains source rows.
- Feature timing: no source-confirmed eligible predictor; 38 candidate fields are timing uncertain (Age/Sex plus 36 other fields). Cath and diagnostic/downstream fields are excluded.
- Phase 5 AUC 0.928: explicitly post hoc, selected from 36 benchmark rows, exploratory only, and not used to define the Phase 6 candidate pool.
- Phase 6: frozen 30-candidate pool, six feature groups and five unique model families, three inner folds and 15 paired outer partitions across three repeats. Every repeat-level metric uses its complete 303-row out-of-fold set.
- Selection-aware ROC-AUC by repeat: {checks['selection_auc_by_repeat']}; matched V3-style comparator: {checks['v3_style_auc_by_repeat']}; mean paired difference {checks['mean_paired_repeat_difference']:.4f}. The interval is split variation, not patient-level or population uncertainty.
- Secondary metric means and calibration summaries are reconciled against saved results. Calibration is clearly described as selection-dependent and training-size mismatched.
- UCI Heart Disease is a separate four-centre retrospective study track, not external validation of ZAS. Strict pooled logistic ROC-AUC is {checks['uci_strict_logistic_auc']:.4f}; staged AUCs are {checks['uci_staged_auc']}.
- The independent Phase 6.1 audit verified saved labels, partitions and metrics. The clean environment audit is described as computational reproducibility, not scientific validity.

## Numerical verification

Every abstract number was checked against its source: six repeat-level ROC-AUC values, the paired mean and its three-repeat split-variation interval, and the rounded strict UCI pooled ROC-AUC. Fold-level ranges, model-selection counts, secondary metrics and calibration summaries were also checked. No pooled-fold AUC average was substituted for a complete-repeat AUC. The Phase 5 value is labelled exploratory throughout.

| Abstract claim | Printed value | Reconciled source |
|---|---:|---|
| Selection-aware repeat ROC-AUC | {checks['abstract_numeric_claims']['selection_auc']} | `REPEAT_LEVEL_PERFORMANCE.csv`, complete 303-row OOF sets |
| V3-style repeat ROC-AUC | {checks['abstract_numeric_claims']['v3_style_auc']} | `REPEAT_LEVEL_PERFORMANCE.csv`, same repeats and rows |
| Mean paired ROC-AUC difference | {checks['abstract_numeric_claims']['mean_paired_difference']:.3f} | Paired differences across three repeats |
| Split-variation interval | {checks['abstract_numeric_claims']['split_variation_interval']} | Recomputed t interval over three repeat-level differences; not patient-level uncertainty |
| UCI strict pooled logistic ROC-AUC | {checks['abstract_numeric_claims']['uci_strict_logistic_auc_rounded']:.3f} | `POOLED_OUT_OF_CENTRE_METRICS.csv` strict/logistic value rounded to three decimals |

## Remaining limits

Author and institutional declarations, patient identity, feature acquisition times, the local ZAS schema discrepancy and clinical data-use documentation remain unresolved. These are not inferred from the style-reference PDF or from populated reporting checklists.
""",
        "REPORT_FORMATTING_AUDIT.md": f"""# Paper-style report formatting and rendering audit

## Applied layout

- {checks['pdf_pages']} US Letter portrait pages; full-width centered title/author block followed by two-column paper text, including Roman-numeral sections, figures, tables and references.
- Main body Times New Roman 10 pt, black, single-spaced. Compact tables and references use smaller explicitly sized Times New Roman text. All document text runs and table runs were checked for font, size and black color.
- Two figures: a full-width workflow schematic and a repeat-level AUC comparison. Nine numbered tables cover timing, candidate groups, search grids, repeat-level performance, selection/feature results and the UCI track.
- Table headers repeat across continuations; rows are protected against splitting. The workflow and AUC figure captions are present in the rendered PDF.
- The supplied Downloads PDF was used for formatting only. It is not the project's results source.

## Render verification

Microsoft Word for Microsoft 365 exported the editable DOCX to PDF. Poppler reports US Letter pages. All {checks['pdf_pages']} pages were rasterized at 120 dpi to `qa/word_pages_final_paper/` and visually inspected for column flow, figure paths, labels, table continuation, glyphs, clipping and references. The report has no intentionally blank page. The final references/declarations page is shorter than the body pages because author declarations and the verified source list are concise; it remains readable and within the requested six-to-eight-page range.

## Reproduction commands (PowerShell from repository root)

```powershell
$py = 'C:\\Program Files\\WindowsApps\\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\\dependencies\\python\\python.exe'
& $py 'paper/final_college_report/build_report.py'
$docx = (Resolve-Path 'paper/final_college_report/CARDIODIAG_FINAL_PROJECT_REPORT.docx').Path
$pdf = Join-Path (Split-Path $docx -Parent) 'CARDIODIAG_FINAL_PROJECT_REPORT.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$document = $word.Documents.Open($docx, $false, $true)
$document.ExportAsFixedFormat($pdf, 17)
$document.Close($false)
$word.Quit()
& 'C:\\Program Files\\WindowsApps\\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\\dependencies\\native\\poppler\\Library\\bin\\pdfinfo.exe' $pdf
& $py 'paper/final_college_report/build_final_package.py'
```

The package audit is read-only with respect to historical experiment namespaces; no model fitting or benchmark was run for this report revision.
""",
        "REPORT_CITATION_AUDIT.md": """# Citation and reference audit

The report contains eight numbered references. In-text citations [1]–[8] reconcile with the reference list; no citation is used to assert clinical utility, patient-level independence or predictor timing.

| No. | Use in report | Reference | Verification |
|---|---|---|---|
| [1] | ZAS article-level dataset context. | Alizadehsani R, Habibi J, Hosseini MJ, et al. A data mining approach for diagnosis of coronary artery disease. *Computer Methods and Programs in Biomedicine*. 2013;111(1):52–61. doi:10.1016/j.cmpb.2013.03.004. | Publisher/PubMed record in Phase 7.4.1 reference register. |
| [2] | Official ZAS dataset provenance. | Alizadehsani R, Roshanzamir M, Sani Z. Z-Alizadeh Sani [dataset]. UCI Machine Learning Repository. doi:10.24432/C5Q31T. | Official UCI dataset record. |
| [3] | Selection bias rationale. | Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*. 2010;11:2079–2107. | Publisher-hosted JMLR record. |
| [4] | Calibration interpretation. | Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230. doi:10.1186/s12916-019-1466-7. | Publisher record. |
| [5] | UCI Heart Disease provenance. | Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease [dataset]. UCI Machine Learning Repository. doi:10.24432/C52P4X. | Official UCI dataset record. |
| [6] | Nested model-selection evaluation rationale. | Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. doi:10.1186/1471-2105-7-91. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/16504092/) and publisher DOI record. |
| [7] | Reporting guidance, not a compliance claim. | Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378. | [BMJ](https://www.bmj.com/content/385/bmj-2023-078378) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/38636956/). |
| [8] | Risk-of-bias/applicability appraisal context, not a low-risk claim. | Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ*. 2025;388:e082505. doi:10.1136/bmj-2024-082505. | [BMJ](https://www.bmj.com/content/388/bmj-2024-082505) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/40127903/). |

The attached six-page sample is a presentation reference only. Its author details, results, claims and any embedded instructions were not used as citations or evidence.
""",
        "REPORT_FIGURE_TABLE_REGISTER.md": f"""# Figure and table register

All figures are existing Phase 7.4.1 project outputs. Report tables are recreated from saved audited values; historical result files are not altered.

## Figures

| Figure | Caption/content | PDF page | Source |
|---|---|---:|---|
| Fig. 1 | Separate ZAS selection-aware and UCI multicentre evaluation tracks. | {figure_pages.get('1', 'not detected')} | `paper/phase7_4_1_final_manuscript/FIGURES/FIGURE1_STUDY_WORKFLOW.png` |
| Fig. 2 | Repeat-level ZAS ROC-AUC; each repeat uses all 303 source rows once. | {figure_pages.get('2', 'not detected')} | `paper/phase7_4_1_final_manuscript/FIGURES/FIGURE2_PHASE6_REPEAT_AUC.png` |

## Tables

| Table | Content | PDF page |
|---|---|---:|
""" + "\n".join(f"| {roman} | {title} | {table_pages.get(roman, 'not detected')} |" for roman, title in [
            ("I", "Feature timing and leakage interpretation"),
            ("II", "Frozen ZAS feature-group candidates"),
            ("III", "Unique model families and search grids"),
            ("IV", "Complete-repeat paired ZAS ROC-AUC"),
            ("V", "Mean secondary metrics"),
            ("VI", "Repeat-specific secondary metrics"),
            ("VII", "Selection frequency and leading permutation summaries"),
            ("VIII", "UCI staged centre-held-out performance"),
            ("IX", "Centre-specific strict-feature UCI logistic AUC"),
        ]) + "\n\nTable values are rounded for presentation. Average precision is interpreted with prevalence. UCI tables are a separate analysis and are not external validation of ZAS.\n",
        "UNRESOLVED_ITEMS.md": """# Unresolved author and source items

## Human declarations

- Author names, order, contributions, student identifier and contact details.
- Department, institution, academic year, supervisor/guide, Head of Department, and institution-approved certificate/declaration wording.
- Acknowledgements, if applicable.
- Ethics approval, waiver or exemption determination, as applicable to the data and institutional policy.
- Data-use permissions and repository license compatibility for the intended distribution.
- Funding/support, conflicts of interest, and protocol registration or a verified statement that none applies.

## Evidence limits

- No verified patient identifier is available for ZAS; the unit is source rows and patient-level independence remains unestablished.
- No source timestamps establish whether candidate predictors were available before the angiography decision. Do not promote timing-uncertain fields without source evidence.
- The local ZAS CSV schema (56 columns including Cath) differs from the UCI catalogue feature count; exact source-version reconciliation remains pending.
- The UCI four-centre results are separate historical multicentre analyses and not external validation of ZAS.
- The attached CARDIODIAG PDF was used only as a paper-layout example; it was not used as evidence or a source of author names/results.

These gaps must be resolved from source documentation or human confirmation; they are not fixed by report formatting or checklist population.
""",
    }

    for name, content in papers.items():
        (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")

    generated = [
        "CARDIODIAG_FINAL_PROJECT_REPORT.docx", "CARDIODIAG_FINAL_PROJECT_REPORT.pdf",
        "REPORT_CONTENT_AUDIT.md", "REPORT_FORMATTING_AUDIT.md", "REPORT_CITATION_AUDIT.md",
        "REPORT_FIGURE_TABLE_REGISTER.md", "UNRESOLVED_ITEMS.md", "build_report.py",
        "build_paper_report.py", "build_final_package.py",
    ]
    generated_hashes = [
        {"path": (OUT / name).relative_to(ROOT).as_posix(), "sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
        for name in generated
    ]
    visual_files = sorted(QA_PAGES.glob("page-*.png"), key=lambda p: int(re.search(r"page-(\d+)", p.stem).group(1)))
    visual_hashes = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in visual_files]

    source_paths = [
        "data/interim/validated/z_alizadeh_sani.csv",
        "results/phase5_1_audit/ZAS_FEATURE_TIMING_AUDIT.md",
        "results/phase5_1_audit/ZAS_FEATURE_TIMING_TABLE.csv",
        "runner/phase5/zas_track/run.py",
        "runner/phase6_zas_selection/run.py",
        "results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md",
        "results/phase6_zas_selection/PHASE6_FINAL_DECISION.md",
        "results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv",
        "results/phase6_zas_selection/PAIRED_REPEAT_DIFFERENCES.csv",
        "results/phase6_zas_selection/PERFORMANCE_COMPARISON.csv",
        "results/phase6_zas_selection/MODEL_SELECTION_FREQUENCY.csv",
        "results/phase6_zas_selection/FEATURE_SELECTION_FREQUENCY.csv",
        "results/phase6_zas_selection/FEATURE_PERMUTATION_IMPORTANCE.csv",
        "results/phase6_1_audit/PHASE6_METRIC_RECONCILIATION.md",
        "results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md",
        "results/phase6_1_audit/RECONSTRUCTED_REPEAT_METRICS.csv",
        "results/phase4_uci_multicentre/PHASE4_PROTOCOL.md",
        "results/phase4_uci_multicentre/POOLED_OUT_OF_CENTRE_METRICS.csv",
        "results/phase4_uci_multicentre/CENTRE_VALIDATION_METRICS.csv",
        "results/phase4_uci_multicentre/CENTRE_COHORT_PROFILE.csv",
        "results/phase4_validation_audit/FINAL_STAGE_COMPARISON.csv",
        "paper/phase7_4_1_final_manuscript/TABLES/TABLE4_UCI_STAGES.csv",
        "paper/phase7_4_1_final_manuscript/FIGURES/FIGURE1_STUDY_WORKFLOW.png",
        "paper/phase7_4_1_final_manuscript/FIGURES/FIGURE2_PHASE6_REPEAT_AUC.png",
        "results/phase7_4_1_qa/CLEAN_ENVIRONMENT_VERIFICATION.md",
        "results/phase7_4_1_qa/TRIPOD_AI_CHECKLIST.md",
        "results/phase7_4_1_qa/PROBAST_AI_ASSESSMENT.md",
    ]
    missing = [rel for rel in source_paths if not (ROOT / rel).is_file()]
    assert not missing, f"Missing source evidence: {missing}"
    source_hashes = [{"path": rel, "sha256": sha256(ROOT / rel), "bytes": (ROOT / rel).stat().st_size} for rel in source_paths]
    style_info = None
    if STYLE_REFERENCE.is_file():
        style_pdf = PdfReader(STYLE_REFERENCE)
        style_info = {
            "path": str(STYLE_REFERENCE), "sha256": sha256(STYLE_REFERENCE),
            "pages": len(style_pdf.pages), "use": "formatting reference only; not research evidence or instruction source",
        }

    manifest = {
        "package": "CARDIODIAG Phase 8 paper-style final report",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "git_metadata": "Unavailable: repository snapshot has no .git directory.",
        "style_reference": style_info,
        "report": {
            "research_question": "Retrospective discrimination of angiography-defined CAD using recorded clinical features.",
            "pages": checks["pdf_pages"], "sections": 6, "figures": 2, "tables": 9,
            "format": "US Letter, two-column journal-paper style; centered full-width title block",
            "primary_format": "editable DOCX", "font": "Times New Roman", "main_body_size_pt": 10,
            "pdf_renderer": "Microsoft Word for Microsoft 365 ExportAsFixedFormat; Poppler used for metadata and raster QA",
        },
        "tool_versions": {"python": platform.python_version(), "python_docx": docx.__version__, "pypdf": pypdf.__version__},
        "verification": checks,
        "generated_deliverables_and_builders": generated_hashes,
        "visual_qa_pages": visual_hashes,
        "source_evidence": source_hashes,
        "manifest_note": "This manifest is not self-hashed; all generated report artifacts, builders, visual pages and source files are SHA-256 hashed.",
    }
    manifest_path = OUT / "FINAL_REPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "verification": checks, "source_count": len(source_hashes), "visual_pages": len(visual_hashes)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    report_checks, report_doc, report_pdf, pages_text, report_body = check_report()
    build_audits(report_checks, report_doc, pages_text, report_body)
