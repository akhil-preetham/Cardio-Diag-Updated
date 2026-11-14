"""Build the editable, paper-style CARDIODIAG report from frozen project evidence."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOCX_PATH = OUT / "CARDIODIAG_FINAL_PROJECT_REPORT.docx"
FIGURE1 = ROOT / "paper/phase7_4_1_final_manuscript/FIGURES/FIGURE1_STUDY_WORKFLOW.png"
FIGURE2 = ROOT / "paper/phase7_4_1_final_manuscript/FIGURES/FIGURE2_PHASE6_REPEAT_AUC.png"


def set_font(run, size=10, *, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), "Times New Roman")


def set_columns(section, count: int, gap_inches=0.28):
    cols = section._sectPr.xpath("./w:cols")
    if not cols:
        cols_el = OxmlElement("w:cols")
        section._sectPr.append(cols_el)
    else:
        cols_el = cols[0]
    cols_el.set(qn("w:num"), str(count))
    cols_el.set(qn("w:space"), str(int(gap_inches * 1440)))


def configure_section(section, *, columns=2):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.58)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.25)
    set_columns(section, columns)


def switch_columns(doc, count: int):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    configure_section(section, columns=count)
    return section


def add_body(doc, text: str, *, size=10, after=3, before=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY, italic=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.widow_control = True
    r = p.add_run(text)
    set_font(r, size, italic=italic)
    return p


def add_run_paragraph(doc, prefix: str, body: str, *, size=10, after=3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.widow_control = True
    set_font(p.add_run(prefix), size, bold=True)
    set_font(p.add_run(body), size)
    return p


def add_main_heading(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), 10.5, bold=True)
    return p


def add_subheading(doc, marker: str, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(f"{marker} {text}"), 10, bold=True)
    return p


def add_caption(doc, text: str, *, figure=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = not figure
    set_font(p.add_run(text), 8.4, bold=False)
    return p


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "bottom", "insideH"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), "000000")
    for edge in ("left", "right", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_cell_margins(cell, top=35, start=45, bottom=35, end=45):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_table(doc, caption: str, headers: list[str], rows: list[list[str]], widths: list[float], *, size=7.8):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    set_font(cap.add_run(caption.upper()), 8.1)
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = 1
    set_table_borders(table)
    for idx, value in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.width = Inches(widths[idx])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        set_font(p.add_run(value), size, bold=True)
    set_repeat_header(table.rows[0])
    prevent_row_split(table.rows[0])
    for values in rows:
        cells = table.add_row().cells
        prevent_row_split(table.rows[-1])
        for idx, value in enumerate(values):
            cell = cells[idx]
            cell.width = Inches(widths[idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            set_font(p.add_run(str(value)), size)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.line_spacing = 0.8
    return table


def add_picture(doc, path: Path, width: float, alt: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    shape._inline.docPr.set("descr", alt)
    return shape


def add_reference(doc, number: int, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.21)
    p.paragraph_format.first_line_indent = Inches(-0.21)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    set_font(p.add_run(f"[{number}] {text}"), 8.2)


def set_default_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(3)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True


def build():
    if not FIGURE1.is_file() or not FIGURE2.is_file():
        raise FileNotFoundError("The verified Phase 7.4.1 workflow or Figure 2 is missing")

    doc = Document()
    configure_section(doc.sections[0], columns=1)
    set_default_styles(doc)
    props = doc.core_properties
    props.title = "Selection-Aware Validation of Recorded Clinical Features for Angiography-Defined CAD"
    props.subject = "Retrospective discrimination of angiography-defined CAD using recorded clinical features"
    props.author = ""
    props.comments = "Author and institutional details require confirmation."

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("Selection-Aware Validation of Recorded Clinical Features for Angiography-Defined CAD"), 19, bold=False)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run("[AUTHOR NAMES AND ORDER REQUIRE CONFIRMATION]"), 10)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run("[DEPARTMENT AND INSTITUTION REQUIRE CONFIRMATION]"), 9.5)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("Working draft; contributions, affiliations, ethics and declarations require author confirmation"), 8.5, italic=True)

    switch_columns(doc, 2)
    add_run_paragraph(doc, "Abstract—", "Background: Strong discrimination on small retrospective cardiac datasets can reflect feature timing, cohort composition and model selection. Objective: To assess whether a complete feature-group and model-selection procedure retains higher discrimination than a matched V3-style logistic comparator on the Z-Alizadeh Sani (ZAS) source rows, while reporting the separate UCI Heart Disease analyses without treating them as external validation. Methods: Six feature groups and five unique model families were selected within three inner folds across three repeats of five matched outer folds. Preprocessing remained inside training partitions; outer predictions were independently reconstructed. Results: The selection-aware repeat-level ROC-AUC values were 0.9134, 0.8999 and 0.9176, compared with 0.7370, 0.7420 and 0.7300 for V3. The mean paired difference was 0.174, with a split-variation interval of 0.137–0.211. The strict-feature UCI pooled leave-one-centre-out logistic ROC-AUC was 0.675. Interpretation: The ZAS result supports retrospective source-row discrimination only. Patient identity, predictor timing before the diagnostic decision and calibration independence from selection remain unresolved; clinical utility and prospective prediction are not established.", after=3)
    add_run_paragraph(doc, "Index Terms—", "coronary artery disease, retrospective discrimination, nested validation, model selection, calibration, reproducibility.", size=9.3, after=4)

    add_main_heading(doc, "I. INTRODUCTION")
    add_body(doc, "Research question: Retrospective discrimination of angiography-defined CAD using recorded clinical features.", after=2)
    add_body(doc, "Small public cardiovascular datasets can yield strong classification scores while leaving uncertainty about source provenance, outcome definitions, predictor timing, preprocessing and model-selection optimism. Selecting the best result after comparing many feature groups and model families can make its apparent performance optimistic; evaluation should therefore include the complete selection procedure [3,6]. Discrimination and calibration also answer different questions: a score may rank cases well while its probabilities remain inaccurate [4]. Neither property alone demonstrates clinical benefit.")
    add_body(doc, "CARDIODIAG began as a public-data machine-learning project. Earlier author-faithful experiments reproduced historical notebook behavior, including evaluation-boundary problems; later phases audited leakage, timing and dataset provenance, separated UCI tasks, and reconstructed saved predictions. Phase 5 reported ROC-AUC 0.928 for an eligible_combined Logistic/Ridge result selected from 36 benchmark rows. That result was post hoc and is not an unbiased final estimate. Phase 6 instead froze a 30-candidate pool and evaluated the whole selection procedure in outer folds.")
    add_body(doc, "The primary question is whether this selection-aware ZAS procedure discriminates angiography-defined CAD better than a matched re-evaluated V3-style comparator under the saved source-row partitions. Contributions are a selection-aware comparison, independent reconstruction of its predictions and metrics, and explicit treatment of feature-timing and calibration limits. The UCI analyses are a separate study track. The report claims neither a new classifier nor confirmed prediagnostic prediction.")
    add_body(doc, "The methodological contribution is narrower than algorithm development: it evaluates the full candidate-selection process after a historical benchmark had highlighted a post hoc best row, and reconciles the resulting saved predictions independently. This addresses a documented source of optimism in model comparisons [3,6]. The comparison can establish only what these source rows and partitions support; it cannot resolve patient linkage, the time at which a predictor was recorded, or transport to another care setting.")

    add_main_heading(doc, "II. DATA AND STUDY DESIGN")
    add_subheading(doc, "A.", "ZAS cohort and endpoint")
    add_body(doc, "The validated local ZAS file contains 303 source rows. The outcome is Cath, mapped as Cad = 1 and Normal = 0; 216 rows are labelled CAD and 87 Normal (71.3% positive). The count describes rows, not verified unique patients. No persistent patient identifier was verified, so patient-level independence is unestablished. The local CSV has 56 columns including Cath, whereas the UCI catalogue reports 54 features; this schema discrepancy has not been silently reconciled [1,2].")
    add_body(doc, "The endpoint therefore represents retrospective classification of a recorded angiography label. It is not a forecast made at a defined visit or a documented pre-test decision point. Because the outcome is common in these rows, accuracy-oriented summaries and average precision require their prevalence context; the 71.3% event proportion is a property of this selected dataset and should not be interpreted as population prevalence.")
    add_subheading(doc, "B.", "Feature timing and leakage status")
    add_body(doc, "No predictor has source-confirmed availability before the diagnostic decision. The 38-field eligible_combined group contains Age and Sex, which are likely known before the Cath outcome but lack decision-time records, and 36 history, symptom, examination, ECG and laboratory fields whose timing remains uncertain. The label eligible_combined is a historical candidate-group name, not a clinical eligibility determination. Cath, EF-TTE, Region RWMA, VHD, angiographic and vessel-count fields are excluded as outcome or downstream information.", after=2)
    add_table(doc, "TABLE I. FEATURE-ELIGIBILITY INTERPRETATION", ["Status", "Evidence and handling"], [
        ["Confirmed eligible", "None; no source timestamps establish decision-time availability."],
        ["Timing uncertain", "Age/Sex: decision time unverified; 36 other candidate fields: acquisition timing unresolved."],
        ["Exploratory only", "Phase 5 AUC 0.928 and Age/Sex sensitivity; neither is the primary estimate."],
        ["Excluded", "Cath target; EF-TTE, Region RWMA, VHD, angiographic and vessel fields."],
    ], [1.0, 2.35], size=7.4)
    add_subheading(doc, "C.", "Separate UCI analyses")
    add_body(doc, "UCI Heart Disease includes Cleveland (303 rows), Hungary (294), Switzerland (123) and VA Long Beach (200), totalling 920 rows (509 CAD labels; prevalence 55.3%) [5]. Its target is num = 0 versus num = 1–4. Phase 4 evaluated a strict-feature pooled leave-one-centre-out logistic model; Phase 4.3 evaluated four staged information sets. These analyses have distinct protocols and remain separate from ZAS; neither is described as external validation of ZAS.")

    add_main_heading(doc, "III. METHODS")
    add_subheading(doc, "A.", "Selection-aware ZAS evaluation")
    add_body(doc, "The candidate pool was frozen before outer evaluation: six groups (V3 five-predictor, basic risk, symptoms/examination, resting ECG, laboratory and eligible_combined) crossed with five unique model families (L2 logistic, elastic-net logistic, spline logistic, random forest and histogram gradient boosting), for 30 candidates. Ridge was removed as a duplicate Logistic alias. Three repeats of five matched outer folds produced 15 paired outer partitions. Within each outer training set, three inner folds selected the group, family and hyperparameters by ROC-AUC. The same outer assignments were used for the V3-style comparator, with Age, Sex, Current Smoker, EX-Smoker and FH, balanced logistic class weighting, and C chosen in outer training. This comparator is re-evaluated and is not the original frozen fitted equation.")
    add_body(doc, "The outer assignments are the first three repeats of a previously frozen 25-fold split manifest; the stored four-fold inner assignments were not reused. Phase 6 deterministically generated three stratified inner folds within each outer training partition (seed 61051 + 100 × repeat + fold). The full planned 25-outer-fold computation exceeded 35 minutes of CPU without producing outputs, so the prespecified continuation used 15 paired partitions. This reduces compute, but leaves only three repeat-level realizations for describing split variation.")
    add_table(doc, "TABLE II. FROZEN FEATURE-GROUP CANDIDATES", ["Group", "Features in candidate group"], [
        ["V3 five-predictor (5)", "Age, Sex, Current Smoker, EX-Smoker, FH"],
        ["Basic risk (7)", "Age, Sex, DM, HTN, Current Smoker, EX-Smoker, FH"],
        ["Symptoms/examination (10)", "Typical Chest Pain, Dyspnea, Function Class, Atypical, Nonanginal, Exertional CP, LowTH Ang, BP, PR, Edema"],
        ["Resting ECG (7)", "Q Wave, St Elevation, St Depression, Tinversion, LVH, Poor R Progression, BBB"],
        ["Laboratory (14)", "FBS, CR, TG, LDL, HDL, BUN, ESR, HB, K, Na, WBC, Lymph, Neut, PLT"],
        ["eligible_combined (38)", "Union of basic risk, symptoms/examination, resting ECG and laboratory variables"],
    ], [1.13, 2.2], size=6.9)
    add_table(doc, "TABLE III. UNIQUE MODEL FAMILIES AND INNER SEARCH GRIDS", ["Family", "Parameters searched"], [
        ["L2 logistic", "C ∈ {0.1, 1, 10}"],
        ["Elastic-net logistic", "C ∈ {0.1, 1, 10}; l1_ratio ∈ {0.25, 0.75}"],
        ["Spline logistic", "C ∈ {0.1, 1, 10}; 3 knots, degree 2"],
        ["Random forest", "max_depth ∈ {None, 5}; min_samples_leaf ∈ {2, 8}; 250 trees"],
        ["Histogram gradient boosting", "learning_rate ∈ {0.05, 0.1}; max_leaf_nodes ∈ {7, 15}; 120 iterations"],
    ], [1.0, 2.33], size=6.9)
    add_body(doc, "Numeric values were median-imputed with missingness indicators for non-V3 candidates, then scaled; categorical values were imputed by the most frequent training value and one-hot encoded. Spline logistic added a three-knot, degree-two numeric spline transformation. The V3 comparator used the same general train-only pipeline without missingness indicators and used balanced class weights. Each transformation was fitted only on the current training partition. Ties between candidate scores followed the declared group and family order.")
    add_body(doc, "Sigmoid calibration and Youden-J thresholds were derived from inner out-of-fold scores. Outer test labels did not enter selection, preprocessing, calibration or threshold choice; after predictions were fixed they were used for evaluation and post-fit permutation summaries. Hyperparameter selection nevertheless used labels from rows later included in calibration cross-fitting, so the calibration scores depend on selection. Inner calibration models also trained on about two-thirds of outer-training rows, while the final refit used all outer-training rows. These limitations affect probability calibration but do not create an outer-test-driven candidate-selection path [3,4].")
    add_subheading(doc, "B.", "Metrics, thresholds and uncertainty")
    add_body(doc, "ROC-AUC summarizes ranking across thresholds. Average precision (AP) is reported with the positive prevalence as its reference; AP is prevalence-sensitive and a larger value alone does not imply clinical utility. Brier score is the mean squared probability error, (1/n)Σ(pᵢ−yᵢ)²; log loss summarizes probability error on the log scale. Sensitivity and specificity use thresholds selected from training-side out-of-fold predictions using Youden J, not from outer-test outcomes. Calibration intercept and slope are estimated on outer-test predictions and are descriptive, particularly with small test folds.")
    add_body(doc, "Each repeat-level metric is calculated from its complete 303-row out-of-fold set, not from an unweighted mean of fold AUCs. Fold-level results are retained to show variation across 60–61-row partitions. The three-repeat paired t interval describes split variation only; folds and repeats reuse source rows, so this is not a patient-level or population confidence interval. We did not use an ordinary record bootstrap or treat repeated predictions as independent patients.")
    add_subheading(doc, "C.", "UCI evaluation")
    add_body(doc, "The UCI strict-feature logistic result is pooled across four held-out centres. Its six inputs are age, sex, resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs) and resting ECG (restecg); the endpoint is num = 0 versus num > 0. In each leave-one-centre-out iteration, the held-out centre is excluded from development and inner model selection. Imputation, missingness indicators, encoding, scaling and the logistic grid are fitted on the other centres only. Stage 1–4 analyses represent different information sets: age and sex; then chest-pain type; then resting pressure, cholesterol, glucose and ECG; and finally exercise-test variables (thalach, exang, oldpeak and slope). These stages are distinct saved analyses and not a single feature-addition experiment. Their AP values are interpreted against UCI prevalence. Centre-specific AUCs are shown descriptively; Switzerland has only eight negative records. The UCI results are not pooled with ZAS and do not establish transport to a new cohort.")

    # A full-width workflow figure matches the supplied paper's broad figure treatment.
    switch_columns(doc, 1)
    add_picture(doc, FIGURE1, 6.65, "Research workflow showing separate ZAS selection-aware and UCI centre-held-out evidence tracks")
    add_caption(doc, "Fig. 1. Separate ZAS selection-aware analysis and UCI multicentre evaluation.", figure=True)
    switch_columns(doc, 2)

    add_main_heading(doc, "IV. RESULTS")
    add_subheading(doc, "A.", "ZAS repeat-level performance")
    add_body(doc, "All three complete-repeat out-of-fold estimates favored the selection-aware procedure. Table II reports the paired AUCs. The mean paired difference was +0.174; the descriptive split-variation 95% t interval was +0.137 to +0.211. This interval is not a patient-level or population confidence interval.", after=2)
    add_table(doc, "TABLE IV. COMPLETE-REPEAT ZAS ROC-AUC", ["Repeat", "Selection", "V3-style", "Difference"], [
        ["1", "0.9134", "0.7370", "+0.1764"],
        ["2", "0.8999", "0.7420", "+0.1579"],
        ["3", "0.9176", "0.7300", "+0.1877"],
        ["Mean", "0.9103", "0.7363", "+0.1740"],
    ], [0.55, 0.9, 0.9, 1.0], size=7.6)
    add_body(doc, "Across repeats, selection-aware vs V3-style mean AP was 0.9573 vs 0.8469 (prevalence 0.7129); Brier score 0.1076 vs 0.1746; log loss 0.3467 vs 0.5286; sensitivity 0.8318 vs 0.6898; and specificity 0.8314 vs 0.7088. Mean outer-test calibration intercept/slope were 0.173/0.868 vs −0.007/1.005, respectively. These calibration summaries are descriptive and inherit the selection-dependent cross-fit limitation.", after=2)
    add_table(doc, "TABLE V. MEAN SECONDARY METRICS ACROSS THREE REPEATS", ["Metric", "Selection", "V3-style"], [
        ["Average precision", "0.9573", "0.8469"],
        ["Brier score", "0.1076", "0.1746"],
        ["Log loss", "0.3467", "0.5286"],
        ["Sensitivity", "0.8318", "0.6898"],
        ["Specificity", "0.8314", "0.7088"],
    ], [1.55, 0.85, 0.95], size=7.5)
    add_table(doc, "TABLE VI. REPEAT-SPECIFIC SECONDARY METRICS (SELECTION / V3-STYLE)", ["Metric", "Repeat 1", "Repeat 2", "Repeat 3"], [
        ["Average precision", "0.9602 / 0.8528", "0.9537 / 0.8519", "0.9580 / 0.8359"],
        ["Brier score", "0.1075 / 0.1741", "0.1137 / 0.1734", "0.1016 / 0.1763"],
        ["Log loss", "0.3447 / 0.5257", "0.3666 / 0.5251", "0.3289 / 0.5349"],
        ["Sensitivity", "0.8657 / 0.6713", "0.8148 / 0.6852", "0.8148 / 0.7130"],
        ["Specificity", "0.7931 / 0.7126", "0.8161 / 0.7126", "0.8851 / 0.7011"],
    ], [0.99, 0.78, 0.78, 0.78], size=6.8)
    add_body(doc, "Average precision values remained above the prevalence reference of 0.7129 for the ZAS source rows in every repeat, including for the V3-style comparator. This comparison is descriptive: prevalence is unusually high in this source file, and AP will change with event prevalence even if ranking characteristics are unchanged. The lower Brier and log-loss means for the selected procedure are encouraging probability-error summaries, but the calibration construction audit prevents treating them as evidence of a stable, transportable risk scale.", after=2)
    add_picture(doc, FIGURE2, 3.25, "Repeat-level ZAS ROC-AUC on a complete zero-to-one axis")
    add_caption(doc, "Fig. 2. Repeat-level ROC-AUC; each point uses all 303 source rows once.", figure=True)
    add_body(doc, "Across the 15 small outer test folds, ROC-AUC ranged from 0.8372 to 0.9799 for the selection procedure and 0.6375 to 0.8579 for V3. This fold spread is material and should be considered alongside the stable ordering of the three complete-repeat AUCs. The repeat-level rows aggregate all 303 out-of-fold source rows before AUC calculation.", after=2)
    add_body(doc, "The eligible_combined group was selected in 15/15 outer fits. Model-family counts were L2 logistic 9/15, elastic net 4/15 and spline logistic 2/15. Because all selected groups were identical, each of the 38 variables in that group was structurally included in all outer fits; this does not demonstrate individual feature necessity. Marginal permutation AUC decreases were largest for Typical Chest Pain (mean 0.1253, SD 0.0500), followed by Age (0.0416, SD 0.0166) and DM (0.0189, SD 0.0110). These post-fit associations are not causal, can be unstable with correlated predictors and may reflect ascertainment.", after=2)
    add_table(doc, "TABLE VII. SELECTION FREQUENCY AND LEADING PERMUTATION SUMMARIES", ["Quantity", "Result across 15 outer fits"], [
        ["Selected feature group", "eligible_combined: 15/15"],
        ["Selected model family", "L2 logistic: 9; elastic net: 4; spline logistic: 2"],
        ["Largest marginal AUC decrease", "Typical Chest Pain: 0.1253 (SD 0.0500)"],
        ["Next marginal AUC decrease", "Age: 0.0416 (SD 0.0166)"],
        ["Third marginal AUC decrease", "DM: 0.0189 (SD 0.0110)"],
    ], [1.22, 2.13], size=7.0)
    add_body(doc, "The separate Age/Sex-only sensitivity model had mean repeat ROC-AUC 0.729 (range 0.726–0.731), below the selected full candidate procedure; Age and Sex nevertheless remain timing-uncertain. Phase 5's 0.928 result remains exploratory and post hoc, and did not define the Phase 6 candidate pool.", after=2)
    add_body(doc, "A separate Phase 6.1 audit reconstructed all 1,818 saved predictions (303 source rows × three repeats × two procedures), verified labels against Cath and assignments against the frozen outer manifest, and checked selection metadata for every partition. Fold and repeat metrics reconciled to the saved outputs within 1e−10; no prediction or metric discrepancy was found. This was an artifact-level verification and did not refit models. A clean Python 3.11.16 environment installed the pinned 126-package snapshot and passed the package-integrity check and lightweight read-only smoke checks. These checks support computational reproducibility of the saved analysis, not external scientific validity.", after=2)

    add_subheading(doc, "B.", "Separate UCI results")
    add_body(doc, "The strict-feature pooled leave-one-centre-out logistic analysis included 920 records and yielded ROC-AUC 0.6753, AP 0.6809, sensitivity 0.7328, specificity 0.5547, Brier score 0.2254 and log loss 0.6456. AP should be read against positive prevalence 0.553. Strict-feature centre-specific logistic AUCs were Cleveland 0.6771, Hungary 0.6299, Switzerland 0.4576 and VA Long Beach 0.5736; Switzerland had 115 positive and only eight negative rows.")
    add_body(doc, "The later staged centre-held-out analyses gave the following ROC-AUC/AP/Brier/log-loss values: Stage 1 basic risk, 0.7179/0.7293/0.2112/0.6117; Stage 2 symptoms, 0.8225/0.8241/0.1681/0.5106; Stage 3 resting measures, 0.8215/0.8191/0.1684/0.5139; and Stage 4 exercise test, 0.8450/0.8645/0.1584/0.4861. Stage 2 to 3 changed AUC by about −0.001; Stage 3 to 4 by about +0.024. These information stages are not a single monotonic test of one feature addition.")
    add_table(doc, "TABLE VIII. UCI STAGED CENTRE-HELD-OUT PERFORMANCE", ["Information stage", "ROC-AUC", "AP", "Brier", "Log loss"], [
        ["1. Basic risk", "0.7179", "0.7293", "0.2112", "0.6117"],
        ["2. Symptoms", "0.8225", "0.8241", "0.1681", "0.5106"],
        ["3. Resting measures", "0.8215", "0.8191", "0.1684", "0.5139"],
        ["4. Exercise test", "0.8450", "0.8645", "0.1584", "0.4861"],
    ], [1.18, 0.55, 0.45, 0.52, 0.64], size=6.9)
    add_table(doc, "TABLE IX. CENTRE-SPECIFIC STRICT-FEATURE LOGISTIC AUC", ["Centre", "Rows", "CAD / Normal", "ROC-AUC"], [
        ["Cleveland", "303", "139 / 164", "0.6771"],
        ["Hungary", "294", "106 / 188", "0.6299"],
        ["Switzerland", "123", "115 / 8", "0.4576"],
        ["VA Long Beach", "200", "149 / 51", "0.5736"],
    ], [0.95, 0.52, 1.05, 0.8], size=7.2)

    add_main_heading(doc, "V. DISCUSSION")
    add_body(doc, "The primary finding is a consistent source-row discrimination advantage for the complete Phase 6 selection procedure over its matched V3-style comparator across three repeats. Independent reconstruction verified the saved labels, outer partitions and repeat-level metrics. This supports a retrospective comparison within the available ZAS rows. It does not establish performance among independent patients because patient identifiers are absent, and it does not demonstrate performance at a clinical decision time because feature acquisition times are undocumented.")
    add_body(doc, "The evidence also places limits on interpretation. The selected group includes symptoms, ECG and laboratory variables whose timing remains uncertain, and Typical Chest Pain had the largest marginal permutation association. It could reflect referral or diagnostic ascertainment. The selection procedure remains a model-selection benchmark, not a fixed, clinically validated score. The calibration map is selection-dependent, is learned from smaller inner-fold models, and is applied to larger outer-training refits. Discrimination remains separately supported, but probability-based decisions require stronger calibration evidence.")
    add_body(doc, "The UCI strict and staged results are distinct retrospective tasks on one historical multicentre dataset. Their centre-level AUCs vary and do not validate ZAS. Neither track establishes clinical utility, prospective prediction, decision-curve benefit or transport to contemporary hospitals. The apparent Phase 5 AUC of 0.928 must not replace the Phase 6 estimate because it was selected after comparing 36 rows. This work's contribution is an evidence-constrained re-evaluation and audit, not a claim of a novel algorithm or external validation.")
    add_subheading(doc, "B.", "Interpretation of the paired comparison")
    add_body(doc, "The paired Phase 6 contrast is against a V3-style comparator evaluated anew on the same outer partitions. It uses the five V3 variables, balanced logistic regression, inner-selected regularization, and the same calibration and threshold operation. This matching avoids comparing the new procedure with a score calculated under a different split, but it also means the comparator is not the exact historical V3 fitted equation or its original operating threshold. The result supports a difference between two training procedures under this protocol; it should not be phrased as a universal improvement over every V3 implementation.")
    add_body(doc, "The selected feature group remained constant while the selected model family varied among three logistic forms. This pattern suggests that, within the frozen search universe and these partitions, the broader group was more consistently preferred than one particular model family. It does not identify which of the 38 variables is essential. Correlated predictors can substitute for one another in permutation summaries, and an association for Typical Chest Pain may capture case ascertainment or referral pathways as readily as a pre-existing disease signal. Neither group selection nor permutation importance is causal evidence.")
    add_subheading(doc, "C.", "Uncertainty, calibration and transport")
    add_body(doc, "The mean paired AUC difference of 0.174 and its 0.137–0.211 interval summarize only the three repeat-specific differences. Each repeat reuses the same 303 source rows under a different partition, so the three differences are not independent study replications. The 15 outer folds are also not 15 independent centres or patient cohorts. No patient-level bootstrap, centre-level interval or population-level precision claim is justified by these data. The observed fold ranges show that a single fold can depart materially from the repeat summary.")
    add_body(doc, "Outer-test calibration intercept and slope were computed from outer-test predictions, but the calibrator itself was trained on scores from inner rows whose labels had already influenced candidate selection. Those inner scoring models also used about 161–162 training rows, compared with 242 rows in the final outer refit. In effect, the outer fold evaluates the practical calibration procedure, including its mismatch; it does not certify a calibrator independent of selection or a fixed model for patient-level risk communication. Sensitivity and specificity at training-selected Youden thresholds likewise describe this dataset's operating points and do not express the consequences of false positives and false negatives in a clinical pathway.")
    add_subheading(doc, "D.", "Meaning of the separate UCI results")
    add_body(doc, "The four UCI source centres differ in both size and class mix. Switzerland contributes 115 positive and eight negative rows, making its strict logistic AUC particularly sensitive to the ordering of a small negative group. The centre-specific values therefore add useful heterogeneity context, but they do not establish a reproducible centre effect. Stage 1–4 results also change the information set, and the stage differences cannot be attributed to one isolated predictor because the staged protocols are not a single feature-addition experiment. UCI supplies a separate retrospective multicentre analysis; it was not used as a ZAS test set and does not validate a ZAS model.")
    add_body(doc, "The reporting value of this project is consequently procedural and evidential: it makes a post hoc benchmark visible, states the selection universe in advance, preserves a paired reference, reconstructs stored predictions, and separates ranking, probability error and feature-timing questions. The analyses do not demonstrate that these particular variables are available before a clinical decision, nor that using them would change management or outcomes. A high ROC-AUC on these records must not be equated with a useful clinical tool.")
    add_body(doc, "Reporting completeness and risk-of-bias appraisal are separate tasks. The project package includes a TRIPOD+AI reporting checklist and a PROBAST+AI applicability assessment as companion materials [7,8]; populated checklists do not by themselves establish compliance, low risk of bias or clinical applicability. Remaining author declarations and source-documentation gaps are retained for confirmation.")
    add_subheading(doc, "E.", "Limitations and next study")
    add_body(doc, "The ZAS sample contains only 303 source rows and has three complete repeats. No verified patient identifier is available. The split unit is therefore a source row; if the file contains multiple rows from one person, related records could cross folds, although this audit did not establish whether such duplication occurred. The resulting estimates cannot be interpreted as performance in new patients. Nor are acquisition timestamps available to show that Age, Sex, symptoms, ECG findings or laboratory values preceded the angiography decision. Cath is the recorded target, so the task remains retrospective discrimination of that label rather than a prediction made at a defined clinical index time.")
    add_body(doc, "The local ZAS file has 56 columns including the target while the UCI catalogue describes 54 features. Until the exact source version and variable dictionary are reconciled, the report's results should be tied to the local validated file rather than generalized to every release of the dataset. The 30-candidate Phase 6 pool was frozen for this evaluation, but the candidate groups originated in an earlier benchmark. Nested selection addresses choosing among that stated pool; it cannot eliminate all design choices made before the pool was frozen or establish that its timing-uncertain variables are clinically appropriate. The Phase 5 0.928 result was not used to alter the pool.")
    add_body(doc, "The three repeat-specific paired differences produce a descriptive split-variation interval, not a conventional confidence interval for independent observations. The repeated partitions share the same source rows, and the calibration cross-fitting is dependent on candidate selection. The separate UCI tasks also have limits: only four historical source centres are represented, the centre mix is heterogeneous, Switzerland has eight negative records, and the four stages encode different information sets. Neither the ZAS nor UCI results estimate transport to a contemporary hospital network or provide a prespecified decision threshold with established clinical consequences.")
    add_body(doc, "A defensible prediagnostic study would begin with a verified source dictionary, patient linkage and data-use permission. Investigators would define the prediction index time and outcome window, then retain only features demonstrably available by that time. The split unit should be the patient, with centre-held-out evaluation where enough centres exist; the complete model and feature-selection procedure should be frozen before a genuinely independent, endpoint-compatible cohort is evaluated. Calibration should be constructed with selection-aware cross-fitting and assessed on held-out predictions. Clinical utility would require a separate evaluation of decision consequences and net benefit, followed by prospective or impact evaluation appropriate to the intended use. Until these steps are completed, the evidence supports a retrospective discrimination report only.")

    add_main_heading(doc, "VI. CONCLUSION")
    add_body(doc, "Selection-aware validation and independent reconstruction support higher repeat-level ROC-AUC for the Phase 6 procedure than for a matched re-evaluated V3-style comparator on ZAS source-row partitions. Patient independence, predecision feature timing and calibration remain unresolved or limited. The evidence supports a cautious retrospective discrimination paper using recorded clinical features. It supports neither a prediagnostic prediction paper nor clinical use. UCI remains a separate evaluation track.")
    add_main_heading(doc, "DECLARATIONS")
    add_body(doc, "Author names/order/contributions, affiliations, guide and institution, certificate and student declaration, acknowledgements, ethics determination, data-use permissions, funding, conflicts of interest, and protocol/registration status require human confirmation. No approval, funding or declaration is asserted here.", size=9.2)
    add_main_heading(doc, "REFERENCES")
    add_reference(doc, 1, "Alizadehsani R, Habibi J, Hosseini MJ, et al. A data mining approach for diagnosis of coronary artery disease. Computer Methods and Programs in Biomedicine. 2013;111(1):52–61. doi:10.1016/j.cmpb.2013.03.004.")
    add_reference(doc, 2, "Alizadehsani R, Roshanzamir M, Sani Z. Z-Alizadeh Sani [dataset]. UCI Machine Learning Repository. doi:10.24432/C5Q31T.")
    add_reference(doc, 3, "Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research. 2010;11:2079–2107.")
    add_reference(doc, 4, "Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Medicine. 2019;17:230. doi:10.1186/s12916-019-1466-7.")
    add_reference(doc, 5, "Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease [dataset]. UCI Machine Learning Repository. doi:10.24432/C52P4X.")
    add_reference(doc, 6, "Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. doi:10.1186/1471-2105-7-91.")
    add_reference(doc, 7, "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.")
    add_reference(doc, 8, "Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505. doi:10.1136/bmj-2024-082505.")

    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX_PATH)
    print(f"Wrote {DOCX_PATH}")


if __name__ == "__main__":
    build()
