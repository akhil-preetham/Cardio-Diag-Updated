"""Local Markdown-to-PDF renderer for the complete Phase 7.4.1 manuscript copy."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, LongTable, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/phase7_4_1_final_manuscript/MANUSCRIPT.md"
OUT = ROOT / "paper/phase7_4_1_final_manuscript/MANUSCRIPT.pdf"
FIGURES = MANUSCRIPT.parent / "FIGURES"

font_path = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(font_path / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_path / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(font_path / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(font_path / "arialbi.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Arial-Bold", fontSize=18, leading=22, alignment=TA_LEFT, spaceAfter=12, textColor=colors.HexColor("#12263A")))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=14, leading=17, spaceBefore=12, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#17324D")))
styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontName="Arial-Bold", fontSize=11.5, leading=14, spaceBefore=9, spaceAfter=4, keepWithNext=True, textColor=colors.HexColor("#17324D")))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Arial", fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY, spaceAfter=5, splitLongWords=True))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Arial", fontSize=7.4, leading=9, spaceAfter=2, splitLongWords=True))
styles.add(ParagraphStyle(name="Cellx", parent=styles["BodyText"], fontName="Arial", fontSize=7.0, leading=8.2, spaceAfter=0, splitLongWords=True))
styles.add(ParagraphStyle(name="HeaderCellx", parent=styles["BodyText"], fontName="Arial-Bold", fontSize=7.0, leading=8.2, textColor=colors.white, spaceAfter=0, splitLongWords=True))
styles.add(ParagraphStyle(name="Bulletx", parent=styles["BodyText"], fontName="Arial", fontSize=9.1, leading=12, leftIndent=14, firstLineIndent=-9, spaceAfter=3))


def inline(text: str) -> str:
    links: list[str] = []

    def hold(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if not target.startswith(("https://", "http://")):
            # Keep the sidecar path visible without emitting fragile PDF file URIs.
            return html.escape(f"{label} ({target})")
        idx = len(links)
        links.append(f'<link href="{html.escape(target, quote=True)}" color="#145D8C">{html.escape(label)}</link>')
        return f"@@LINK{idx}@@"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", hold, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for i, link in enumerate(links):
        text = text.replace(f"@@LINK{i}@@", link)
    return text


def split_row(line: str) -> list[str]:
    return [cell.strip().replace(r"\|", "|") for cell in line.strip().strip("|").split("|")]


def page_decor(canvas, doc):
    canvas.saveState()
    w, h = doc.pagesize
    canvas.setStrokeColor(colors.HexColor("#D8E0E8"))
    canvas.line(doc.leftMargin, h - 0.42 * inch, w - doc.rightMargin, h - 0.42 * inch)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(colors.HexColor("#52677B"))
    canvas.drawString(doc.leftMargin, h - 0.32 * inch, "CARDIODIAG — Phase 7.4.1 manuscript rendering")
    canvas.drawRightString(w - doc.rightMargin, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story() -> list:
    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    code_lines: list[str] | None = None
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if code_lines is None:
                code_lines = []
            else:
                story.append(Paragraph("<br/>".join(html.escape(x) for x in code_lines), styles["Code"]))
                code_lines = None
            i += 1
            continue
        if code_lines is not None:
            code_lines.append(line)
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            raw = [split_row(line)]
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw.append(split_row(lines[i]))
                i += 1
            ncol = max(map(len, raw))
            raw = [r + [""] * (ncol - len(r)) for r in raw]
            width = 9.8 * inch if ncol >= 10 else 7.0 * inch
            if ncol >= 10:
                story.extend([NextPageTemplate("Landscape"), PageBreak()])
            cellstyle = styles["HeaderCellx"]
            data = [[Paragraph(inline(cell), cellstyle if ridx == 0 else styles["Cellx"]) for cell in row] for ridx, row in enumerate(raw)]
            tbl = LongTable(data, colWidths=[width / ncol] * ncol, repeatRows=1, hAlign="LEFT", splitByRow=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244B68")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEBBC7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([tbl, Spacer(1, 8)])
            if ncol >= 10:
                story.extend([NextPageTemplate("Portrait"), PageBreak()])
            continue
        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            image_path = (MANUSCRIPT.parent / image_match.group(2)).resolve()
            if not image_path.is_file():
                story.append(Paragraph(f"[Missing figure: {html.escape(str(image_path))}]", styles["Bodyx"]))
            else:
                img = Image(str(image_path))
                maxw, maxh = 6.5 * inch, 4.5 * inch
                scale = min(maxw / img.imageWidth, maxh / img.imageHeight, 1.0)
                img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
                img.hAlign = "CENTER"
                story.append(img)
                story.append(Spacer(1, 8))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["PaperTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H2x"]))
        elif line.startswith("### "):
            if line.strip().lower() == "### figure captions":
                story.append(PageBreak())
            story.append(Paragraph(inline(line[4:]), styles["H3x"]))
        elif line.startswith("---"):
            story.append(Spacer(1, 6))
        elif line.startswith("- "):
            story.append(Paragraph("• " + inline(line[2:]), styles["Bulletx"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), styles["Bulletx"]))
        else:
            paragraph = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3} |\|.*\||- |\d+\. |!\[|---|```)", lines[i]):
                paragraph.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), styles["Bodyx"]))
            continue
        i += 1
    if code_lines is not None:
        story.append(Paragraph("<br/>".join(html.escape(x) for x in code_lines), styles["Code"]))
    return story


def main() -> None:
    portrait = letter
    wide = landscape(letter)
    doc = BaseDocTemplate(str(OUT), pagesize=portrait, leftMargin=0.72*inch, rightMargin=0.72*inch,
                          topMargin=0.62*inch, bottomMargin=0.62*inch, title="CARDIODIAG Phase 7.4.1 Manuscript")
    portrait_frame = Frame(doc.leftMargin, doc.bottomMargin, portrait[0]-doc.leftMargin-doc.rightMargin,
                           portrait[1]-doc.topMargin-doc.bottomMargin, id="portrait", leftPadding=0,
                           rightPadding=0, topPadding=0, bottomPadding=0)
    wide_frame = Frame(0.48*inch, 0.58*inch, wide[0]-0.96*inch, wide[1]-1.16*inch,
                       id="landscape", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="Portrait", frames=[portrait_frame], onPage=page_decor, pagesize=portrait),
        PageTemplate(id="Landscape", frames=[wide_frame], onPage=page_decor, pagesize=wide),
    ])
    doc.build(build_story())
    print(f"Rendered {OUT}")


if __name__ == "__main__":
    main()
