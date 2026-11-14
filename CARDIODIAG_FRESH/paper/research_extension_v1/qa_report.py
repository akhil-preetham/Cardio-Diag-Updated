"""Render final compiled report for page-by-page QA; no predictive fitting."""
from pathlib import Path
import hashlib
import json
import re

from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps

OUT = Path(__file__).resolve().parent
PDF = OUT / "CARDIODIAG_RESEARCH_REPORT.pdf"
RENDER = OUT / "rendered_pages"
RENDER.mkdir(exist_ok=True)
reader = PdfReader(PDF)
texts = [p.extract_text() or "" for p in reader.pages]
text = "\n".join(texts)
normalized_text = re.sub(r"\s+", "", text).casefold()
required = ["Abstract", "Introduction", "Related Work", "Experimental Design", "Results", "Discussion", "Limitations", "Conclusion", "References"]
checks = {"pdf_pages":len(reader.pages), "requested_page_range_met":6 <= len(reader.pages) <= 8,
          "blank_pages":[i+1 for i,t in enumerate(texts) if len(t.strip()) < 40],
          "missing_section_markers":[v for v in required if re.sub(r"\s+", "", v).casefold() not in normalized_text],
          "replacement_glyph_count":text.count("\ufffd"),
          "unresolved_placeholders":re.findall(r"@[A-Z_]+@",text),
          "latex_compiled":True,"compiler":"Tectonic 0.17.0; IEEEtran conference class",
          "pdf_sha256":hashlib.sha256(PDF.read_bytes()).hexdigest(),"visual_review":"Pending rendered-page inspection"}
doc = pdfium.PdfDocument(str(PDF))
thumbnails=[]
for i in range(len(doc)):
    page=doc[i]
    image=page.render(scale=1.65).to_pil().convert("RGB")
    image.save(RENDER/f"page_{i+1:02d}.png")
    thumb=Image.new("RGB",(430,590),"white")
    thumb.paste(ImageOps.contain(image,(420,550)),(5,28))
    ImageDraw.Draw(thumb).text((8,7),f"Page {i+1}",fill="black")
    thumbnails.append(thumb)
columns=3
sheet=Image.new("RGB",(columns*430,((len(thumbnails)+columns-1)//columns)*590),"#cccccc")
for i,im in enumerate(thumbnails):sheet.paste(im,((i%columns)*430,(i//columns)*590))
sheet.save(RENDER/"contact_sheet.png")
(OUT/"REPORT_RENDER_QA.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
(OUT/"PDF_EXTRACTED_TEXT.txt").write_text(text,encoding="utf-8")
print(json.dumps(checks,indent=2))
if checks["blank_pages"] or checks["missing_section_markers"] or checks["unresolved_placeholders"] or checks["replacement_glyph_count"]:
    raise SystemExit(1)
