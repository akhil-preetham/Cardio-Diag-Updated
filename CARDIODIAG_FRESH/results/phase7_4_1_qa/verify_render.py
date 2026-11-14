from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
from pypdfium2 import PdfDocument

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "paper/phase7_4_1_final_manuscript/MANUSCRIPT.pdf"
OUT = ROOT / "results/phase7_4_1_qa/rendered_pages_v2"
OUT.mkdir(parents=True, exist_ok=True)
reader = PdfReader(str(PDF))
document = PdfDocument(str(PDF))
pages = []
issues = []
for index, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if not text.strip():
        issues.append((index + 1, "no extractable text"))
    if "\ufffd" in text:
        issues.append((index + 1, "replacement glyph"))
    image = document[index].render(scale=1.15).to_pil().convert("RGB")
    image.save(OUT / f"page_{index + 1:02}.png")
    pages.append(image)

tiles = []
for index, image in enumerate(pages, 1):
    image.thumbnail((260, 340))
    tile = Image.new("RGB", (280, 375), "white")
    tile.paste(image, ((280 - image.width) // 2, 20))
    ImageDraw.Draw(tile).text((12, 355), f"Page {index}", fill="black")
    tiles.append(tile)
columns = 4
rows = (len(tiles) + columns - 1) // columns
sheet = Image.new("RGB", (columns * 280, rows * 375), (220, 226, 232))
for index, tile in enumerate(tiles):
    sheet.paste(tile, ((index % columns) * 280, (index // columns) * 375))
sheet.save(OUT / "contact_sheet.png")

all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
required = ["Structured abstract", "Introduction", "Materials and methods", "Results",
            "Discussion", "Limitations", "Conclusion", "References", "Table 1.",
            "Table 6.", "Figure 1.", "Figure 4."]
missing = [value for value in required if value.lower() not in all_text.lower()]
links = sum(len(page.get("/Annots") or []) for page in reader.pages)
print(json.dumps({
    "page_count": len(reader.pages),
    "page_sizes_points": sorted({tuple(float(x) for x in page.mediabox[2:]) for page in reader.pages}),
    "rendered_page_count": len(pages),
    "text_issues": issues,
    "missing_expected_content": missing,
    "link_annotations": links,
    "pdf_bytes": PDF.stat().st_size,
    "contact_sheet": str(OUT / "contact_sheet.png"),
}, indent=2))
