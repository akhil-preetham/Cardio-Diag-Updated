"""Run with the document runtime (pypdf, pypdfium2 and Pillow)."""
from pathlib import Path
import hashlib
import json
import re
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageOps
from portable_release import discover_workspace, PROJECT

root = discover_workspace()
out = root / PROJECT / 'paper/research_extension_v2'
pdf = out / 'CARDIODIAG_RESEARCH_REPORT_V2.pdf'
render = root.parent / 'v2_report_qa'
render.mkdir(exist_ok=True)
reader = PdfReader(pdf)
texts = [p.extract_text() for p in reader.pages]
joined = re.sub(r'\s+', '', '\n'.join(texts)).casefold()
required = ['Abstract', 'Experimental Design', 'Results', 'Limitations',
            'Additive Reliability Follow-Up', 'References', '0.914449', '0.150723']
checks = {'pages': len(texts), 'requested_page_range_met': 6 <= len(texts) <= 8,
          'blank_pages': [i+1 for i,t in enumerate(texts) if len(t) < 80],
          'replacement_glyphs': sum(t.count('\ufffd') for t in texts),
          'missing_markers': [x for x in required if re.sub(r'\s+', '', x).casefold() not in joined],
          'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
          'compiler': 'Tectonic 0.17.0, IEEEtran conference', 'visual_review': 'pending'}
doc = pdfium.PdfDocument(str(pdf))
sheet = Image.new('RGB', (1350, ((len(texts)+2)//3)*600), '#cccccc')
for i in range(len(doc)):
    image = doc[i].render(scale=1.5).to_pil().convert('RGB')
    image.save(render / f'page_{i+1:02}.png')
    sheet.paste(ImageOps.contain(image, (440,570)), ((i%3)*450+5, (i//3)*600+25))
    ImageDraw.Draw(sheet).text(((i%3)*450+8,(i//3)*600+5), f'Page {i+1}', fill='black')
sheet.save(render / 'contact_sheet.png')
(out / 'REPORT_QA.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
assert checks['requested_page_range_met'] and not checks['blank_pages']
assert not checks['replacement_glyphs'] and not checks['missing_markers']
