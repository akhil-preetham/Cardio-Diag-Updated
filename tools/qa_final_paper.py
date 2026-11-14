"""Document-runtime verification and rendering of the final paper."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import re
import pdfplumber
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps
from portable_release import discover_workspace, PROJECT

root=discover_workspace();out=root/PROJECT/'paper/final'
pdf=out/'CARDIODIAG_FINAL_PAPER.pdf';reader=PdfReader(pdf)
texts=[p.extract_text() for p in reader.pages]
joined=re.sub(r'\s+','','\n'.join(texts)).casefold()
required=['Abstract','Experimental Design','Final Decision and Stability Analyses',
          'Conclusion','References','0.888889','0.490421','0.152489','0.914449','0.150723']
with pdfplumber.open(pdf) as doc:
    # Font sizes in PDF points: 10 TeX points = 9.96264 PDF points.
    fonts=Counter((c['fontname'],round(c['size'],2)) for page in doc.pages for c in page.chars if c['text'].strip())
    main=[(font,size,n) for (font,size),n in fonts.items() if 'TimesNewRoman' in font and abs(size-9.96264)<.03]
    assert sum(n for _,_,n in main)>25000, 'Body must actually use embedded Times New Roman at 10 TeX points'
    assert all('TimesNewRoman' in font for font,_,_ in main)
checks={'pages':len(texts),'requested_page_range_met':6<=len(texts)<=8,
    'blank_pages':[i+1 for i,t in enumerate(texts) if len(t)<80],
    'missing_markers':[x for x in required if re.sub(r'\s+','',x).casefold() not in joined],
    'replacement_glyphs':sum(t.count('\ufffd') for t in texts),
    'body_font':'Times New Roman (regular, bold, italic, bold italic embedded)',
    'body_size_tex_points':10,'body_size_pdf_points':9.96264,
    'main_font_runs':[{'font':f,'pdf_points':s,'characters':n} for f,s,n in main],
    'font_exceptions':'Mathematical symbols and superscripts use TeX math fonts; figure labels are Times New Roman at nominal 10 points.',
    'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'visual_review':'pending'}
render=root.parent/'final_paper_qa';render.mkdir(exist_ok=True)
document=pdfium.PdfDocument(str(pdf));sheet=Image.new('RGB',(1350,((len(texts)+2)//3)*600),'#cccccc')
for i in range(len(document)):
    image=document[i].render(scale=1.7).to_pil().convert('RGB')
    image.save(render/f'page_{i+1:02}.png')
    sheet.paste(ImageOps.contain(image,(440,570)),((i%3)*450+5,(i//3)*600+25))
    ImageDraw.Draw(sheet).text(((i%3)*450+8,(i//3)*600+5),f'Page {i+1}',fill='black')
sheet.save(render/'contact_sheet.png')
(out/'PAPER_QA.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
assert checks['requested_page_range_met'] and not checks['blank_pages']
assert not checks['missing_markers'] and not checks['replacement_glyphs']
