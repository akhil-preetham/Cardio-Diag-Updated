"""Document-runtime QA for the rebuilt supplementary PDFs."""
import json
from collections import Counter
from pathlib import Path
import pdfplumber
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps
from portable_release import discover_workspace, PROJECT, sha256

root=discover_workspace();out=root/PROJECT/'paper/final';render=root.parent/'submission_documents_qa'
render.mkdir(exist_ok=True)
checks={}
for name,required in [('SUPPLEMENTARY_METHODS',['S1.','S8.','NUMERIC_TRACEABILITY.csv','0.490421']),
                      ('SUBMISSION_AUDIT',['Decision and integration record','Author-dependent','independent investigator'])]:
    pdf=out/(name+'.pdf');reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages]
    with pdfplumber.open(pdf) as doc:
        fonts=Counter((c['fontname'],round(c['size'],2)) for page in doc.pages for c in page.chars if c['text'].strip())
        outside=[i+1 for i,page in enumerate(doc.pages) if any(c['x0']<0 or c['x1']>page.width+1 for c in page.chars)]
    assert any('TimesNewRoman' in f and abs(size-9.96264)<.03 for f,size in fonts)
    record={'pages':len(texts),'blank_pages':[i+1 for i,t in enumerate(texts) if len(t)<80],
            'missing_markers':[x for x in required if x not in '\n'.join(texts)],
            'replacement_glyphs':sum(t.count('\ufffd') for t in texts),
            'text_outside_page':outside,'sha256':sha256(pdf),'font':'Times New Roman, 10 TeX points',
            'visual_review':'pending'}
    assert not record['blank_pages'] and not record['missing_markers']
    assert not outside and not record['replacement_glyphs']
    doc=pdfium.PdfDocument(str(pdf));sheet=Image.new('RGB',(1350,((len(doc)+2)//3)*600),'#cccccc')
    for i in range(len(doc)):
        image=doc[i].render(scale=1.6).to_pil().convert('RGB');image.save(render/f'{name}_{i+1:02}.png')
        sheet.paste(ImageOps.contain(image,(440,570)),((i%3)*450+5,(i//3)*600+25))
        ImageDraw.Draw(sheet).text(((i%3)*450+8,(i//3)*600+5),f'Page {i+1}',fill='black')
    sheet.save(render/(name+'_CONTACT.png'));checks[name]=record
(out/'SUBMISSION_DOCUMENTS_QA.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
