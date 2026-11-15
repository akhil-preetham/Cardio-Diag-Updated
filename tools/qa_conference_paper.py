"""Review-ready PDF/Word QA; run in the bundled document runtime."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import pdfplumber
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageOps
from portable_release import discover_workspace, PROJECT, sha256
from verify_conference_paper import verify_structure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-dir',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path);args=parser.parse_args()
    root=discover_workspace();out=root/PROJECT/'paper/reference_format'
    build=json.loads((out/'BUILD_PROVENANCE.json').read_text())
    if sha256(args.reference)!=build['reference_sha256']:raise ValueError('Retained reference changed')
    source=args.render_dir/'CARDIODIAG_CONFERENCE_PAPER.pdf'
    pdf=out/source.name;shutil.copyfile(source,pdf)
    reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages]
    markers=['I. Introduction','II. Literature Survey','III. Methodology','IV. Algorithm',
             'V. Results and Discussion','VI. Conclusion','References','Golla Vennela',
             '0.9174','0.9103','0.1507','0.4904']
    joined=' '.join(' '.join(texts).split())
    missing=[x for x in markers if x not in joined]
    with pdfplumber.open(pdf) as doc:
        fonts=Counter((c['fontname'],round(c['size'],2)) for page in doc.pages for c in page.chars if c['text'].strip())
        main_fonts=[(f,s,n) for (f,s),n in fonts.items() if 'TimesNewRoman' in f]
        outside=[i+1 for i,page in enumerate(doc.pages) if any(c['x0']<0 or c['x1']>page.width+1 for c in page.chars)]
    # Word's PDF exporter quantizes 10 pt to 9.96 pt; DOCX half-point sizes
    # are checked exactly by verify_structure().
    assert main_fonts and all(abs(size-10)<.05 for _,size,_ in main_fonts),'Main text must be 10-point Times New Roman'
    structure=verify_structure()
    check={'pdf_pages':len(texts),'requested_page_range_met':6<=len(texts)<=8,
        'blank_pages':[i+1 for i,t in enumerate(texts) if len(t)<80],
        'missing_markers':missing,'replacement_glyphs':sum(t.count('\ufffd') for t in texts),
        'text_outside_page':outside,'all_main_text_10pt':True,'main_font':'Times New Roman',
        'font_runs':[{'font':f,'points':s,'characters':n} for f,s,n in main_fonts],
        'math_representation':'Three native editable Word equations; mathematical glyphs use Word math fonts',
        'render_backend':'Packaged render_docx.py with native Microsoft Word read-only PDF export and PDFium rasterization',
        'reference_unchanged':True,'docx_sha256':sha256(out/'CARDIODIAG_CONFERENCE_PAPER.docx'),
        'pdf_sha256':sha256(pdf),'structure':structure,'visual_review':'pending'}
    files=sorted(args.render_dir.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    assert len(files)==len(texts)
    sheet=Image.new('RGB',(1350,((len(files)+2)//3)*600),'#cccccc')
    for i,path in enumerate(files):
        with Image.open(path) as image:sheet.paste(ImageOps.contain(image,(440,570)),((i%3)*450+5,(i//3)*600+25))
        ImageDraw.Draw(sheet).text(((i%3)*450+8,(i//3)*600+5),f'Page {i+1}',fill='black')
    sheet.save(args.render_dir/'contact.png')
    (out/'CONFERENCE_QA.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(check,indent=2))
    assert check['requested_page_range_met'] and not check['blank_pages'] and not missing and not outside and not check['replacement_glyphs']


if __name__=='__main__':main()
