"""Render the evidence-only manuscript; no predictive training or historical writes."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
import statistics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'CARDIODIAG_FRESH'
SOURCE = OUT / 'IEEE_MANUSCRIPT_DRAFT.md'
TEXT = SOURCE.read_text(encoding='utf-8')
PDF = OUT / 'CARDIODIAG_IEEE_STYLE_REVIEW_DRAFT.pdf'

def blocks(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in row):
                    rows.append(row)
                i += 1
            yield 'table', rows
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            yield f'h{level}', line.lstrip('# ')
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|')):
            para.append(lines[i].strip())
            i += 1
        yield 'p', ' '.join(para)

def escape(s):
    return html.escape(s)

body = ParagraphStyle('Body', fontName='Times-Roman', fontSize=10, leading=11.6,
                      alignment=TA_JUSTIFY, spaceAfter=5, splitLongWords=True)
head = ParagraphStyle('Section', parent=body, fontName='Times-Bold', fontSize=10,
                      alignment=TA_CENTER, spaceBefore=8, spaceAfter=5, keepWithNext=True)
sub = ParagraphStyle('Subsection', parent=body, fontName='Times-Italic', spaceBefore=5,
                     spaceAfter=4, keepWithNext=True)
small = ParagraphStyle('Small', parent=body, fontSize=8.1, leading=9.3, alignment=0)
caption = ParagraphStyle('Caption', parent=small, fontName='Times-Bold', alignment=TA_CENTER,
                         spaceBefore=5, spaceAfter=4, keepWithNext=True)
title_style = ParagraphStyle('Title', fontName='Times-Bold', fontSize=18, leading=20, alignment=TA_CENTER)
note_style = ParagraphStyle('Note', fontName='Times-Italic', fontSize=8.5, leading=10, alignment=TA_CENTER)
W, H = 612, 792
M, GAP = 43.2, 18
CW = (W - 2*M - GAP) / 2
parsed = list(blocks(TEXT))
title = parsed[0][1]

def furniture(canvas, doc):
    if doc.page == 3:
        doc.handle_nextPageTemplate('Last')
    canvas.saveState()
    canvas.setFont('Times-Italic', 8)
    canvas.drawString(M, H-24, 'CARDIODIAG | Working research draft | 14 November 2026')
    canvas.setFont('Times-Roman', 8)
    canvas.drawCentredString(W/2, 22, f'{doc.page}  |  IEEE-style review layout; venue template not validated')
    if doc.page == 1:
        p = Paragraph(escape(title), title_style)
        _, ph = p.wrap(W-2*M, 90)
        p.drawOn(canvas, M, H-43-ph)
        n = Paragraph('Author names, order and affiliations to be confirmed<br/>Existing evidence only; no submission or acceptance claimed', note_style)
        _, nh = n.wrap(W-2*M, 40)
        n.drawOn(canvas, M, H-49-ph-nh)
    canvas.restoreState()

first_top = H-155
normal_top = H-43
def frames(top):
    return [Frame(M, 38, CW, top-38, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(M+CW+GAP, 38, CW, top-38, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]

doc = BaseDocTemplate(str(PDF), pagesize=(W,H), title=title,
                      author='Authorship pending confirmation', allowSplitting=1)
doc.addPageTemplates([PageTemplate(id='First', frames=frames(first_top), onPage=furniture, autoNextPageTemplate='Normal'),
                      PageTemplate(id='Normal', frames=frames(normal_top), onPage=furniture),
                      PageTemplate(id='Last', frames=[Frame(M,normal_top-390,CW,390,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),
                                                       Frame(M+CW+GAP,normal_top-390,CW,390,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)], onPage=furniture)])
story = []
references = False
skip_intro = True
for kind, value in parsed[1:]:
    if kind == 'h2':
        skip_intro = False
        references = value == 'References'
        story.append(Paragraph(escape(value), head))
    elif skip_intro:
        continue
    elif kind == 'h3':
        story.append(Paragraph(escape(value), sub))
    elif kind == 'table':
        count = len(value[0])
        widths = [CW/count]*count
        if count == 3:
            widths = [CW*.48, CW*.26, CW*.26]
        content = [[Paragraph(escape(c), small) for c in row] for row in value]
        table = Table(content, colWidths=widths, repeatRows=1, hAlign='CENTER')
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                                   ('LINEABOVE',(0,0),(-1,0),.6,colors.black),
                                   ('LINEBELOW',(0,0),(-1,0),.4,colors.black),
                                   ('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),
                                   ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                                   ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.append(KeepTogether([table, Spacer(1,7)]))
    else:
        style = small if references else caption if re.match(r'^Table [IVX]+\.', value) else body
        story.append(Paragraph(escape(value), style))
doc.build(story)

def tex_escape(s):
    mapping = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in s)

tex = [r'\documentclass[conference]{IEEEtran}',r'\usepackage[T1]{fontenc}',r'\usepackage{array}',
       r'\title{'+tex_escape(title)+'}',
       r'\author{\IEEEauthorblockN{Authorship pending confirmation}\IEEEauthorblockA{Affiliations and correspondence pending confirmation}}',
       r'\begin{document}',r'\maketitle']
special = None
refs = False
skip_intro = True
for kind, value in parsed[1:]:
    if kind == 'h2':
        skip_intro = False
        if special:
            tex.append(r'\end{'+special+'}')
            special = None
        if value == 'Abstract':
            special='abstract'; tex.append(r'\begin{abstract}')
        elif value == 'Index Terms':
            special='IEEEkeywords'; tex.append(r'\begin{IEEEkeywords}')
        elif value == 'References':
            refs=True; tex.append(r'\begin{thebibliography}{9}')
        elif value == 'Author and Submission Notes':
            tex.append(r'\section*{Author and Submission Notes}')
        else:
            tex.append(r'\section{'+tex_escape(re.sub(r'^[IVX]+\.\s*','',value))+'}')
    elif skip_intro:
        continue
    elif kind == 'h3':
        tex.append(r'\subsection{'+tex_escape(re.sub(r'^[A-Z]\.\s*','',value))+'}')
    elif kind == 'table':
        n=len(value[0]); colspec=''.join(r'p{\dimexpr\columnwidth/'+str(n)+r'-2\tabcolsep\relax}' for _ in range(n))
        tex.extend([r'\begin{center}\footnotesize',r'\begin{tabular}{'+colspec+'}',r'\hline'])
        for j,row in enumerate(value):
            tex.append(' & '.join(tex_escape(c) for c in row)+r' \\')
            if j==0: tex.append(r'\hline')
        tex.extend([r'\hline',r'\end{tabular}',r'\end{center}'])
    elif refs:
        match=re.match(r'\[(\d+)\]\s*(.*)',value)
        if match: tex.append(r'\bibitem{r'+match[1]+'} '+tex_escape(match[2]))
    else:
        s=tex_escape(value)
        s=re.sub(r'\[(\d+)\]',lambda m:r'\cite{r'+m[1]+'}',s)
        tex.append(s+'\n')
if special: tex.append(r'\end{'+special+'}')
if refs: tex.append(r'\end{thebibliography}')
tex.append(r'\end{document}')
(OUT/'IEEE_MANUSCRIPT_DRAFT.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')

metrics_path = ROOT/'results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv'
rows = list(csv.DictReader(metrics_path.open(encoding='utf-8-sig',newline='')))
numeric=[]
for model in sorted({r['model_id'] for r in rows}):
    group=[r for r in rows if r['model_id']==model]
    for metric in ('roc_auc','pr_auc_average_precision','brier_score','log_loss','sensitivity','specificity','calibration_intercept','calibration_slope'):
        numeric.append({'procedure':model,'metric':metric,'mean':statistics.mean(float(r[metric]) for r in group),
                        'source':'CARDIODIAG_FRESH/results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv',
                        'scope':'three complete-repeat OOF summaries; descriptive'})
with (OUT/'MANUSCRIPT_NUMERIC_TRACEABILITY.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(numeric[0]));writer.writeheader();writer.writerows(numeric)

reader=PdfReader(PDF)
texts=[p.extract_text() or '' for p in reader.pages]
source_paragraphs=[v for k,v in parsed if k=='p']
cleaned=[]
for t in texts:
    t=re.sub(r'CARDIODIAG \| Working research draft \| 14 November 2026', '', t)
    t=re.sub(r'\d+\s*\|\s*IEEE-style review layout; venue template not validated', '', t)
    cleaned.append(t)
alltext=' '.join(cleaned)
normalize=lambda s: re.sub(r'\s+','',s)
missing=[v[:100] for v in source_paragraphs[2:] if normalize(v) not in normalize(alltext)]
qa={'pdf_pages':len(reader.pages),'blank_pages':[i+1 for i,t in enumerate(texts) if len(t.strip())<30],
    'replacement_glyph':any('\ufffd' in t for t in texts),'paragraphs_missing_from_extract':missing,
    'latex_compiled':False,'renderer':'ReportLab two-column IEEE-style review layout, not venue-certified IEEE typesetting',
    'new_predictive_fits':0,'figure_note':'Tables used for primary result; no synthetic performance figure',
    'visual_qa':'Pending manual PNG inspection'}
render=OUT/'rendered_pages'; render.mkdir(exist_ok=True)
pdf=pdfium.PdfDocument(str(PDF))
thumbs=[]
for i in range(len(pdf)):
    im=pdf[i].render(scale=1.4).to_pil().convert('RGB')
    im.save(render/f'page_{i+1:02d}.png')
    thumb=Image.new('RGB',(430,590),'white');thumb.paste(ImageOps.contain(im,(420,555)),(5,24))
    ImageDraw.Draw(thumb).text((8,6),f'Page {i+1}',fill='black'); thumbs.append(thumb)
cols=3; lines=(len(thumbs)+cols-1)//cols
sheet=Image.new('RGB',(cols*430,lines*590),'#cccccc')
for i,im in enumerate(thumbs):sheet.paste(im,((i%cols)*430,(i//cols)*590))
sheet.save(render/'contact_sheet.png')
(OUT/'BUILD_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
