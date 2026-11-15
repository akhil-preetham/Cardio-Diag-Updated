"""Portable checks for the reference-format Word/PDF reporting artifacts."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
from portable_release import discover_workspace, PROJECT, sha256

ROOT=discover_workspace();OUT=ROOT/PROJECT/'paper/reference_format'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}'


def text(element):return ''.join(n.text or '' for n in element.iter(W+'t'))


def verify_structure():
    build=json.loads((OUT/'BUILD_PROVENANCE.json').read_text())
    profile=json.loads((OUT/'REFERENCE_LAYOUT.json').read_text())
    authors=json.loads((ROOT/'AUTHORS.json').read_text())['authors']
    path=OUT/'CARDIODIAG_CONFERENCE_PAPER.docx'
    with zipfile.ZipFile(path) as z:
        xml=ET.fromstring(z.read('word/document.xml'));body=xml.find('w:body',NS)
        content=text(body)
        for value in [a[k] for a in authors for k in ['name','email']]:
            if value not in content:raise ValueError('Missing current author contact: '+value)
        for term in ['Swin','NR-IQA','Uddagiri','Pavithra','akkinasrisiri','sirisha.uddagiri']:
            if term.casefold() in content.casefold():raise ValueError('Reference research content leaked: '+term)
        if '\ufffd' in content:raise ValueError('Replacement character in Word content')
        for run in xml.findall('.//w:r',NS):
            if not text(run).strip():continue
            size=run.find('w:rPr/w:sz',NS);font=run.find('w:rPr/w:rFonts',NS)
            if size is None or size.get(W+'val')!='20' or font is None or font.get(W+'ascii')!='Times New Roman':
                raise ValueError('Every text run must use Times New Roman 10 point')
        sections=xml.findall('.//w:sectPr',NS)
        if len(sections)!=6:raise ValueError('Six source section patterns must be retained')
        for sec,expected in zip(sections,profile['sections']):
            size=sec.find('w:pgSz',NS);mar=sec.find('w:pgMar',NS);cols=sec.find('w:cols',NS)
            assert int(size.get(W+'w'))==expected['page_width_twips'] and int(size.get(W+'h'))==expected['page_height_twips']
            assert [int(mar.get(W+x)) for x in ['top','right','bottom','left']]==expected['margins_twips']
            assert dict(cols.attrib)==expected['columns']
        for name,digest in build['preserved_parts'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Opaque template part changed: '+name)
        actual_images={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
        own_images={sha256(OUT/name) for name in ['workflow.png','roc.png','learning.png','decision.png']}
        if actual_images!=own_images:raise ValueError('Word must contain only its four own figures')
        equations=len(xml.findall('.//m:oMath',NS))
        if equations!=3:raise ValueError('Expected three editable Word equations')
        tables=body.findall('w:tbl',NS)
        if len(tables)!=4:raise ValueError('Expected one author block and three result tables')
        trace=pd.read_csv(ROOT/PROJECT/'paper/final/NUMERIC_TRACEABILITY.csv')
        values=trace.groupby(['analysis','model_id','condition','metric']).value.mean()
        model_lists=[('source',['historical_v3_balanced','l2_v3_ordinary_historical_calibration','l2_risk_symptoms17_historical_calibration',
            'l2_no_symptoms28_historical_calibration','l2_full38_historical_calibration','historical_selection',
            'same_components_uncalibrated_ensemble','selection_independent_calibrated_ensemble'],['roc_auc','ap','brier','log_loss']),
            ('calibration',['raw','weak_v1','offset','logit_c1','logit_cv'],['roc_auc','brier','log_loss']),
            ('adaptation',['frozen_raw','frozen_source_calibrated','adaptation_prevalence','target_offset','target_logit_c1','target_logit_cv'],['roc_auc','brier','log_loss'])]
        checked=0
        for table,(analysis,models,metrics) in zip(tables[1:],model_lists):
            rows=table.findall('w:tr',NS)[1:]
            if len(rows)!=len(models):raise ValueError('Scientific table row coverage changed')
            for row,model in zip(rows,models):
                cells=row.findall('w:tc',NS)[1:]
                for cell,metric in zip(cells,metrics):
                    expected=f'{values.loc[(analysis,model,"{}",metric)]:.4f}'
                    if text(cell)!=expected:raise ValueError('Word table differs from frozen evidence')
                    checked+=1
    return {'sections':6,'editable_equations':equations,'metric_table_cells_verified':checked,'reference_research_copied':False}


def write_manifest():
    build=json.loads((OUT/'BUILD_PROVENANCE.json').read_text())
    inputs=dict(build['inputs'])
    for name in ['verify_conference_paper.py','render_conference_word.py','qa_conference_paper.py']:
        p=ROOT/'tools'/name;inputs[p.relative_to(ROOT).as_posix()]=sha256(p)
    artifacts={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='REFERENCE_FORMAT_MANIFEST.json'}
    (OUT/'REFERENCE_FORMAT_MANIFEST.json').write_text(json.dumps({'scope':'Reference-derived presentation; frozen experimental results unchanged',
        'inputs':inputs,'artifacts':artifacts},indent=2)+'\n',encoding='utf-8')


def verify():
    manifest=json.loads((OUT/'REFERENCE_FORMAT_MANIFEST.json').read_text())
    for section in ['inputs','artifacts']:
        for name,digest in manifest[section].items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe conference paper path')
            current=(ROOT/name).resolve()
            if not current.is_relative_to(ROOT) or sha256(current)!=digest:raise ValueError('Conference paper hash mismatch: '+name)
    qa=json.loads((OUT/'CONFERENCE_QA.json').read_text())
    if qa['visual_review']=='pending' or not qa['all_main_text_10pt'] or not qa['requested_page_range_met']:raise ValueError('Conference render QA incomplete')
    return {**verify_structure(),'pdf_pages':qa['pdf_pages'],'new_model_fits':0,'artifacts_verified':len(manifest['artifacts'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',action='store_true');args=parser.parse_args()
    if args.manifest:write_manifest()
    else:print(json.dumps(verify(),indent=2))
