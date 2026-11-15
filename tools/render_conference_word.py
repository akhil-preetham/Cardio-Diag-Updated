"""Windows document-runtime adapter for the packaged DOCX renderer.

Microsoft Word opens input read-only and exports PDF. The packaged renderer
handles page naming/output; PDFium replaces unavailable Windows Poppler/LO.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pypdfium2 as pdfium


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--renderer',type=Path,default=os.environ.get('CARDIODIAG_DOCX_RENDERER'))
    args=parser.parse_args()
    if sys.platform!='win32':raise RuntimeError('This adapter requires Windows and installed Microsoft Word')
    if not args.renderer or not args.renderer.is_file():parser.error('Provide the bundled documents skill render_docx.py with --renderer')
    spec=importlib.util.spec_from_file_location('packaged_docx_renderer',args.renderer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

    def convert(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
        pdf=Path(convert_tmp_dir)/(stem+'.pdf');script=Path(convert_tmp_dir)/'export.ps1'
        script.write_text(r'''param([string]$InputDoc,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$word=$null
$document=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false
 $word.DisplayAlerts=0
 $word.AutomationSecurity=3
 $document=$word.Documents.Open($InputDoc,$false,$true,$false)
 $document.Repaginate()
 $document.ExportAsFixedFormat($OutputPdf,17)
} finally {
 if($null -ne $document){$document.Close(0);[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)}
 if($null -ne $word){$word.Quit();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)}
}
''',encoding='utf-8')
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',str(script),
            '-InputDoc',str(Path(doc_path).resolve()),'-OutputPdf',str(pdf)],capture_output=True,text=True,timeout=180)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return str(pdf),'Microsoft Word read-only ExportAsFixedFormat backend'

    def raster(pdf_path,dpi=144,output_folder=None,**kwargs):
        doc=pdfium.PdfDocument(pdf_path);paths=[]
        try:
            for i,page in enumerate(doc):
                path=Path(output_folder)/f'word-page-{i+1:03}.png'
                bitmap=page.render(scale=dpi/72);image=bitmap.to_pil()
                image.save(path);image.close();bitmap.close();page.close();paths.append(str(path))
        finally:doc.close()
        return paths

    module.convert_to_pdf=convert;module.convert_from_path=raster
    sys.argv=[str(args.renderer),str(args.input),'--output_dir',str(args.output),'--dpi','144','--emit_pdf']
    module.main()


if __name__=='__main__':main()
