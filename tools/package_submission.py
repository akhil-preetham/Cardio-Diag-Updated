"""Bundle reviewed reporting artifacts without raw data or licensed font files."""
import json
from pathlib import Path
import zipfile
from portable_release import discover_workspace, PROJECT, sha256
from paper_evidence import verify
from verify_conference_paper import verify as verify_conference


def main():
    print(json.dumps(verify(),indent=2))
    root=discover_workspace();out=root/PROJECT/'paper/final'
    manifest=out/'PAPER_EVIDENCE_MANIFEST.json'
    data=json.loads(manifest.read_text())
    files=[root/name for name in data['artifact_hashes']]+[manifest]
    print(json.dumps(verify_conference(),indent=2))
    conference=root/PROJECT/'paper/reference_format'
    extra=conference/'REFERENCE_FORMAT_MANIFEST.json'
    files.extend([root/name for name in json.loads(extra.read_text())['artifacts']]+[extra])
    if len({p.name for p in files})!=len(files):
        raise ValueError('Duplicate package filenames')
    archive=root.parent/'CARDIODIAG_REVISED_SUBMISSION_PACKAGE.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if path.parent not in [out,conference] or path.suffix.lower() in ['.ttf','.otf']:
                raise ValueError('Only reviewed reporting files may be bundled')
            z.write(path,'CARDIODIAG_Revised_Package/'+path.name)
        checks='\n'.join(sha256(path)+'  '+path.name for path in files)+'\n'
        z.writestr('CARDIODIAG_Revised_Package/SHA256SUMS.txt',checks)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Invalid package archive')
    print(archive)


if __name__=='__main__':main()
