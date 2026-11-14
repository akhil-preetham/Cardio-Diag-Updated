"""Download only small, openly licensed metadata files used by the audit."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "public_dataset_acquisition_v1" / "downloads"
FILES = {
    "uci_heart_disease_official.zip": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    "AngioCAD_Features.xlsx": "https://zenodo.org/api/records/15826856/files/AngioCAD_Features.xlsx/content",
    "AngioCAD_Labels.xlsx": "https://zenodo.org/api/records/15826856/files/AngioCAD_Labels.xlsx/content",
}


def download(output_dir: Path = DEFAULT_OUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = output_dir / name
        if not target.exists():
            urlretrieve(url, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    download(args.output_dir)
