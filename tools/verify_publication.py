"""Verify the published file inventory without importing model code or unpickling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, digest in manifest['sha256'].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid inventory path: {relative}')
        target = (root / Path(*path.parts)).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError(relative)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Publication hash mismatch: {relative}')
    return len(manifest['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    count = verify(args.root)
    print(f'Publication inventory verified: {count} files; no model fits or pickle loads')
