from __future__ import annotations
import importlib.metadata
import sys, platform, importlib.util, subprocess
from pathlib import Path
from common import load_config, resolve, write_json, utc_now

MODULES = ["numpy", "pandas", "sklearn", "imblearn", "xgboost",
           "matplotlib", "jupyter", "nbformat", "nbclient", "ipykernel"]
DIST_NAMES = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}

def main(config_file):
    cfg, root = load_config(Path(config_file).resolve())
    extracted = resolve(root, cfg["extracted_archives"])
    datasets = {k: resolve(root, v) for k, v in cfg["datasets"].items()}
    out = resolve(root, cfg["output_root"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    modules = {m: bool(importlib.util.find_spec(m)) for m in MODULES}
    data = {k: {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None}
            for k, p in datasets.items()}
    git = None
    try:
        git = subprocess.check_output(["git", "-C", str(resolve(root, cfg["author_repository"])),
                                       "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    package_versions = {}
    for module in MODULES:
        try:
            package_versions[module] = importlib.metadata.version(DIST_NAMES.get(module, module))
        except importlib.metadata.PackageNotFoundError:
            package_versions[module] = None
    report = {
        "created_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "extracted_root_exists": extracted.exists(),
        "notebook_count": len(list(extracted.rglob("*.ipynb"))) if extracted.exists() else 0,
        "modules_available": modules, "package_versions": package_versions,
        "datasets": data, "author_git_commit": git,
        "author_repository_clean": git is not None,
    }
    write_json(out / "preflight.json", report)
    write_json(resolve(root, cfg["output_root"]) / "dependency_versions.json", package_versions)
    print(report)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/preflight.py config/runner_config.json")
    main(sys.argv[1])
