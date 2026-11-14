from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_config(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = (config_path.parent / config.get("project_root", "..")).resolve()
    return config, project_root

def resolve(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
