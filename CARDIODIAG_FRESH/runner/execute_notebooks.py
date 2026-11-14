from __future__ import annotations
import hashlib, json, shutil, sys, traceback
from pathlib import Path
from common import load_config, resolve, write_json, append_jsonl, utc_now

def main(config_file):
    cfg, root = load_config(Path(config_file).resolve())
    settings = cfg["notebook_execution"]
    out_root = resolve(root, cfg["output_root"])
    log_dir = out_root / "logs" / "author_faithful"
    work_root = out_root / "author_faithful" / "working_copies"
    results_root = out_root / "author_faithful" / "executed_notebooks"
    log_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    if not settings.get("enabled", False):
        print("Notebook execution is DISABLED in config. Nothing executed.")
        print("Review audit reports, then set notebook_execution.enabled=true and add exact relative paths to allowlist.")
        return
    if settings.get("approval_required", True) and not settings.get("approval_granted", False):
        raise SystemExit("Execution requires explicit approval. Set approval_granted=true only after review.")

    allowlist = list(settings.get("allowlist", []))
    if not allowlist:
        raise SystemExit("Execution enabled but allowlist is empty. Refusing to execute all notebooks.")

    targets = []
    for configured_path in allowlist:
        target = resolve(root, configured_path)
        if target.suffix.lower() != ".ipynb" or not target.is_file():
            raise SystemExit(f"Allowlisted notebook does not exist or is not an .ipynb file: {target}")
        targets.append((configured_path, target))

    dataset = settings.get("dataset", {})
    dataset_path = resolve(root, dataset.get("path", ""))
    expected_dataset_hash = dataset.get("sha256")
    if not expected_dataset_hash or not dataset_path.is_file():
        raise SystemExit("Execution requires an existing dataset path and expected SHA-256 in config.")
    actual_dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_dataset_hash != expected_dataset_hash:
        raise SystemExit(f"Dataset hash mismatch: {dataset_path}")

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        raise SystemExit("Missing nbformat/nbclient. Install reviewed requirements first: pip install nbformat nbclient ipykernel")

    manifest_path = log_dir / "execution_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    completed = {item["notebook"] for item in previous if item.get("status") == "success"}
    manifest = list(previous)
    for configured_path, nb in targets:
        rel = str(Path(configured_path))
        source_hash_before = hashlib.sha256(nb.read_bytes()).hexdigest()

        # Skip patterns are advisory filters; explicit allowlist still controls selection.
        skip = any(p.lower() in nb.name.lower() for p in settings.get("skip_patterns", []))
        if skip:
            if not any(item.get("notebook") == rel for item in manifest):
                manifest.append({"notebook": rel, "status": "skipped_by_pattern", "time_utc": utc_now()})
            continue
        if rel in completed:
            print(f"SKIPPED_COMPLETED: {rel}")
            continue

        token = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
        target_dir = work_root / f"{nb.stem}_{token}"
        output_nb = results_root / f"{nb.stem}_{token}_executed.ipynb"
        target_dir.mkdir(parents=True, exist_ok=True)
        working_copy = target_dir / nb.name
        if working_copy.exists() or output_nb.exists():
            raise SystemExit(
                f"Refusing to overwrite an existing run artifact: {working_copy} or {output_nb}"
            )
        shutil.copy2(nb, working_copy)

        entry = {"notebook": rel, "source_sha256_before": source_hash_before,
                 "dataset": str(dataset_path), "dataset_sha256": actual_dataset_hash,
                 "working_copy": str(working_copy), "output": str(output_nb),
                 "started_utc": utc_now(), "status": "running"}
        append_jsonl(log_dir / "execution_events.jsonl", entry)
        try:
            notebook = nbformat.read(working_copy, as_version=4)
            client = NotebookClient(
                notebook, timeout=int(settings.get("timeout_seconds", 3600)),
                kernel_name=settings.get("kernel_name", "python3"),
                resources={"metadata": {"path": str(target_dir)}}
            )
            executed = client.execute()
            nbformat.write(executed, output_nb)
            source_hash_after = hashlib.sha256(nb.read_bytes()).hexdigest()
            if source_hash_after != source_hash_before:
                raise RuntimeError("Source notebook hash changed during execution; refusing success status.")
            entry.update({"status": "success", "finished_utc": utc_now(),
                          "source_sha256_after": source_hash_after})
        except Exception as e:
            entry["source_sha256_after"] = hashlib.sha256(nb.read_bytes()).hexdigest()
            entry.update({"status": "failed", "finished_utc": utc_now(),
                          "error": repr(e), "traceback": traceback.format_exc()})
            (log_dir / f"{nb.stem}_error.txt").write_text(entry["traceback"], encoding="utf-8")
        manifest = [item for item in manifest if item.get("notebook") != rel]
        manifest.append(entry)
        append_jsonl(log_dir / "execution_events.jsonl", entry)
        write_json(log_dir / "execution_manifest.json", manifest)
        print(f"{entry['status'].upper()}: {rel}")

    write_json(log_dir / "execution_manifest.json", manifest)
    print(f"Execution manifest: {log_dir / 'execution_manifest.json'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/execute_notebooks.py config/runner_config.json")
    main(sys.argv[1])

