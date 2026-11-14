from __future__ import annotations
import csv, hashlib, json, re, sys
from pathlib import Path
from common import load_config, resolve, write_json, utc_now

IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)
RISK_PATTERNS = {
    "shell_or_process": r"\b(subprocess\.|os\.system|Popen\(|shell=True)",
    "network_or_download": r"\b(requests\.|urllib\.|wget\b|curl\b|git\s+clone)",
    "file_deletion": r"\b(shutil\.rmtree|os\.remove|os\.unlink|Path\([^)]*\)\.unlink)",
    "dynamic_execution": r"\b(eval\(|exec\()",
    "credential_or_mount": r"\b(google\.colab|drive\.mount|keyring|password|token|api_key)\b",
    "notebook_shell": r"^\s*!|^\s*%",
}
METHOD_TERMS = [
    "SMOTE", "LoRAS", "XAIH", "OVEC", "OSEC", "LIME",
    "GridSearchCV", "RandomizedSearchCV", "StackingClassifier",
    "VotingClassifier", "bootstrap", "wilcoxon", "delong",
    "CalibratedClassifierCV", "train_test_split"
]
STDlib = {
    "collections", "json", "math", "os", "pathlib", "random", "shutil",
    "statistics", "sys", "time", "warnings",
}

def source_text(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def classify_notebook(relative_path, source, imports, hits, risks, paths):
    lower = relative_path.lower()
    if "base_reproduction" in lower:
        order, phase = 10, "base reproduction"
        runtime, memory = "5-15 minutes", "<1 GB"
    elif "full_methodology" in lower:
        order, phase = 20, "full methodology"
        runtime, memory = "15-45 minutes", "2-4 GB"
    elif "xaih" in lower:
        order, phase = 30, "XAIH-DRC"
        runtime, memory = "10-30 minutes", "1-2 GB"
    elif "ovec" in lower:
        order, phase = 40, "OVEC"
        runtime, memory = "15-45 minutes", "2-4 GB"
    elif "osec" in lower:
        order, phase = 50, "OSEC"
        runtime, memory = "15-45 minutes", "2-4 GB"
    elif "lime" in lower:
        order, phase = 60, "LIME"
        runtime, memory = "30-120 minutes", "4-8 GB"
    elif "fairness" in lower:
        order, phase = 70, "fairness"
        runtime, memory = "20-60 minutes", "2-4 GB"
    elif any(term in lower for term in ("bootstrap", "wilcoxon", "delong", "clinical")):
        order, phase = 80, "statistics/clinical utility"
        runtime, memory = "10-60 minutes", "2-4 GB"
    else:
        order, phase = 90, "unclassified"
        runtime, memory = "10-30 minutes", "1-2 GB"

    datasets = []
    if "mendeley" in lower or "mendeley" in source.lower():
        datasets.append("MendeleyCardiovascular_Disease_Dataset.csv")
    if any(term in lower or term in source.lower() for term in ("five_states", "five states", "heart5")):
        datasets.append("heart5.csv")
    if not datasets:
        datasets.append("dataset unresolved from static evidence")

    methodological_risks = []
    if "SMOTE" in hits and "train_test_split" in hits:
        methodological_risks.append("Verify SMOTE is applied after splitting; source contains both operations.")
    if "GridSearchCV" in hits and "train_test_split" in hits:
        methodological_risks.append("Verify test data are not used for model selection or repeated tuning.")
    if "XGBRegressor" in source and "accuracy_score" in source:
        methodological_risks.append("XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.")
    if "LabelEncoder" in source:
        methodological_risks.append("LabelEncoder on feature categories may impose arbitrary ordinal structure.")
    if "patientid" in source.lower():
        methodological_risks.append("Patient identifier appears in source; exclude unless a defensible policy is documented.")
    if "dropna" in source:
        methodological_risks.append("Row deletion may change the analysis population and requires missingness justification.")
    if any(term in hits for term in ("LIME", "OVEC", "OSEC", "XAIH")):
        methodological_risks.append("Specialized method is source-detected; correctness and executed code path require manual review.")
    if any(term in hits for term in ("bootstrap", "wilcoxon", "delong")):
        methodological_risks.append("Statistical inference requires independent predictions and a prespecified comparison.")
    if not methodological_risks:
        methodological_risks.append("No additional configured heuristic; static audit is not proof of methodological correctness.")

    external = sorted({
        item.split(".")[0] for item in imports
        if item.split(".")[0] not in STDlib
    })
    colab_paths = sorted(item for item in paths if "/content" in item or "drive" in item.lower())
    return {
        "execution_order": order,
        "execution_phase": phase,
        "dependencies": external,
        "dataset_requirements": sorted(set(datasets)),
        "expected_runtime": runtime,
        "memory_requirement": memory,
        "colab_specific_paths": colab_paths,
        "methodological_risks": methodological_risks,
        "execution_prerequisites": (
            "Run only after prior phase outputs are verified; use a copied notebook and isolated output directory."
            if order > 10 else "Requires local dataset path adaptation; preserve original Colab path in the audit."
        ),
    }

def main(config_file):
    config_path = Path(config_file).resolve()
    cfg, root = load_config(config_path)
    extracted = resolve(root, cfg["extracted_archives"])
    out_root = resolve(root, cfg["output_root"])
    audit_dir = out_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name in ("author_faithful", "corrected_validation", "logs", "reports"):
        (out_root / name).mkdir(parents=True, exist_ok=True)

    notebooks = sorted(extracted.rglob("*.ipynb")) if extracted.exists() else []
    inventory, all_imports, method_hits, risk_records = [], set(), {}, []
    for nb in notebooks:
        raw = nb.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8"))
            cells = data.get("cells", [])
        except Exception as e:
            inventory.append({"path": str(nb), "sha256": digest, "status": "invalid_json", "error": str(e)})
            continue

        cell_records, imports, hits, risks = [], set(), set(), {}
        paths = []
        execution_counts, outputs_count = [], 0
        for idx, cell in enumerate(cells, 1):
            src = source_text(cell)
            ctype = cell.get("cell_type", "unknown")
            if ctype == "code":
                execution_counts.append(cell.get("execution_count"))
                outputs_count += len(cell.get("outputs", []))
                for m in IMPORT_RE.finditer(src):
                    imports.add(m.group(1) or m.group(2))
                for term in METHOD_TERMS:
                    if re.search(re.escape(term), src, re.I):
                        hits.add(term)
                for label, pattern in RISK_PATTERNS.items():
                    if re.search(pattern, src, re.I | re.M):
                        risks.setdefault(label, []).append(idx)
                paths.extend(re.findall(r"""(?:"|')([^"']+\.(?:csv|xlsx|json|pkl|joblib|png|pdf))(?:"|')""", src, re.I))
            cell_records.append({
                "index": idx, "type": ctype, "execution_count": cell.get("execution_count"),
                "source": src
            })

        all_imports.update(imports)
        method_hits[str(nb.relative_to(extracted))] = sorted(hits)
        classification = classify_notebook(
            str(nb.relative_to(extracted)), "\n".join(c["source"] for c in cell_records),
            imports, hits, risks, paths
        )
        inventory.append({
            "path": str(nb), "relative_path": str(nb.relative_to(extracted)),
            "sha256": digest, "bytes": len(raw), "cell_count": len(cells),
            "code_cells": sum(c.get("cell_type") == "code" for c in cells),
            "saved_outputs": outputs_count, "execution_counts": execution_counts,
            "imports": sorted(imports), "method_terms": sorted(hits),
            "risk_flags": risks, "referenced_paths": sorted(set(paths)), "status": "audited",
            **classification
        })
        risk_records.append({
            "relative_path": str(nb.relative_to(extracted)),
            "risk_flags": risks,
            "referenced_paths": sorted(set(paths)),
        })

        md = ["# Static notebook audit", "", f"- Notebook: `{nb}`", f"- SHA-256: `{digest}`",
              f"- Cells: {len(cells)}", f"- Saved outputs: {outputs_count}", "",
              "## Imports", *[f"- `{x}`" for x in sorted(imports)], "",
              "## Method keywords", *[f"- {x}" for x in sorted(hits)], "",
              "## Risk flags"]
        if risks:
            for k, v in risks.items():
                md.append(f"- **{k}**: cells {v}")
        else:
            md.append("- No configured risk patterns detected (this is not a safety guarantee).")
        md += ["", "## Cell sources"]
        for c in cell_records:
            md += ["", f"### Cell {c['index']} — {c['type']} (execution_count={c['execution_count']})",
                   "```python" if c["type"] == "code" else "```text", c["source"], "```"]
        report_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", nb.stem) + "_audit.md"
        (audit_dir / report_name).write_text("\n".join(md), encoding="utf-8")

    report = {
        "created_utc": utc_now(), "extracted_root": str(extracted),
        "notebook_count": len(notebooks), "inventory": inventory,
        "unique_import_roots": sorted({x.split(".")[0] for x in all_imports}),
        "method_hits_by_notebook": method_hits,
        "important_notice": "Static scanning is not a security sandbox. Do not execute an unreviewed notebook."
    }
    write_json(audit_dir / "notebook_inventory.json", report)
    (audit_dir / "notebook_inventory.csv").write_text(
        "relative_path,sha256,cell_count,code_cells,saved_outputs,status\n" +
        "".join(
            '"' + x.get("relative_path","").replace('"','""') + '",' +
            x.get("sha256","") + ',' + str(x.get("cell_count","")) + ',' +
            str(x.get("code_cells","")) + ',' + str(x.get("saved_outputs","")) + ',' +
            x.get("status","") + "\n" for x in inventory
        ), encoding="utf-8"
    )
    write_json(audit_dir / "dependency_imports.json", {"imports": sorted(all_imports)})
    write_json(audit_dir / "dependency_audit.json", {
        "imports": sorted(all_imports),
        "top_level_imports": sorted({x.split(".")[0] for x in all_imports}),
        "notebook_count": len(notebooks),
        "note": "Import presence is evidence of a reference in source, not proof that the code path executed."
    })
    write_json(audit_dir / "methodology_map.json", method_hits)
    execution_plan = [
        {
            "relative_path": item["relative_path"],
            **{key: item[key] for key in (
                "execution_order", "execution_phase", "dependencies",
                "dataset_requirements", "expected_runtime", "memory_requirement",
                "colab_specific_paths", "methodological_risks",
                "execution_prerequisites",
            )}
        }
        for item in inventory if item.get("status") == "audited"
    ]
    execution_plan.sort(key=lambda item: (item["execution_order"], item["relative_path"]))
    write_json(audit_dir / "execution_plan.json", {
        "note": "Runtime and memory are conservative static estimates, not measured execution results.",
        "notebooks": execution_plan,
    })
    with (audit_dir / "execution_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "execution_order", "execution_phase", "relative_path",
            "dependencies", "dataset_requirements", "expected_runtime",
            "memory_requirement", "colab_specific_paths", "methodological_risks",
        ])
        writer.writeheader()
        for item in execution_plan:
            writer.writerow({
                "execution_order": item["execution_order"],
                "execution_phase": item["execution_phase"],
                "relative_path": item["relative_path"],
                "dependencies": "|".join(item["dependencies"]),
                "dataset_requirements": "|".join(item["dataset_requirements"]),
                "expected_runtime": item["expected_runtime"],
                "memory_requirement": item["memory_requirement"],
                "colab_specific_paths": "|".join(item["colab_specific_paths"]),
                "methodological_risks": " | ".join(item["methodological_risks"]),
            })
    inventory_md = [
        "# Complete notebook inventory and risk classification", "",
        "Runtime and memory values are conservative static estimates, not measurements.",
        "No notebook was executed during this audit.", "",
    ]
    for item in execution_plan:
        inventory_md.extend([
            f"## {item['execution_order']}: `{item['relative_path']}`",
            f"- Phase: **{item['execution_phase']}**",
            f"- Dependencies: `{', '.join(item['dependencies']) or 'none detected'}`",
            f"- Dataset requirements: `{', '.join(item['dataset_requirements'])}`",
            f"- Expected runtime: **{item['expected_runtime']}**",
            f"- Memory requirement: **{item['memory_requirement']}**",
            f"- Colab-specific paths: `{', '.join(item['colab_specific_paths']) or 'none detected'}`",
            f"- Prerequisites: {item['execution_prerequisites']}",
            "- Methodological risks:",
            *[f"  - {risk}" for risk in item["methodological_risks"]],
            "",
        ])
    (audit_dir / "complete_notebook_inventory.md").write_text(
        "\n".join(inventory_md), encoding="utf-8"
    )
    (audit_dir / "methodology_map.md").write_text(
        "# Methodology map\n\n"
        "This map records source-level keyword evidence only; it does not establish execution.\n\n"
        + "\n".join(
            f"- `{path}`: {', '.join(terms) if terms else 'no configured method terms detected'}"
            for path, terms in sorted(method_hits.items())
        ) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "risk_audit.md").write_text(
        "# Notebook risk audit\n\n"
        "Static flags require human review before execution and are not a security sandbox.\n\n"
        + "\n".join(
            f"## `{item['relative_path']}`\n\n"
            + ( "\n".join(f"- **{key}**: cells {cells}" for key, cells in item["risk_flags"].items())
                if item["risk_flags"] else "- No configured risk patterns detected.")
            + (f"\n- Referenced paths: {', '.join(item['referenced_paths'])}" if item["referenced_paths"] else "")
            for item in risk_records
        ) + "\n",
        encoding="utf-8",
    )
    supplementary = []
    for path in sorted(extracted.rglob("*")):
        if path.is_file() and path.suffix.lower() != ".ipynb":
            supplementary.append({
                "relative_path": str(path.relative_to(extracted)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    write_json(audit_dir / "supplementary_inventory.json", {
        "root": str(extracted), "file_count": len(supplementary), "files": supplementary
    })
    print(f"Audited {len(notebooks)} notebooks.")
    print(f"Reports: {audit_dir}")
    print("No notebook was executed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runner/audit.py config/runner_config.json")
    main(sys.argv[1])
