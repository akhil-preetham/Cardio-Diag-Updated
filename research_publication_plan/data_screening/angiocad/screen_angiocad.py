"""Acquire only official small tabular files; report aggregate schema/label checks.

Run with Python 3.11+. Uses the standard library; never downloads image archives.
This is data screening, not model development or external validation.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
API = "https://zenodo.org/api/records/15826856"
FILES = ("AngioCAD_Features.xlsx", "AngioCAD_Labels.xlsx")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SEVERE = {"51-75", "76-90", "91-99", "100"}
VESSELS = (
    "LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig",
    "Prox LCX", "Mid LCX", "Dist LCX", "OM", "Prox RCA", "Mid RCA",
    "Dist RCA", "PDA", "PLB",
)


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CARDIODIAG-data-screening/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def read_table(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                node = cell.find("m:v", NS)
                value = node.text if node is not None else ""
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", NS).itertext())
                column = "".join(c for c in cell.get("r", "") if c.isalpha())
                values[column] = value
            rows.append(values)
    header = rows[0]
    # Ignore trailing worksheet formatting rows without a case ID.
    records = [{header.get(k, k).strip(): v for k, v in row.items()}
               for row in rows[1:] if row.get("A", "").strip()]
    return [v.strip() for v in header.values()], records


def summarize_labels(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = collections.Counter()
    for row in rows:
        values = {row.get(vessel, "").strip() for vessel in VESSELS}
        if values & SEVERE:
            counts["any_segment_gt50"] += 1
        else:
            counts["no_segment_gt50"] += 1
            if "26-50" in values:
                counts["no_gt50_but_26_50_boundary_ambiguous_for_ge50"] += 1
            else:
                counts["no_gt50_and_no_26_50"] += 1
    return dict(counts)


def main() -> None:
    metadata_path = ROOT / "zenodo_record_15826856.json"
    if not metadata_path.exists():
        metadata_path.write_bytes(get(API))
    record = json.loads(metadata_path.read_bytes())
    if record["metadata"].get("access_right") != "open":
        raise RuntimeError("Provider record no longer says open access")
    provenance = []
    tables = {}
    for name in FILES:
        entry = next(f for f in record["files"] if f["key"] == name)
        url = entry["links"]["self"]
        path = ROOT / name
        if not path.exists():
            path.write_bytes(get(url))
        payload = path.read_bytes()
        if len(payload) != entry["size"]:
            raise RuntimeError(f"Provider size mismatch: {name}")
        provider_md5 = entry.get("checksum", "")
        if provider_md5.startswith("md5:") and hashlib.md5(payload).hexdigest() != provider_md5[4:]:
            raise RuntimeError(f"Provider checksum mismatch: {name}")
        headers, rows = read_table(payload)
        tables[name] = {"headers": headers, "rows": rows}
        provenance.append({"file": name, "url": url, "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest(), "provider_checksum": provider_md5})
    features = tables[FILES[0]]["rows"]
    labels = tables[FILES[1]]["rows"]
    feature_ids = [r["ID"] for r in features]
    label_ids = [r["ID"] for r in labels]
    if len(feature_ids) != len(set(feature_ids)) or len(label_ids) != len(set(label_ids)):
        raise RuntimeError("Duplicate case IDs require investigation before join")
    common = set(feature_ids) & set(label_ids)
    aggregate = {
        "feature_rows": len(features), "feature_unique_ids": len(set(feature_ids)),
        "label_rows": len(labels), "label_unique_ids": len(set(label_ids)),
        "matched_unique_ids": len(common),
        "feature_only_ids": len(set(feature_ids) - set(label_ids)),
        "label_only_ids": len(set(label_ids) - set(feature_ids)),
        "feature_columns": tables[FILES[0]]["headers"],
        "label_columns": tables[FILES[1]]["headers"],
        "diagnosis_counts": dict(collections.Counter(r.get("Diagnosis", "") for r in features)),
        "all_label_class_counts_provisional_gt50": summarize_labels(labels),
        "matched_class_counts_provisional_gt50": summarize_labels([r for r in labels if r["ID"] in common]),
        "all_model_fits": 0,
    }
    (ROOT / "aggregate_screening.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    manifest = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "record_url": "https://zenodo.org/records/15826856", "api_url": API,
                "doi": record["metadata"]["doi"], "license": record["metadata"]["license"],
                "access_right": record["metadata"]["access_right"],
                "scope": "Only two clinical/label workbooks and provider metadata; no image archive",
                "files": provenance,
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "screening_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = aggregate["matched_class_counts_provisional_gt50"]
    report = f"""# AngioCAD tabular screening

Official source: https://zenodo.org/records/15826856 (DOI 10.5281/zenodo.15826856).
Provider metadata reports open access under CC BY 4.0. Downloaded only the two small
clinical/label workbooks; all image archives were excluded. File sizes, provider MD5
and local SHA-256 are recorded in PROVENANCE_MANIFEST.json. Source data are preserved
unchanged; all reports are aggregate. No model was fitted.

| Screening quantity | Count |
|---|---:|
| Nonblank clinical-feature IDs | {len(feature_ids)} |
| Nonblank label IDs | {len(label_ids)} |
| Matched unique IDs | {len(common)} |
| Matched provisional >50% cases | {counts['any_segment_gt50']} |
| Matched no-segment->50% cases | {counts['no_segment_gt50']} |
| Matched nonpositive cases containing a 26-50 segment | {counts.get('no_gt50_but_26_50_boundary_ambiguous_for_ge50', 0)} |

## Endpoint sensitivity

The provisional positive rule is any segment in 51-75, 76-90, 91-99 or 100.
It operationalizes **>50%** recorded segment stenosis, not ZAS's **>=50%** outcome.
The 26-50 bucket combines exact-50 with lower stenosis and cannot distinguish them.
Cases without a >50 segment but with a 26-50 segment are boundary-ambiguous for >=50.
Do not label all 26-50 cases positive as if this represented >=50; that is an upper-bound
sensitivity assignment, not the true endpoint. Present lower/upper assignments only
as sensitivity analyses with their changed definitions. No-segment->50 controls are
non-obstructive under this rule and are not necessarily normal/healthy.

## Clinical and feature compatibility

The clinical table's Diagnosis counts (aggregate_screening.json) include STEMI,
NSTEMI, unstable angina and stable angina. This coronary referral spectrum differs
from population screening and may differ substantially from ZAS. FH, EX-Smoker and
the ZAS symptom/ECG groups are absent from the header. An unchanged five-feature V3
or 38-field model therefore cannot be externally evaluated with these fields.

A new shared-feature development procedure could be planned using Age, Sex, DM,
HTN, Current Smoker and verified common laboratory variables. Check unit/encoding,
measurement timing, missingness, eligibility and outcome mapping before fitting.
Exclude outcome-defining artery labels, Gensini Score, Diagnosis and diagnostic
series fields. LVEF/echo and treatment fields require explicit timing restrictions.
Do not select mappings, predictors or model settings by external-cohort performance.

The 413 label cases do not imply 413 usable tabular cases: only 377 have feature IDs.
Reasons for the 36 missing clinical cases remain unresolved and may introduce selection.
Unique IDs are case keys in this release; this alone does not establish every case
represents an independent patient with no repeated visits.

## Reproduce

Run `python screen_angiocad.py` in this directory. Existing downloads are verified
and reused; the script prints only aggregate counts. It writes aggregate JSON,
this report and provenance metadata. It does not train or generate a paper result.
"""
    (ROOT / "SCREENING_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: aggregate[k] for k in ["feature_rows", "label_rows", "matched_unique_ids", "matched_class_counts_provisional_gt50", "all_model_fits"]}))


if __name__ == "__main__":
    main()
