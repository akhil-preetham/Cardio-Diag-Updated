"""Build the Phase 7.4.1 SHA-256 artifact manifest without fitting models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "results/phase7_4_1_qa"
MANIFEST = QA / "PHASE7_4_1_MANIFEST.json"

REPORTS = [
    "CLEAN_ENVIRONMENT_VERIFICATION.md",
    "FIGURE2_CORRECTION_REPORT.md",
    "MANUSCRIPT_RENDERING_REPORT.md",
    "TRIPOD_AI_CHECKLIST.md",
    "PROBAST_AI_ASSESSMENT.md",
    "CORRECTION_LOG.md",
    "PHASE7_4_1_FINAL_DECISION.md",
]
SUPPORT = [
    "test_clean_environment_smoke.py",
    "render_markdown_manuscript.py",
    "verify_render.py",
    "build_manifest.py",
    "verify_manifest.py",
]
INPUTS = [
    "results/phase7_4_final_qa/requirements-pinned-working-env.txt",
    "results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv",
    "paper/phase7_3_manuscript/PHASE7_3_MANIFEST.json",
    "paper/phase7_3_manuscript/MANUSCRIPT.md",
    "paper/phase7_3_manuscript/build_tables_figures.py",
    "paper/phase7_3_manuscript/FIGURES/FIGURE2_PHASE6_REPEAT_AUC.png",
    "paper/phase7_3_manuscript/FIGURES/FIGURE2_PHASE6_REPEAT_AUC.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifact_paths = set(INPUTS)
    artifact_paths.update(f"results/phase7_4_1_qa/{name}" for name in REPORTS + SUPPORT)
    artifact_paths.update(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "paper/phase7_4_1_final_manuscript").rglob("*")
        if path.is_file()
    )
    artifact_paths.update(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (QA / "rendered_pages_v2").glob("*.png")
        if path.is_file()
    )

    manifest = {
        "phase": "Phase 7.4.1 — Final Technical Corrections and Rendering Verification",
        "created_local_date": "2026-11-14",
        "decision": "READY WITH SPECIFIED REMAINING ISSUES",
        "repository_root": str(ROOT),
        "historical_preservation": {
            "phase7_3_manifest": "paper/phase7_3_manuscript/PHASE7_3_MANIFEST.json",
            "input_hashes_checked": 46,
            "output_hashes_checked": 28,
            "mismatches": 0,
            "git_metadata_available": False,
        },
        "clean_environment": {
            "path": ".venv-phase7_4_1",
            "python": "3.11.16",
            "dependency_snapshot": "results/phase7_4_final_qa/requirements-pinned-working-env.txt",
            "pinned_package_count": 126,
            "matching_installed_package_count": 126,
            "unmatched_or_extra_project_packages": 0,
            "pip_check": "passed: no broken requirements",
            "safe_smoke_tests": "4 passed",
            "model_fitting_or_benchmark_run": False,
        },
        "figure2": {
            "source": "paper/phase7_4_1_final_manuscript/regenerate_figure2.py",
            "data": "results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv",
            "axis_before": [0.65, 1.0],
            "axis_after": [0.0, 1.0],
            "values_changed": False,
            "formats_regenerated_together": ["PNG", "PDF"],
            "originals_preserved": True,
        },
        "manuscript_render": {
            "pdf": "paper/phase7_4_1_final_manuscript/MANUSCRIPT.pdf",
            "method": "Markdown rendered to PDF with bundled ReportLab; not a LaTeX compile",
            "pages": 12,
            "page_size_points": [612, 792],
            "all_pages_rasterized_and_visually_reviewed": True,
            "text_issues": [],
            "missing_expected_content": [],
            "pdf_link_annotations": 0,
        },
        "checklist_scope": {
            "tripod_ai_items_and_subitems": 52,
            "probast_ai_development_signalling_questions": 16,
            "probast_ai_evaluation_signalling_questions": 18,
            "probast_ai_applicability_cells": 6,
            "compliance_or_low_risk_claimed": False,
        },
        "manifest_self_hash_included": False,
        "artifacts_sha256": {
            relative: sha256(ROOT / relative)
            for relative in sorted(artifact_paths)
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST} with {len(artifact_paths)} artifact hashes")


if __name__ == "__main__":
    main()
