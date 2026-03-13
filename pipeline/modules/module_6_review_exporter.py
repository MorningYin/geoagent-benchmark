"""Module 6: Review Exporter — XLSX batch review + HTML review cards."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.jsonl_io import read_jsonl

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "05_queries"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "06_review"
FINAL_PATH = DATA_IN / "final.jsonl"
XLSX_PATH = DATA_OUT / "review_batch.xlsx"
CARDS_DIR = DATA_OUT / "cards"


def _export_xlsx(tasks: List[Dict[str, Any]]) -> Path:
    """Export tasks to XLSX for batch review."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill
    except ImportError:
        print("[Module 6] openpyxl not installed. Skipping XLSX export.")
        return XLSX_PATH

    wb = Workbook()
    ws = wb.active
    ws.title = "Review"

    headers = [
        "image_id", "task_type", "category", "query", "ranking_objective",
        "distance_limit_m", "min_rating", "answer_place", "answer_distance_m",
        "scene_type", "richness", "review_priority", "passed",
    ]
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font

    # Sort by review_priority descending
    sorted_tasks = sorted(tasks, key=lambda t: t.get("quality_report", {}).get("review_priority", 1), reverse=True)

    for row_idx, task in enumerate(sorted_tasks, 2):
        td = task.get("task_dimensions", {})
        ad = task.get("answer_data", {})
        qr = task.get("quality_report", {})
        vp = task.get("vision_parse", {})

        values = [
            task.get("image_id", ""),
            task.get("task_type", ""),
            td.get("place_category_canonical", ""),
            task.get("natural_language_query", "")[:100],
            td.get("ranking_objective", ""),
            td.get("distance_limit_m"),
            td.get("min_rating"),
            ad.get("expected_place_name", ""),
            ad.get("expected_distance_m"),
            vp.get("scene_type", ""),
            task.get("information_richness", ""),
            qr.get("review_priority", ""),
            qr.get("passed", ""),
        ]
        for col, v in enumerate(values, 1):
            ws.cell(row=row_idx, column=col, value=v)

    # Adjust column widths
    for col in range(1, len(headers) + 1):
        ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = 18
    ws.column_dimensions["D"].width = 50  # query column wider

    wb.save(XLSX_PATH)
    return XLSX_PATH


def _export_html_card(task: Dict[str, Any], output_path: Path) -> None:
    """Export a single task as an HTML review card."""
    td = task.get("task_dimensions", {})
    gc = task.get("global_context", {})
    ad = task.get("answer_data", {})
    qr = task.get("quality_report", {})
    vp = task.get("vision_parse", {})
    igp = task.get("information_gap_plan", {})
    image_path = task.get("image_path", "")

    # Use relative path for image
    image_rel = ""
    if image_path:
        try:
            image_rel = str(Path(image_path).relative_to(Path(output_path).parent))
        except ValueError:
            image_rel = image_path

    candidates_html = ""
    for c in ad.get("candidate_pois", []):
        candidates_html += f"<tr><td>{c.get('name','')}</td><td>{c.get('distance_m','')}</td><td>{c.get('rating','')}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Review: {task.get('image_id','')}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 20px auto; padding: 0 15px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 15px 0; }}
.card h3 {{ margin-top: 0; color: #333; }}
img {{ max-width: 100%; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
.tag {{ display: inline-block; background: #e8f0fe; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.85em; }}
.pass {{ color: green; font-weight: bold; }}
.fail {{ color: red; font-weight: bold; }}
pre {{ background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Review Card: {task.get('image_id','')}</h1>

<div class="card">
<h3>Street View Image</h3>
<img src="{image_rel}" alt="street view">
<p><strong>Scene:</strong> {vp.get('scene_type','')},
   <strong>Weather:</strong> {vp.get('weather_hint','')},
   <strong>Time:</strong> {vp.get('time_hint','')}</p>
<p><strong>OCR Texts:</strong> {', '.join(vp.get('ocr_texts', []))}</p>
<p><strong>Geo Hint:</strong> {json.dumps(vp.get('geo_hints', {}), ensure_ascii=False)}</p>
</div>

<div class="card">
<h3>Natural Language Query</h3>
<p style="font-size: 1.1em; background: #fffde7; padding: 10px; border-radius: 4px;">
{task.get('natural_language_query', '')}
</p>
</div>

<div class="card">
<h3>Task Dimensions</h3>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>task_type</td><td>{task.get('task_type','')}</td></tr>
<tr><td>category</td><td>{td.get('place_category_canonical','')}</td></tr>
<tr><td>ranking_objective</td><td>{td.get('ranking_objective','')}</td></tr>
<tr><td>distance_limit_m</td><td>{td.get('distance_limit_m','')}</td></tr>
<tr><td>min_rating</td><td>{td.get('min_rating','')}</td></tr>
<tr><td>must_be_open_now</td><td>{td.get('must_be_open_now','')}</td></tr>
<tr><td>result_count_limit</td><td>{td.get('result_count_limit','')}</td></tr>
</table>
</div>

<div class="card">
<h3>Answer</h3>
<p><strong>Expected:</strong> {ad.get('expected_place_name','')} (distance: {ad.get('expected_distance_m','')}m, rating: {ad.get('expected_rating','')})</p>
<h4>Candidate POIs</h4>
<table>
<tr><th>Name</th><th>Distance (m)</th><th>Rating</th></tr>
{candidates_html}
</table>
</div>

<div class="card">
<h3>Information Gap Plan</h3>
<p><strong>Image provides:</strong> {', '.join(f'<span class="tag">{x}</span>' for x in igp.get('image_provides', []))}</p>
<p><strong>Query must state:</strong> {', '.join(f'<span class="tag">{x}</span>' for x in igp.get('query_must_state', []))}</p>
<p><strong>Must not leak:</strong> {', '.join(f'<span class="tag">{x}</span>' for x in igp.get('must_not_leak_in_query', []))}</p>
</div>

<div class="card">
<h3>Quality Report</h3>
<p>Status: <span class="{'pass' if qr.get('passed') else 'fail'}">{'PASSED' if qr.get('passed') else 'REJECTED'}</span></p>
<p>Review Priority: {qr.get('review_priority', '')}</p>
{'<p>Rejection reasons: ' + str(qr.get("rejection_reasons", [])) + '</p>' if not qr.get('passed') else ''}
</div>

<div class="card">
<h3>Global Context</h3>
<pre>{json.dumps(gc, ensure_ascii=False, indent=2)}</pre>
</div>

</body></html>"""

    output_path.write_text(html, encoding="utf-8")


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(FINAL_PATH)
    if not tasks:
        print("[Module 6] No final samples found. Run module 5 first.")
        return XLSX_PATH

    # XLSX export
    xlsx_path = _export_xlsx(tasks)
    print(f"[Module 6] XLSX exported: {xlsx_path}")

    # HTML cards
    for task in tasks:
        image_id = task.get("image_id", "unknown")
        card_path = CARDS_DIR / f"{image_id}.html"
        _export_html_card(task, card_path)

    print(f"[Module 6] {len(tasks)} HTML cards exported to {CARDS_DIR}")
    return xlsx_path


if __name__ == "__main__":
    main()
