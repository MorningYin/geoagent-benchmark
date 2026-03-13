"""Module 2: Vision Parser — parse street view images with qwen-vl-max."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import DashScopeClient
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "01_images"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "02_parsed"
MANIFEST_PATH = DATA_IN / "manifest.jsonl"
PARSED_PATH = DATA_OUT / "parsed.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"

VISION_PROMPT = """你是一个地理场景分析专家。请仔细观察这张街景照片，提取以下结构化信息。

返回严格的 JSON 格式：
{
  "scene_type": "street/intersection/commercial_area/park/transport_hub/residential/campus/tourist_spot/other",
  "time_hint": "daytime/nighttime/dawn_dusk/unknown",
  "weather_hint": "clear/cloudy/rain/snow/foggy/unknown",
  "ocr_texts": ["图中可见的所有文字，如路牌、店名、标识等"],
  "visible_entities": [
    {"type": "building/road/sign/vehicle/vegetation/person/other", "description": "简短描述"}
  ],
  "geo_hints": {
    "level": "city/district/street/building/unknown",
    "hint": "根据可见线索推断的地理位置描述"
  },
  "overall_description": "用一句话描述这张图的主要内容"
}

注意：
1. ocr_texts 只写你确信能看到的文字
2. visible_entities 最多列 8 个最重要的
3. geo_hints 根据路牌、建筑风格、文字语言等推断
4. 如果无法确定某项，用 unknown 或空数组"""


class VisionParser:
    """Parse street view images using a vision LLM."""

    def __init__(self, llm_client: DashScopeClient):
        self.llm_client = llm_client

    def parse_image(self, image_path: str | Path) -> Dict[str, Any]:
        return self.llm_client.vision_json_completion(image_path, VISION_PROMPT, temperature=0.3)


def _assess_richness(parse_result: Dict[str, Any]) -> str:
    """Programmatically assess information richness: high/medium/low."""
    score = 0
    ocr = parse_result.get("ocr_texts", [])
    entities = parse_result.get("visible_entities", [])
    geo = parse_result.get("geo_hints", {})

    if len(ocr) >= 3:
        score += 2
    elif len(ocr) >= 1:
        score += 1

    if len(entities) >= 4:
        score += 2
    elif len(entities) >= 2:
        score += 1

    if geo.get("level") in ("street", "building"):
        score += 2
    elif geo.get("level") in ("district", "city"):
        score += 1

    if parse_result.get("scene_type") not in ("other", "unknown"):
        score += 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _assess_usable_tasks(parse_result: Dict[str, Any]) -> List[str]:
    """Determine which task types this image could support."""
    tasks = []
    ocr = parse_result.get("ocr_texts", [])
    scene = parse_result.get("scene_type", "")

    # Almost any image with geo hints can support place_filter_rank
    tasks.append("place_filter_rank")

    if scene in ("commercial_area", "street", "intersection"):
        tasks.append("place_discovery")
    if ocr:
        tasks.append("place_lookup")
    if scene == "transport_hub":
        tasks.append("route_planning")

    return tasks


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = DashScopeClient(config)
    parser = VisionParser(llm)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(MANIFEST_PATH)
    if not manifest:
        print("[Module 2] No manifest found. Run module 1 first.")
        return PARSED_PATH

    results: List[Dict[str, Any]] = []
    filtered_count = 0

    for i, record in enumerate(manifest, 1):
        image_path = record.get("image_path", "")
        image_id = record.get("image_id", "")
        print(f"[Module 2] Parsing image {i}/{len(manifest)}: {image_id}")

        if not Path(image_path).exists():
            append_error(ERRORS_PATH, {"image_id": image_id, "error": "image file not found"})
            continue

        try:
            parse_result = parser.parse_image(image_path)
        except Exception as e:
            append_error(ERRORS_PATH, {"image_id": image_id, "error": str(e)})
            continue

        richness = _assess_richness(parse_result)
        usable_tasks = _assess_usable_tasks(parse_result)

        if richness == "low":
            filtered_count += 1
            print(f"  -> filtered (low richness)")
            continue

        parsed_record = {
            **record,
            "vision_parse": parse_result,
            "information_richness": richness,
            "usable_for_tasks": usable_tasks,
        }
        results.append(parsed_record)
        print(f"  -> richness={richness}, usable_tasks={usable_tasks}")

    count = write_jsonl(PARSED_PATH, results)
    print(f"[Module 2] Done. {count} parsed, {filtered_count} filtered (low richness)")
    return PARSED_PATH


if __name__ == "__main__":
    main()
