"""Module 2: Vision Parser — parse street view images and classify image utility.

Two outputs per image:
  1. vision_parse          — structured scene analysis from the vision LLM
  2. image_utility_class   — anchor / context / weak (drives Module 3 task routing)
  3. anchor_evidence       — (anchor only) extracted readable signs + geo info

Filtering: "weak" images are discarded and do not enter Module 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import BaseLLMClient, ClientFactory, STAGE_VISION_PARSE
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

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def parse_image(self, image_path: str | Path) -> Dict[str, Any]:
        return self.llm_client.vision_json_completion(image_path, VISION_PROMPT, temperature=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Image utility classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_image_utility(parse_result: Dict[str, Any]) -> str:
    """Classify image into anchor / context / weak.

    anchor  — has readable text (OCR) AND fine-grained geo hints (street/building).
              These images can serve as the primary information source for a task;
              without them the agent cannot solve the problem.
    context — has a recognisable scene type or coarse geo hints, but no text that
              uniquely identifies a location. Useful as supplementary context.
    weak    — too little information to contribute to any task. Discard.
    """
    raw_ocr = parse_result.get("ocr_texts", [])
    geo = parse_result.get("geo_hints", {})
    geo_level = geo.get("level", "unknown")
    scene = parse_result.get("scene_type", "unknown")
    entities = parse_result.get("visible_entities", [])

    # Filter out watermarks / copyright / generic noise before classifying
    _NOISE = {"google", "©", "copyright", "imagery", "map data",
              "street view", "all rights reserved"}
    ocr = [t for t in raw_ocr if len(t.strip()) >= 2
           and not any(n in t.lower() for n in _NOISE)]

    has_readable_text = len(ocr) >= 1
    has_fine_geo = geo_level in ("street", "building")
    has_coarse_geo = geo_level in ("district", "city")
    has_clear_scene = scene not in ("other", "unknown")

    # ── anchor: readable text + fine-grained location ──
    if has_readable_text and has_fine_geo:
        return "anchor"

    # ── anchor (relaxed): multiple OCR texts even with coarse geo ──
    #    e.g. a photo with 3+ shop signs is highly informative
    if len(ocr) >= 3 and has_coarse_geo:
        return "anchor"

    # ── context: has scene info or some geo hints ──
    if has_clear_scene and (has_coarse_geo or has_fine_geo):
        return "context"
    if has_clear_scene and len(entities) >= 3:
        return "context"
    if has_readable_text and has_clear_scene:
        # Has some text but geo is too vague to be anchor
        return "context"

    # ── weak: not enough signal ──
    return "weak"


def build_anchor_evidence(parse_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract anchor evidence from a vision parse result.

    Returns None for non-anchor images.
    """
    ocr = parse_result.get("ocr_texts", [])
    geo = parse_result.get("geo_hints", {})

    if not ocr:
        return None

    # Filter OCR: remove watermarks, copyright notices, and generic text
    _NOISE_PATTERNS = {"google", "©", "copyright", "imagery", "map data",
                       "street view", "all rights reserved"}
    readable_signs = [
        t for t in ocr
        if len(t.strip()) >= 2
        and not any(noise in t.lower() for noise in _NOISE_PATTERNS)
    ]
    if not readable_signs:
        return None

    return {
        "readable_signs": readable_signs,
        "geo_level": geo.get("level", "unknown"),
        "geo_hint": geo.get("hint", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task routing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def assess_usable_tasks(parse_result: Dict[str, Any],
                        utility_class: str) -> List[str]:
    """Determine which task types this image can support, based on utility class."""
    ocr = parse_result.get("ocr_texts", [])
    scene = parse_result.get("scene_type", "")

    if utility_class == "anchor":
        # Anchor images can drive tasks that require the image
        tasks = ["place_lookup", "geocode_resolution"]
        # Also usable for POI-based tasks
        tasks.append("place_filter_rank")
        if scene in ("commercial_area", "street", "intersection"):
            tasks.append("place_discovery")
        if scene == "transport_hub":
            tasks.append("route_planning")
        return tasks

    if utility_class == "context":
        tasks = ["place_filter_rank"]
        if scene in ("commercial_area", "street", "intersection"):
            tasks.append("place_discovery")
        if scene == "transport_hub":
            tasks.append("route_planning")
        return tasks

    # weak — should have been filtered, but return empty just in case
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy / compat helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SCENE_TO_AREA_TYPE = {
    "commercial_area": "commercial",
    "transport_hub":   "transport_hub",
    "campus":          "university",
    "tourist_spot":    "tourist",
    "residential":     "residential",
}


def _infer_area_type(scene_type: str) -> Optional[str]:
    """Map vision-parsed scene_type to area_type (for GAEA sources lacking area_type)."""
    return _SCENE_TO_AREA_TYPE.get(scene_type)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = ClientFactory.for_stage(STAGE_VISION_PARSE, config)
    parser = VisionParser(llm)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(MANIFEST_PATH)
    if not manifest:
        print("[Module 2] No manifest found. Run module 1 first.")
        return PARSED_PATH

    results: List[Dict[str, Any]] = []
    stats = {"anchor": 0, "context": 0, "weak": 0}

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

        # ── classify ──
        utility_class = classify_image_utility(parse_result)
        stats[utility_class] += 1

        if utility_class == "weak":
            print(f"  -> filtered (weak)")
            continue

        # ── build fields ──
        usable_tasks = assess_usable_tasks(parse_result, utility_class)
        anchor_evidence = build_anchor_evidence(parse_result) if utility_class == "anchor" else None

        # Backfill area_type from scene_type if not already set (e.g. GAEA source)
        if not record.get("area_type"):
            record["area_type"] = _infer_area_type(parse_result.get("scene_type", ""))

        parsed_record = {
            **record,
            "vision_parse": parse_result,
            "image_utility_class": utility_class,
            "anchor_evidence": anchor_evidence,
            "usable_for_tasks": usable_tasks,
        }
        results.append(parsed_record)
        sign_preview = anchor_evidence["readable_signs"][:3] if anchor_evidence else []
        print(f"  -> {utility_class} | tasks={usable_tasks} | signs={sign_preview}")

    count = write_jsonl(PARSED_PATH, results)
    print(f"[Module 2] Done. {count} images kept "
          f"(anchor={stats['anchor']}, context={stats['context']}), "
          f"{stats['weak']} filtered (weak)")
    return PARSED_PATH


if __name__ == "__main__":
    main()
