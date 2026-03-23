"""Module 2: Scene Describer — extract free-text scene description + OCR from images.

Pipeline v4: simplified from v3's structured parsing + classification.
No longer classifies images into anchor/context/weak; Module 4 (Judge)
decides if an image is usable. All images pass through to Module 3.

Output per image:
  - scene_description  — free-text scene description (3-5 sentences)
  - ocr_texts          — list of readable text extracted from the image
"""

from __future__ import annotations

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

VISION_PROMPT = """请仔细观察这张街景照片，提供以下信息。

返回严格的 JSON 格式：
{
  "scene_description": "详细描述你看到的场景（3-5句），包括建筑类型、道路特征、可见地标、商铺类型、天气、大致时间段、以及任何能帮助识别地理位置的线索。",
  "ocr_texts": ["图中所有可读文字的列表（招牌、路牌、门牌号、标识等）"]
}

注意：
1. scene_description 要丰富且具体，着重描述有地理意义的信息
2. ocr_texts 只写你确信能看到的文字，排除水印和版权标注
3. 如果图中没有可读文字，ocr_texts 返回空数组"""


# Watermark / copyright noise patterns to filter from OCR
_NOISE_PATTERNS = frozenset({
    "google", "©", "copyright", "imagery", "map data",
    "street view", "all rights reserved",
})

# Keywords in scene_description that indicate a useless image
_INVALID_IMAGE_KEYWORDS = [
    "错误页面", "error page", "没有可用", "不可用", "无法加载",
    "no imagery", "not available", "街景错误", "提示信息",
]

# Keywords indicating indoor/generic images with low geo value
_LOW_GEO_VALUE_KEYWORDS = [
    "室内展览厅", "室内大厅", "室内门厅", "室内楼梯",
    "完全空旷", "完全空置",
]


def _filter_ocr(raw_ocr: List[str]) -> List[str]:
    """Remove watermarks, copyright notices, and very short noise from OCR."""
    return [
        t for t in raw_ocr
        if len(t.strip()) >= 2
        and not any(n in t.lower() for n in _NOISE_PATTERNS)
    ]


class SceneDescriber:
    """Describe street view images using a vision LLM."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def describe(self, image_path: str | Path) -> Dict[str, Any]:
        """Return {"scene_description": str, "ocr_texts": list[str]}."""
        result = self.llm_client.vision_json_completion(
            image_path, VISION_PROMPT, temperature=0.3
        )
        # Ensure expected keys exist
        result.setdefault("scene_description", "")
        result.setdefault("ocr_texts", [])
        # Filter OCR noise
        result["ocr_texts"] = _filter_ocr(result["ocr_texts"])
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = ClientFactory.for_stage(STAGE_VISION_PARSE, config)
    describer = SceneDescriber(llm)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(MANIFEST_PATH)
    if not manifest:
        print("[Module 2] No manifest found. Run module 1 first.")
        return PARSED_PATH

    results: List[Dict[str, Any]] = []

    for i, record in enumerate(manifest, 1):
        image_path = record.get("image_path", "")
        image_id = record.get("image_id", "")
        print(f"[Module 2] Describing image {i}/{len(manifest)}: {image_id}")

        if not Path(image_path).exists():
            append_error(ERRORS_PATH, {
                "image_id": image_id, "error": "image file not found",
            })
            continue

        try:
            desc = describer.describe(image_path)
        except Exception as e:
            append_error(ERRORS_PATH, {
                "image_id": image_id, "error": str(e),
            })
            continue

        scene_text = desc["scene_description"].lower()

        # Filter: Google Street View error pages
        if any(kw in scene_text for kw in _INVALID_IMAGE_KEYWORDS):
            print(f"  -> FILTERED (invalid image / error page)")
            continue

        # Filter: indoor/generic images with no geographic value
        if any(kw in desc["scene_description"] for kw in _LOW_GEO_VALUE_KEYWORDS):
            if not desc["ocr_texts"]:  # no OCR text to compensate
                print(f"  -> FILTERED (low geo value indoor scene)")
                continue

        parsed_record = {
            **record,
            "scene_description": desc["scene_description"],
            "ocr_texts": desc["ocr_texts"],
        }
        results.append(parsed_record)

        ocr_preview = desc["ocr_texts"][:3]
        desc_preview = desc["scene_description"][:80]
        print(f"  -> OCR={ocr_preview} | {desc_preview}...")

    count = write_jsonl(PARSED_PATH, results)
    print(f"[Module 2] Done. {count} images described.")
    return PARSED_PATH


if __name__ == "__main__":
    main()
