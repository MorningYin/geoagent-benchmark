"""Module 3: Task Creator — prompt-driven creative task generation.

Pipeline v4: replaces v3's 994-line schema-driven builder with a single
LLM call per image. Claude generates the complete task including natural
language query, expected behavior chain, and verification rules.

Distribution steering is injected via prompt to guide toward underrepresented
coordinate values without hard constraints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from shared.config import PipelineConfig
from shared.llm_client import ClientFactory, STAGE_TASK_CREATE
from shared.maps_client import GoogleMapsClient
from shared.distribution import DistributionController
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = PIPELINE_ROOT / "data" / "02_parsed"
DATA_OUT = PIPELINE_ROOT / "data" / "03_tasks"
PARSED_PATH = DATA_IN / "parsed.jsonl"
OUTPUT_PATH = DATA_OUT / "tasks.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"

_MAX_RETRIES = 2

# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一个地理 Agent 基准测试的任务设计师。你的目标是基于一张真实街景照片和其附近的真实 POI 数据，设计一个测试 AI Agent 地理推理能力的任务。

## 设计原则

1. **场景自然** — 真实用户在这个地方真的会提出这种需求，不要生硬造作
2. **图像不可或缺** — 去掉图像后任务无法完成或完全变味。不要让图像只是摆设
3. **narrative 不能泄露位置** — scenario.narrative 中只能描述用户的处境和动机，**绝对不能提及具体地名、城市、地标名称**。让图像成为唯一的定位来源
4. **基于现实** — 所有提到的地点、约束都应能被附近的真实 POI 数据支撑
5. **有明确答案** — 给出标准答案和可检查的验证条件
6. **用户的话用自然口语** — 如果拍摄地点是中文环境就用中文，外语环境就用英文，像真人在对 Agent 说话
7. **不泄露答案** — 用户的话里不能直接包含答案（如地点名称），应让 Agent 通过图像+工具去发现

## API 参数规范（重要！）

在 expected_behavior.api_calls 中：
- **place_id** 参数必须写成占位符格式，如 `"<nearby_search 返回的第1个结果的 place_id>"`，**不要**用中文名称或编造的 ID
- **坐标参数**要用实际坐标值，如 `{"latitude": 39.9087, "longitude": 116.3975}`
- **reasoning** 中不要预设 API 返回结果，而是描述 Agent 的推理过程："调用 XX API → 根据返回结果筛选 → ..."
- 不要添加冗余的 API 调用。每一步都应有明确的增量价值

## 输出格式（严格 JSON，不要输出其他内容）

{"coordinates": {"task_type": "locate|search|route|plan|judge|adapt", "complexity": "simple|moderate|complex", "image_role": "primary|supporting|trigger"}, "scenario": {"narrative": "场景背景（不含地名）", "turns": [{"role": "user", "content": "用户说的话", "images": ["{{image_id}}"]}]}, "expected_behavior": {"reasoning": "推理过程", "api_calls": [{"method": "nearby_search|place_details|directions|geocoding|reverse_geocode|distance_matrix", "params": {}, "purpose": "原因"}], "write_actions": []}, "verification": {"must_pass": ["检查条件"], "answer_reference": {}}, "coordinate_reasoning": "坐标选择理由"}

## 任务类型说明

- **locate**: 从图像线索定位地方（读招牌、识地标）。image_role 通常是 primary。
- **search**: 按约束搜索筛选地点（找餐厅、药店等）。
- **route**: 规划路线或比较交通方式。时间约束增加难度。
- **plan**: 多点行程规划、时间预算编排。最复杂的类型。
- **judge**: 识别地理信息矛盾，拒绝不合理请求。Agent 不应盲从。
- **adapt**: 环境变化后调整方案。**必须** ≥2 个 turns（先建方案再引入变化）。

## 复杂度说明

- **simple**: 1-2步直接查询，单一约束
- **moderate**: 3-4步推理，多约束联合
- **complex**: 5+步，约束冲突需权衡，或多轮交互

## 图像角色说明

- **primary**: 图像提供关键解题信息（去掉图做不了）
- **supporting**: 图像提供辅助上下文（去掉图变难但能做）
- **trigger**: 图像触发任务（用户看到某物所以提问）"""


def _build_user_prompt(
    scene_description: str,
    ocr_texts: List[str],
    location: Dict[str, float],
    poi_context: str,
    steering: str,
    target: Dict[str, str],
    task_guidance: Dict[str, Any],
    image_id: str,
) -> str:
    """Assemble the user prompt with all context."""

    # Build task type descriptions from spec
    type_desc_lines = []
    for ttype, guidance in task_guidance.items():
        desc = guidance.get("description", "")
        examples = guidance.get("example_scenarios", [])
        examples_str = "；".join(examples[:2]) if examples else ""
        type_desc_lines.append(f"  - {ttype}: {desc}。例如：{examples_str}")
    type_descriptions = "\n".join(type_desc_lines)

    return f"""## 图像信息
{scene_description}

图中可读文字: {json.dumps(ocr_texts, ensure_ascii=False) if ocr_texts else "（无可读文字）"}

图像 ID: {image_id}

## 真实位置数据
坐标: ({location['lat']:.6f}, {location['lng']:.6f})

附近 POI:
{poi_context}

## 分布引导
{steering}
建议本次创作方向: task_type={target.get('task_type', 'any')}, complexity={target.get('complexity', 'any')}, image_role={target.get('image_role', 'any')}
（这是建议，如果图像素材更适合其他类型，可以偏离）

## 各任务类型参考
{type_descriptions}

请基于以上素材，创作一个高质量的地理任务。"""


# ═══════════════════════════════════════════════════════════════════════════════
# POI formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _format_poi_list(pois: List[Dict[str, Any]],
                     seed_lat: float, seed_lng: float,
                     maps: GoogleMapsClient) -> str:
    """Format nearby POIs into a compact text list for the LLM prompt."""
    if not pois:
        return "（附近未找到 POI）"

    lines = []
    for p in pois[:15]:  # cap at 15 to avoid overly long prompts
        name = p.get("displayName", {}).get("text", "未知")
        address = p.get("formattedAddress", "")
        rating = p.get("rating", "N/A")
        rating_count = p.get("userRatingCount", 0)
        primary_type = p.get("primaryType", "")

        # Compute distance from seed
        loc = p.get("location", {})
        plat = loc.get("latitude", seed_lat)
        plng = loc.get("longitude", seed_lng)
        dist = maps.haversine_distance(seed_lat, seed_lng, plat, plng)

        # Opening hours
        hours_info = ""
        cur_hours = p.get("currentOpeningHours", {})
        if cur_hours.get("openNow") is True:
            hours_info = " [营业中]"
        elif cur_hours.get("openNow") is False:
            hours_info = " [已关门]"

        lines.append(
            f"  - {name} ({primary_type}) | {dist:.0f}m | "
            f"评分 {rating}({rating_count}条){hours_info} | {address}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Structural validation
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_TASK_TYPES = {"locate", "search", "route", "plan", "judge", "adapt"}
_VALID_COMPLEXITY = {"simple", "moderate", "complex"}
_VALID_IMAGE_ROLE = {"primary", "supporting", "trigger"}


def _validate_structure(result: Dict[str, Any]) -> List[str]:
    """Lightweight structural validation. Returns list of error messages."""
    errors = []

    # coordinates
    coords = result.get("coordinates", {})
    if coords.get("task_type") not in _VALID_TASK_TYPES:
        errors.append(f"coordinates.task_type 必须是 {_VALID_TASK_TYPES} 之一")
    if coords.get("complexity") not in _VALID_COMPLEXITY:
        errors.append(f"coordinates.complexity 必须是 {_VALID_COMPLEXITY} 之一")
    if coords.get("image_role") not in _VALID_IMAGE_ROLE:
        errors.append(f"coordinates.image_role 必须是 {_VALID_IMAGE_ROLE} 之一")

    # scenario
    scenario = result.get("scenario", {})
    turns = scenario.get("turns", [])
    if not turns:
        errors.append("scenario.turns 不能为空")
    if not scenario.get("narrative"):
        errors.append("scenario.narrative 不能为空")

    # expected_behavior
    behavior = result.get("expected_behavior", {})
    if not behavior.get("reasoning"):
        errors.append("expected_behavior.reasoning 不能为空")

    # verification
    verification = result.get("verification", {})
    must_pass = verification.get("must_pass", [])
    if not must_pass:
        errors.append("verification.must_pass 至少需要 1 条")

    # adapt tasks must have >= 2 turns
    if coords.get("task_type") == "adapt" and len(turns) < 2:
        errors.append("adapt 类型任务必须有 ≥2 个 turns（先建方案再引入变化）")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Record assembly
# ═══════════════════════════════════════════════════════════════════════════════

def _assemble_record(
    llm_output: Dict[str, Any],
    parsed_record: Dict[str, Any],
    pois: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the complete task record from LLM output + source metadata."""

    # Simplify POI data for storage
    poi_summaries = []
    for p in pois[:10]:
        poi_summaries.append({
            "name": p.get("displayName", {}).get("text", ""),
            "place_id": p.get("id", ""),
            "address": p.get("formattedAddress", ""),
            "type": p.get("primaryType", ""),
            "rating": p.get("rating"),
            "location": p.get("location", {}),
        })

    return {
        "coordinates": llm_output["coordinates"],
        "scenario": llm_output["scenario"],
        "expected_behavior": llm_output["expected_behavior"],
        "verification": llm_output["verification"],
        "coordinate_reasoning": llm_output.get("coordinate_reasoning", ""),
        "provenance": {
            "image_id": parsed_record.get("image_id", ""),
            "image_path": parsed_record.get("image_path", ""),
            "location": {
                "lat": parsed_record.get("seed_lat", 0.0),
                "lng": parsed_record.get("seed_lng", 0.0),
            },
            "nearby_pois": poi_summaries,
            "scene_description": parsed_record.get("scene_description", ""),
            "generated_by": "claude-sonnet-4-6",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()

    # Load task spec
    spec_path = PIPELINE_ROOT / "task_spec.yaml"
    with open(spec_path, encoding="utf-8") as f:
        task_spec = yaml.safe_load(f)

    # Initialize components
    parsed_records = read_jsonl(PARSED_PATH)
    if not parsed_records:
        print("[Module 3] No parsed records found. Run module 2 first.")
        return OUTPUT_PATH

    maps = GoogleMapsClient(config)
    llm = ClientFactory.for_stage(STAGE_TASK_CREATE, config)
    dist = DistributionController(task_spec)
    task_guidance = task_spec.get("task_guidance", {})

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = []

    for i, rec in enumerate(parsed_records, 1):
        image_id = rec.get("image_id", "unknown")
        print(f"[Module 3] Creating task {i}/{len(parsed_records)}: {image_id}")

        try:
            # 1. Fetch nearby POIs
            seed_lat = rec.get("seed_lat", 0.0)
            seed_lng = rec.get("seed_lng", 0.0)
            pois = maps.nearby_search(seed_lat, seed_lng, radius_m=1000)
            poi_context = _format_poi_list(pois, seed_lat, seed_lng, maps)

            # 2. Build prompt
            user_prompt = _build_user_prompt(
                scene_description=rec.get("scene_description", ""),
                ocr_texts=rec.get("ocr_texts", []),
                location={"lat": seed_lat, "lng": seed_lng},
                poi_context=poi_context,
                steering=dist.steering_prompt(),
                target=dist.suggest_target(),
                task_guidance=task_guidance,
                image_id=image_id,
            )

            # 3. Call LLM with retry on validation failure
            result = None
            last_errors: List[str] = []

            for attempt in range(_MAX_RETRIES + 1):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                # On retry, append error feedback
                if last_errors:
                    feedback = (
                        "上次输出有以下问题，请修正:\n"
                        + "\n".join(f"- {e}" for e in last_errors)
                        + "\n\n请重新生成完整的 JSON 输出。"
                    )
                    messages.append({"role": "assistant", "content": "我理解了，让我修正。"})
                    messages.append({"role": "user", "content": feedback})

                raw_result = llm.json_completion(messages, temperature=0.9)

                validation_errors = _validate_structure(raw_result)
                if not validation_errors:
                    result = raw_result
                    break
                else:
                    last_errors = validation_errors
                    print(f"  -> attempt {attempt + 1} validation failed: {validation_errors}")

            if result is None:
                raise ValueError(
                    f"Failed after {_MAX_RETRIES + 1} attempts. "
                    f"Last errors: {last_errors}"
                )

            # 4. Assemble record
            task = _assemble_record(result, rec, pois)
            dist.record(task["coordinates"])
            tasks.append(task)

            coords = task["coordinates"]
            print(
                f"  -> {coords['task_type']} / {coords['complexity']} / "
                f"{coords['image_role']}"
            )

        except Exception as e:
            print(f"  -> ERROR: {e}")
            append_error(ERRORS_PATH, {
                "image_id": image_id, "error": str(e),
            })

    count = write_jsonl(OUTPUT_PATH, tasks)
    print(f"\n[Module 3] Done. {count} tasks created.")
    print(f"[Module 3] Distribution:\n{dist.summary()}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
