"""Module 3: Task Builder — construct tasks from real images + POI data.

Three construction paths based on image_utility_class:

  anchor images  →  _build_anchor_task()
                    place_lookup / geocode_resolution (image is primary input)
                    Falls back to LLM POI-based path on failure.

  context images →  _llm_build_task()  (LLM picks scenario + fills dimensions)
                    place_filter_rank / place_discovery
                    Falls back to _rule_based_build_task().

Diversity enforcement: task_type distribution is tracked across the batch and
injected into the LLM Step 1 prompt to avoid monotony.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.config import PipelineConfig
from shared.maps_client import GoogleMapsClient
from shared.llm_client import BaseLLMClient, ClientFactory, STAGE_TASK_DESIGN
from shared.schema_loader import SchemaLoader
from shared.coherence import ClassificationEngine, CoherenceEngine
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "02_parsed"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "03_tasks"
PARSED_PATH = DATA_IN / "parsed.jsonl"
TASKS_PATH = DATA_OUT / "tasks.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.json"

# Task types supported per image utility class
_ANCHOR_TASK_TYPES = {"place_lookup", "geocode_resolution", "place_filter_rank"}
_CONTEXT_TASK_TYPES = {"place_filter_rank", "place_discovery"}
_LLM_SUPPORTED_TASK_TYPES = {"place_filter_rank", "place_discovery"}
_MAX_LLM_RETRIES = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

_POI_TYPE_TO_CATEGORY = {
    "restaurant": "restaurant", "cafe": "cafe", "shopping_mall": "shopping_mall",
    "supermarket": "supermarket", "pharmacy": "pharmacy",
    "convenience_store": "convenience_store", "park": "park", "museum": "museum",
    "tourist_attraction": "tourist_attraction", "transit_station": "transit_station",
    "bus_station": "bus_station", "train_station": "train_station",
    "university": "university", "library": "library", "book_store": "book_store",
    "hospital": "hospital", "hotel": "hotel", "gas_station": "gas_station",
    "bank": "bank", "post_office": "post_office",
}


def _infer_category(poi: Dict[str, Any]) -> Optional[str]:
    primary = poi.get("primaryType", "")
    if primary in _POI_TYPE_TO_CATEGORY:
        return _POI_TYPE_TO_CATEGORY[primary]
    for t in poi.get("types", []):
        if t in _POI_TYPE_TO_CATEGORY:
            return _POI_TYPE_TO_CATEGORY[t]
    return None


def _enrich_pois(pois: List[Dict[str, Any]], seed_lat: float, seed_lng: float,
                 maps: GoogleMapsClient) -> List[Dict[str, Any]]:
    """Add _distance_m and _category to each POI."""
    for poi in pois:
        loc = poi.get("location", {})
        poi["_distance_m"] = maps.haversine_distance(
            seed_lat, seed_lng,
            loc.get("latitude", seed_lat), loc.get("longitude", seed_lng))
        poi["_category"] = _infer_category(poi)
    return pois


def _poi_summary_list(pois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact POI list for LLM prompts (low token cost)."""
    result = []
    for p in pois:
        cat = p.get("_category")
        if cat is None:
            continue
        result.append({
            "name": p.get("displayName", {}).get("text", ""),
            "place_id": p.get("id", ""),
            "category": cat,
            "distance_m": round(p.get("_distance_m", 0)),
            "rating": p.get("rating"),
            "rating_count": p.get("userRatingCount"),
        })
    return result


def _build_information_gap_plan(parsed: Dict[str, Any],
                                task_type: str,
                                task_dimensions: Dict[str, Any]) -> Dict[str, Any]:
    """Determine what info the image provides vs what must come from query/GPS."""
    vp = parsed.get("vision_parse", {})
    ocr = vp.get("ocr_texts", [])
    scene = vp.get("scene_type", "")
    geo = vp.get("geo_hints", {})
    utility = parsed.get("image_utility_class", "context")

    image_provides = []
    query_must_state = []
    gps_provides = []
    must_not_leak = []

    # ── image_provides ──
    if utility == "anchor":
        image_provides.append("primary_location_clue")
        if ocr:
            image_provides.append("visible_text_clues")
            image_provides.append(f"readable_signs:{','.join(ocr[:3])}")
    if geo.get("level") in ("street", "building", "district"):
        image_provides.append("approximate_location")
    elif geo.get("level") == "city":
        image_provides.append("city_level_location")
    if scene:
        image_provides.append(f"scene_type:{scene}")
    entities = vp.get("visible_entities", [])
    if len(entities) >= 2:
        image_provides.append("environmental_context")

    # ── gps_provides ──
    gps_provides.append("exact_coordinates")
    if task_type in ("place_filter_rank", "place_discovery"):
        gps_provides.append("search_center")

    # ── query_must_state ──
    if task_type in ("place_lookup", "geocode_resolution"):
        query_must_state.append("what_to_identify_from_image")
    category = task_dimensions.get("place_category_canonical", "")
    if category:
        query_must_state.append(f"need_category:{category}")
    ranking = task_dimensions.get("ranking_objective", "")
    if ranking:
        query_must_state.append(f"ranking_preference:{ranking}")

    # ── must_not_leak ──
    must_not_leak.append("answer_place_name")
    if task_type in ("place_filter_rank", "place_discovery"):
        must_not_leak.append("exact_distances")
        if task_dimensions.get("min_rating"):
            must_not_leak.append("min_rating_threshold")
    if task_type == "geocode_resolution":
        must_not_leak.append("full_address")

    return {
        "image_provides": image_provides,
        "query_must_state": query_must_state,
        "gps_provides": gps_provides,
        "must_not_leak_in_query": must_not_leak,
    }


def _build_answer_data(answer_poi: Dict[str, Any],
                       candidate_pois: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build answer_data dict from a chosen answer POI and candidate list."""
    loc = answer_poi.get("location", {})
    return {
        "expected_place_name": answer_poi.get("displayName", {}).get("text", ""),
        "expected_place_id": answer_poi.get("id", ""),
        "expected_address": answer_poi.get("formattedAddress", ""),
        "expected_lat": loc.get("latitude"),
        "expected_lng": loc.get("longitude"),
        "expected_rating": answer_poi.get("rating"),
        "expected_distance_m": round(answer_poi.get("_distance_m", 0)),
        "candidate_pois": [
            {
                "name": p.get("displayName", {}).get("text", ""),
                "place_id": p.get("id", ""),
                "distance_m": round(p.get("_distance_m", 0)),
                "rating": p.get("rating"),
            }
            for p in candidate_pois
        ],
    }


def _build_global_context_for_scenario(parsed: Dict[str, Any],
                                        scenario_id: str,
                                        schema_loader: SchemaLoader) -> Dict[str, Any]:
    """Build a global_context suitable for anchor tasks by sampling from the scenario frame."""
    vp = parsed.get("vision_parse", {})
    time_hint = vp.get("time_hint", "unknown")
    weather_hint = vp.get("weather_hint", "unknown")

    now = datetime.now(timezone.utc)
    offset = random.randint(-7, 7)
    hour = random.randint(8, 21)
    request_time = (now + timedelta(days=offset)).replace(
        hour=hour, minute=random.choice([0, 15, 30, 45]), second=0, microsecond=0)

    weather_map = {"clear": "clear", "cloudy": None, "rain": "rain",
                   "snow": "snow", "foggy": "foggy", "unknown": None}

    # Start from scenario frame defaults
    try:
        frame = schema_loader.scenario_frame(scenario_id)
        defaults = frame.get("default_context", {})
        overrides = frame.get("likely_overrides", {})
    except KeyError:
        defaults = {}
        overrides = {}

    gc = {
        "request_time": request_time.isoformat(),
        "time_semantics": "now" if time_hint == "daytime" else "tonight" if time_hint == "nighttime" else None,
        "weather_context": weather_map.get(weather_hint),
        "user_party_composition": defaults.get("user_party_composition",
                                                random.choice([None, "solo", "couple", "friends"])),
        "mobility_context": defaults.get("mobility_context",
                                          random.choice([None, "normal"])),
        "urgency_level": defaults.get("urgency_level", "low"),
        "familiarity_with_area": defaults.get("familiarity_with_area", "uncertain"),
        "trip_state": defaults.get("trip_state",
                                    random.choice(["touring", "routine_errand"])),
        "language_preference": "zh-CN",
    }

    # Apply likely_overrides with some probability
    for field, candidates in overrides.items():
        if field in gc and candidates and random.random() < 0.5:
            gc[field] = random.choice(candidates)

    return gc


def _assemble_task_record(parsed: Dict[str, Any], task_type: str,
                          global_context: Dict[str, Any],
                          task_dimensions: Dict[str, Any],
                          answer_data: Dict[str, Any],
                          nearby_poi_count: int,
                          scenario_frame_id: Optional[str] = None,
                          construction_meta: Optional[Dict[str, Any]] = None,
                          ) -> Dict[str, Any]:
    """Assemble the final task record (shared by all construction paths)."""
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    info_gap = _build_information_gap_plan(parsed, task_type, task_dimensions)
    device_context = {
        "gps": {"lat": seed_lat, "lng": seed_lng},
        "current_time": global_context["request_time"],
        "device_language": global_context.get("language_preference", "zh-CN"),
    }

    record = {
        "image_id": parsed["image_id"],
        "image_path": parsed.get("image_path", ""),
        "task_type": task_type,
        "scenario_frame_id": scenario_frame_id,
        "image_utility_class": parsed.get("image_utility_class", "context"),
        "anchor_evidence": parsed.get("anchor_evidence"),
        "global_context": global_context,
        "task_dimensions": task_dimensions,
        "vision_parse": parsed.get("vision_parse"),
        "real_world_context": {
            "seed": {
                "city": parsed.get("seed_city"),
                "label": parsed.get("seed_label"),
                "area_type": parsed.get("area_type"),
            },
            "nearby_poi_count": nearby_poi_count,
        },
        "information_gap_plan": info_gap,
        "answer_data": answer_data,
        "device_context": device_context,
    }
    if construction_meta:
        record["construction_metadata"] = construction_meta
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor task construction (place_lookup / geocode_resolution)
# ═══════════════════════════════════════════════════════════════════════════════

def _fuzzy_match_poi(ocr_texts: List[str],
                     pois: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Try to match OCR texts against nearby POI names. Returns best match or None."""
    for poi in pois:
        poi_name = poi.get("displayName", {}).get("text", "").strip()
        if not poi_name:
            continue
        for text in ocr_texts:
            text = text.strip()
            if not text or len(text) < 2:
                continue
            # Exact substring match (either direction)
            if text in poi_name or poi_name in text:
                return poi
    return None


def _build_place_lookup_task(parsed: Dict[str, Any],
                              nearby_pois: List[Dict[str, Any]],
                              maps: GoogleMapsClient,
                              schema_loader: SchemaLoader) -> Dict[str, Any]:
    """Build a place_lookup task where the image is the primary query.

    The agent must identify the place shown in the image from its visible text/signs.
    """
    anchor = parsed.get("anchor_evidence", {})
    ocr_texts = anchor.get("readable_signs", [])
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    _enrich_pois(nearby_pois, seed_lat, seed_lng, maps)

    # Try to find a POI that matches one of the OCR texts
    matched_poi = _fuzzy_match_poi(ocr_texts, nearby_pois)
    if matched_poi is None:
        raise ValueError(f"No POI matches OCR texts {ocr_texts}")

    # task_dimensions for place_lookup
    task_dimensions = {
        "query_text": matched_poi.get("displayName", {}).get("text", ""),
        "query_source": "image_ocr",
        "search_region_type": "center_radius",
        "search_region_value": {
            "center_lat": seed_lat,
            "center_lng": seed_lng,
            "radius_m": 500,
        },
        "disambiguation_needed": False,
    }

    answer_data = {
        "expected_place_name": matched_poi.get("displayName", {}).get("text", ""),
        "expected_place_id": matched_poi.get("id", ""),
        "expected_address": matched_poi.get("formattedAddress", ""),
        "expected_lat": matched_poi.get("location", {}).get("latitude"),
        "expected_lng": matched_poi.get("location", {}).get("longitude"),
        "expected_distance_m": round(matched_poi.get("_distance_m", 0)),
        "matched_ocr_text": next(
            (t for t in ocr_texts
             if t in matched_poi.get("displayName", {}).get("text", "")
             or matched_poi.get("displayName", {}).get("text", "") in t),
            ocr_texts[0] if ocr_texts else ""
        ),
    }

    gc = _build_global_context_for_scenario(parsed, "photo_based_location_guess", schema_loader)

    return _assemble_task_record(
        parsed=parsed,
        task_type="place_lookup",
        global_context=gc,
        task_dimensions=task_dimensions,
        answer_data=answer_data,
        nearby_poi_count=len(nearby_pois),
        scenario_frame_id="photo_based_location_guess",
        construction_meta={
            "method": "anchor_place_lookup",
            "matched_sign": answer_data["matched_ocr_text"],
        },
    )


def _build_geocode_resolution_task(parsed: Dict[str, Any],
                                    nearby_pois: List[Dict[str, Any]],
                                    maps: GoogleMapsClient,
                                    schema_loader: SchemaLoader) -> Dict[str, Any]:
    """Build a geocode_resolution task where the image provides location clues.

    The agent must determine the address / location from street signs and landmarks.
    """
    anchor = parsed.get("anchor_evidence", {})
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    _enrich_pois(nearby_pois, seed_lat, seed_lng, maps)

    # Reverse geocode for the ground-truth address
    geo_result = maps.reverse_geocode(seed_lat, seed_lng)
    if not geo_result.get("formatted_address"):
        raise ValueError("Reverse geocode returned empty address")

    task_dimensions = {
        "query_text": anchor.get("geo_hint", ""),
        "query_source": "image_geo_clue",
        "target_resolution_level": anchor.get("geo_level", "street"),
    }

    answer_data = {
        "expected_address": geo_result["formatted_address"],
        "expected_place_id": geo_result.get("place_id", ""),
        "address_components": geo_result.get("address_components", []),
        "expected_lat": seed_lat,
        "expected_lng": seed_lng,
        "visible_clues_used": anchor.get("readable_signs", []),
    }

    gc = _build_global_context_for_scenario(parsed, "photo_based_location_guess", schema_loader)

    return _assemble_task_record(
        parsed=parsed,
        task_type="geocode_resolution",
        global_context=gc,
        task_dimensions=task_dimensions,
        answer_data=answer_data,
        nearby_poi_count=len(nearby_pois),
        scenario_frame_id="photo_based_location_guess",
        construction_meta={
            "method": "anchor_geocode_resolution",
            "geo_clues": anchor.get("readable_signs", [])[:3],
        },
    )


def _build_anchor_task(parsed: Dict[str, Any],
                       nearby_pois: List[Dict[str, Any]],
                       maps: GoogleMapsClient,
                       schema_loader: SchemaLoader,
                       task_type_counts: Dict[str, int]) -> Dict[str, Any]:
    """Route anchor images to the most appropriate task type.

    Prefers place_lookup (if OCR matches a POI), then geocode_resolution,
    with diversity balancing.
    """
    ocr_texts = parsed.get("anchor_evidence", {}).get("readable_signs", [])

    # Check if place_lookup is viable (OCR matches a nearby POI)
    _enrich_pois(nearby_pois, parsed["seed_lat"], parsed["seed_lng"], maps)
    matched = _fuzzy_match_poi(ocr_texts, nearby_pois)

    # Decide which anchor task to attempt based on viability and diversity
    lookup_count = task_type_counts.get("place_lookup", 0)
    geocode_count = task_type_counts.get("geocode_resolution", 0)

    errors = []

    if matched:
        # place_lookup is viable — prefer it if under-represented
        if lookup_count <= geocode_count:
            try:
                return _build_place_lookup_task(parsed, nearby_pois, maps, schema_loader)
            except Exception as e:
                errors.append(f"place_lookup: {e}")

        # Try geocode_resolution
        try:
            return _build_geocode_resolution_task(parsed, nearby_pois, maps, schema_loader)
        except Exception as e:
            errors.append(f"geocode_resolution: {e}")

        # Fallback to place_lookup if geocode failed
        try:
            return _build_place_lookup_task(parsed, nearby_pois, maps, schema_loader)
        except Exception as e:
            errors.append(f"place_lookup fallback: {e}")
    else:
        # No POI match — try geocode_resolution
        try:
            return _build_geocode_resolution_task(parsed, nearby_pois, maps, schema_loader)
        except Exception as e:
            errors.append(f"geocode_resolution: {e}")

    raise ValueError(f"All anchor task paths failed: {'; '.join(errors)}")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-guided construction (place_filter_rank / place_discovery)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_step1_prompt(parsed: Dict[str, Any],
                        poi_summary: List[Dict[str, Any]],
                        schema_loader: SchemaLoader,
                        task_type_counts: Dict[str, int]) -> List[Dict[str, str]]:
    """Build prompt for Step 1: pick scenario_frame + task_type."""
    vp = parsed.get("vision_parse", {})
    cat_counts = Counter(p["category"] for p in poi_summary)

    # Compact frame list for prompt
    frames_info = []
    for fr in schema_loader.scenario_frames():
        compatible = [t for t in fr["compatible_task_types"] if t in _LLM_SUPPORTED_TASK_TYPES]
        if not compatible:
            continue
        frames_info.append({
            "scenario_id": fr["scenario_id"],
            "default_context": fr["default_context"],
            "compatible_task_types": compatible,
        })

    # Diversity hint
    diversity_hint = ""
    if task_type_counts:
        diversity_hint = f"""
## 多样性要求
当前已生成任务类型分布: {json.dumps(task_type_counts, ensure_ascii=False)}
请优先选择数量较少的类型，避免全部集中在同一类型。特别是如果 place_filter_rank 已经较多，请尝试 place_discovery。"""

    system = "你是一个地理智能体基准测试的任务设计师。你需要根据图片场景和附近 POI 分布，选择最合适的场景框架和任务类型。"

    user = f"""## 图片信息
- 场景类型: {vp.get('scene_type', 'unknown')}
- 地理线索: {vp.get('geo_hints', {}).get('hint', 'unknown')}
- 时间: {vp.get('time_hint', 'unknown')}
- 天气: {vp.get('weather_hint', 'unknown')}
- 可见文字: {vp.get('ocr_texts', [])}
- 地点: {parsed.get('seed_label', 'unknown')}

## 附近 POI 类别分布
{json.dumps(dict(cat_counts), ensure_ascii=False)}

## 可选场景框架
{json.dumps(frames_info, ensure_ascii=False, indent=2)}

## 任务类型说明
- place_filter_rank: 多约束筛选+排序，需要 ranking_objective，至少 2 个筛选维度，适合"找最近的/评分最高的某类店"
- place_discovery: 简单发现，最多 1 个筛选维度，无排序，适合"附近有没有某类店"
{diversity_hint}

## 要求
1. 从上面的场景框架中选择一个最贴合图片场景的 scenario_id
2. 从该框架的 compatible_task_types 中选择一个任务类型
3. 选择应该合理：图片场景、地点、POI 分布应该支持你选的框架和任务
4. 尽量选择有创意的、不千篇一律的场景

返回 JSON:
{{"scenario_frame_id": "xxx", "task_type": "place_filter_rank 或 place_discovery", "reasoning": "一句话解释"}}"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_step1(result: Dict[str, Any], schema_loader: SchemaLoader) -> List[str]:
    """Validate Step 1 output. Returns list of error strings (empty = valid)."""
    errors = []
    frame_id = result.get("scenario_frame_id", "")
    task_type = result.get("task_type", "")

    try:
        frame = schema_loader.scenario_frame(frame_id)
    except KeyError:
        errors.append(f"scenario_frame_id '{frame_id}' 不存在")
        return errors

    if task_type not in _LLM_SUPPORTED_TASK_TYPES:
        errors.append(f"task_type '{task_type}' 不在支持列表 {_LLM_SUPPORTED_TASK_TYPES}")
    elif task_type not in frame.get("compatible_task_types", []):
        errors.append(f"task_type '{task_type}' 不在 {frame_id} 的 compatible_task_types 中")

    return errors


def _build_step2_prompt(step1: Dict[str, Any], parsed: Dict[str, Any],
                        poi_summary: List[Dict[str, Any]],
                        schema_loader: SchemaLoader,
                        violations: Optional[List[str]] = None,
                        ) -> List[Dict[str, str]]:
    """Build prompt for Step 2: fill global_context + task_dimensions."""
    frame = schema_loader.scenario_frame(step1["scenario_frame_id"])
    task_type = step1["task_type"]
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    # Build dimension schema description
    dims = schema_loader.input_dimensions(task_type)
    dim_desc = []
    for d in dims:
        vr = schema_loader.get_resolved_value_range(d) or ""
        constraint = d.get("constraint_strength", "")
        required = d.get("required", False)
        dim_desc.append(f"- {d['name']}: {d.get('description','')} | 值域: {vr} | 约束强度: {constraint} | 必填: {required}")

    # Global context schema
    gc_desc = []
    for field in schema_loader.global_context_fields():
        vr = field.get("value_range", "")
        gc_desc.append(f"- {field['name']}: {vr}")

    # Available categories from POIs
    available_cats = sorted(set(p["category"] for p in poi_summary))

    system = "你是一个地理智能体基准测试的任务设计师。你需要填充任务的 global_context 和 task_dimensions。"

    task_rules = ""
    if task_type == "place_filter_rank":
        task_rules = """## place_filter_rank 规则（必须遵守）
- ranking_objective 必须填写（nearest/highest_rated/best_value 等）
- needs_candidate_comparison 必须为 true
- 至少 2 个筛选维度不为 null（distance_limit_m, min_rating, must_be_open_now, price_level_preference 等）
- place_category_canonical 必须从下面的"可用类别"中选择"""
    elif task_type == "place_discovery":
        task_rules = """## place_discovery 规则（必须遵守）
- ranking_objective 必须为 null
- needs_candidate_comparison 必须为 false
- 最多 1 个筛选维度不为 null
- place_category_canonical 必须从下面的"可用类别"中选择"""

    retry_section = ""
    if violations:
        retry_section = f"""
## ⚠️ 上次生成存在以下问题，请修正：
{chr(10).join(f'- {v}' for v in violations)}
"""

    user = f"""## 选定场景
- scenario_frame: {step1['scenario_frame_id']}
- task_type: {task_type}
- 场景默认上下文: {json.dumps(frame.get('default_context', {}), ensure_ascii=False)}
- 高概率覆盖值: {json.dumps(frame.get('likely_overrides', {}), ensure_ascii=False)}
- 低概率值（避免使用）: {json.dumps(frame.get('unlikely_values', {}), ensure_ascii=False)}

## 图片信息
- 场景类型: {parsed.get('vision_parse', {}).get('scene_type', 'unknown')}
- 地点: {parsed.get('seed_label', 'unknown')}
- 坐标: ({seed_lat}, {seed_lng})

{task_rules}

## global_context 字段（请逐一填写，从场景默认值出发）
{chr(10).join(gc_desc)}

## task_dimensions 字段（请逐一填写）
{chr(10).join(dim_desc)}

## 可用 POI 类别（place_category_canonical 只能从中选择）
{available_cats}

## 附近 POI 列表（answer_poi_place_id 只能从中选择）
{json.dumps(poi_summary, ensure_ascii=False, indent=2)}

## 坐标信息
search_region_value 应使用: {{"center_lat": {seed_lat}, "center_lng": {seed_lng}, "radius_m": 你设定的距离限制}}
{retry_section}
## 要求
1. global_context 以场景默认值为基础，合理覆盖，**不要使用 unlikely_values 中的值**
2. task_dimensions 每个字段都要填（不需要的填 null），**数值必须在值域范围内**
3. 从 POI 列表中选一个作为答案：
   - 如果 ranking_objective=nearest，答案必须是满足所有筛选条件的最近 POI
   - 如果 ranking_objective=highest_rated，答案必须是满足条件的最高评分 POI
4. distance_limit_m 要能包含至少 2 个同类别 POI（看 POI 列表的实际距离）
5. min_rating 要设置在合理范围，确保答案 POI 的评分 >= min_rating
6. request_time 使用 ISO-8601 格式

返回 JSON:
{{"global_context": {{...}}, "task_dimensions": {{...}}, "answer_poi_place_id": "xxx", "reasoning": "一句话"}}"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_llm_output(task_type: str, scenario_frame_id: str,
                         global_context: Dict[str, Any],
                         task_dimensions: Dict[str, Any],
                         answer_poi: Optional[Dict[str, Any]],
                         poi_summary: List[Dict[str, Any]],
                         schema_loader: SchemaLoader) -> Tuple[bool, List[str]]:
    """Step 3: run all programmatic validators. Returns (is_valid, error_list)."""
    errors: List[str] = []

    # 1. Classification rules
    cls_engine = ClassificationEngine(schema_loader)
    cls_ok, cls_violations = cls_engine.validate(task_type, task_dimensions)
    for v in cls_violations:
        errors.append(f"[分类规则] {v.message}")

    # 2. Coherence rules
    try:
        frame = schema_loader.scenario_frame(scenario_frame_id)
    except KeyError:
        frame = None
    coh_engine = CoherenceEngine(schema_loader, cls_engine)
    coh_result = coh_engine.validate(task_type, frame, global_context, task_dimensions)
    for v in coh_result.violations:
        if v.severity == "hard":
            errors.append(f"[一致性规则·硬] {v.rule_id}: {v.message}")
        else:
            errors.append(f"[一致性规则·软] {v.rule_id}: {v.message}")

    # 3. Answer POI consistency
    if answer_poi is None:
        errors.append("[答案校验] answer_poi_place_id 在 POI 列表中不存在")
    else:
        distance_limit = task_dimensions.get("distance_limit_m")
        if distance_limit and answer_poi.get("_distance_m", 0) > distance_limit:
            errors.append(f"[答案校验] 答案 POI 距离 {answer_poi['_distance_m']:.0f}m 超过 distance_limit_m={distance_limit}")

        min_rating = task_dimensions.get("min_rating")
        if min_rating and answer_poi.get("rating") is not None:
            if answer_poi["rating"] < min_rating:
                errors.append(f"[答案校验] 答案 POI 评分 {answer_poi['rating']} 低于 min_rating={min_rating}")

        # Verify ranking objective correctness
        ranking = task_dimensions.get("ranking_objective")
        category = task_dimensions.get("place_category_canonical")
        if ranking and category and answer_poi:
            qualifying = [p for p in poi_summary
                          if p["category"] == category
                          and (not distance_limit or p["distance_m"] <= distance_limit)
                          and (not min_rating or (p.get("rating") or 0) >= min_rating)]
            if qualifying:
                if ranking == "nearest":
                    best = min(qualifying, key=lambda p: p["distance_m"])
                    if best["place_id"] != answer_poi.get("id", ""):
                        errors.append(f"[答案校验] ranking=nearest 但答案不是最近的合格 POI（最近的是 {best['name']}）")
                elif ranking == "highest_rated":
                    best = max(qualifying, key=lambda p: p.get("rating") or 0)
                    if best["place_id"] != answer_poi.get("id", ""):
                        errors.append(f"[答案校验] ranking=highest_rated 但答案不是评分最高的合格 POI（最高的是 {best['name']}）")

    # Filter: only hard errors block (soft coherence warnings are OK)
    hard_errors = [e for e in errors if "·软" not in e]
    return len(hard_errors) == 0, errors


def _llm_build_task(parsed: Dict[str, Any], nearby_pois: List[Dict[str, Any]],
                    maps: GoogleMapsClient, schema_loader: SchemaLoader,
                    llm: BaseLLMClient,
                    task_type_counts: Dict[str, int]) -> Dict[str, Any]:
    """LLM-guided task construction with validation loop."""
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    _enrich_pois(nearby_pois, seed_lat, seed_lng, maps)
    poi_summary = _poi_summary_list(nearby_pois)

    if len(poi_summary) < 2:
        raise ValueError("Not enough categorizable POIs for LLM path")

    # ── Step 1: Pick scenario_frame + task_type ──
    step1_messages = _build_step1_prompt(parsed, poi_summary, schema_loader, task_type_counts)
    step1 = llm.json_completion(step1_messages, temperature=0.7)

    step1_errors = _validate_step1(step1, schema_loader)
    if step1_errors:
        raise ValueError(f"Step 1 validation failed: {'; '.join(step1_errors)}")

    print(f"    [LLM] scenario={step1['scenario_frame_id']}, task_type={step1['task_type']}")

    # ── Step 2 + 3: Fill dimensions + validate (with retry loop) ──
    violations: Optional[List[str]] = None
    step2: Optional[Dict[str, Any]] = None
    is_valid = False
    attempt = 0

    for attempt in range(_MAX_LLM_RETRIES + 1):
        step2_messages = _build_step2_prompt(step1, parsed, poi_summary, schema_loader, violations)
        step2 = llm.json_completion(step2_messages, temperature=0.7)

        global_context = step2.get("global_context", {})
        task_dimensions = step2.get("task_dimensions", {})
        answer_place_id = step2.get("answer_poi_place_id", "")

        # Find answer POI in the raw nearby_pois list
        answer_poi = next((p for p in nearby_pois if p.get("id") == answer_place_id), None)

        # Filter candidate POIs by chosen category
        category = task_dimensions.get("place_category_canonical", "")
        candidate_pois = [p for p in nearby_pois if p.get("_category") == category]
        candidate_pois.sort(key=lambda p: p.get("_distance_m", 99999))

        is_valid, violations = _validate_llm_output(
            step1["task_type"], step1["scenario_frame_id"],
            global_context, task_dimensions, answer_poi,
            poi_summary, schema_loader)

        if is_valid:
            break
        print(f"    [LLM] attempt {attempt + 1} validation failed: {violations[:2]}...")

    if not is_valid or step2 is None or answer_poi is None:
        raise ValueError(f"LLM validation failed after {_MAX_LLM_RETRIES + 1} attempts")

    # ── Build final record ──
    answer_data = _build_answer_data(answer_poi, candidate_pois)

    return _assemble_task_record(
        parsed=parsed,
        task_type=step1["task_type"],
        global_context=global_context,
        task_dimensions=task_dimensions,
        answer_data=answer_data,
        nearby_poi_count=len(nearby_pois),
        scenario_frame_id=step1["scenario_frame_id"],
        construction_meta={
            "method": "llm",
            "step1_reasoning": step1.get("reasoning", ""),
            "step2_reasoning": step2.get("reasoning", ""),
            "validation_attempts": attempt + 1,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based fallback (original logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_global_context_rule_based(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Build global_context with randomized sampling (original logic)."""
    vp = parsed.get("vision_parse", {})
    time_hint = vp.get("time_hint", "unknown")
    weather_hint = vp.get("weather_hint", "unknown")

    now = datetime.now(timezone.utc)
    offset = random.randint(-7, 7)
    hour = random.randint(8, 21)
    request_time = (now + timedelta(days=offset)).replace(
        hour=hour, minute=random.choice([0, 15, 30, 45]), second=0, microsecond=0)

    weather_map = {"clear": "clear", "cloudy": None, "rain": "rain",
                   "snow": "snow", "foggy": "foggy", "unknown": None}

    return {
        "request_time": request_time.isoformat(),
        "time_semantics": "now" if time_hint == "daytime" else "tonight" if time_hint == "nighttime" else None,
        "weather_context": weather_map.get(weather_hint),
        "user_party_composition": random.choice([None, "solo", "couple", "friends", "family_with_kids"]),
        "mobility_context": random.choice([None, "normal", "with_luggage"]),
        "urgency_level": random.choice(["low", "medium", "high"]),
        "familiarity_with_area": random.choice(["first_time_visitor", "occasional_visitor", "local_resident"]),
        "trip_state": random.choice(["exploring", "commuting", "just_arrived", "touring"]),
        "language_preference": "zh-CN",
    }


def _rule_based_build_task(parsed: Dict[str, Any], nearby_pois: List[Dict[str, Any]],
                           maps: GoogleMapsClient) -> Dict[str, Any]:
    """Original rule-based task construction (place_filter_rank only)."""
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    _enrich_pois(nearby_pois, seed_lat, seed_lng, maps)
    categorized = [(p, p["_category"]) for p in nearby_pois if p.get("_category")]

    if not categorized:
        raise ValueError("No categorizable POIs found")

    cat_counts = Counter(c for _, c in categorized)
    good_cats = [c for c, n in cat_counts.items() if n >= 2]
    if not good_cats:
        good_cats = list(cat_counts.keys())
    chosen_category = random.choice(good_cats)

    target_pois = [p for p, c in categorized if c == chosen_category]
    target_pois.sort(key=lambda p: p.get("_distance_m", 99999))

    max_distance = max(p["_distance_m"] for p in target_pois)
    distance_limit = max(int(max_distance * random.uniform(1.1, 1.5)), 500)

    ratings = [p.get("rating", 0) for p in target_pois if p.get("rating")]
    min_rating = None
    if ratings:
        min_rating = round(max(ratings) - random.uniform(0.3, 1.0), 1)
        min_rating = max(min_rating, 1.0)

    ranking_objective = random.choice(["nearest", "highest_rated"])

    task_dimensions = {
        "place_category_canonical": chosen_category,
        "place_category_surface_form": chosen_category,
        "search_region_type": "center_radius",
        "search_region_value": {"center_lat": seed_lat, "center_lng": seed_lng, "radius_m": distance_limit},
        "must_be_open_now": random.choice([True, None]),
        "distance_limit_m": distance_limit,
        "min_rating": min_rating,
        "min_rating_count": None,
        "price_level_filter": None,
        "brand_filter": None,
        "ranking_objective": ranking_objective,
        "needs_candidate_comparison": True,
        "result_count_limit": random.choice([1, 3, 5]),
    }

    if ranking_objective == "nearest":
        answer_poi = target_pois[0]
    else:
        rated = [p for p in target_pois if p.get("rating")]
        answer_poi = max(rated, key=lambda p: p["rating"]) if rated else target_pois[0]

    answer_data = _build_answer_data(answer_poi, target_pois)
    global_context = _build_global_context_rule_based(parsed)

    return _assemble_task_record(
        parsed=parsed,
        task_type="place_filter_rank",
        global_context=global_context,
        task_dimensions=task_dimensions,
        answer_data=answer_data,
        nearby_poi_count=len(nearby_pois),
        construction_meta={"method": "rule_based"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def build_task(parsed: Dict[str, Any], maps: GoogleMapsClient,
               schema_loader: SchemaLoader,
               llm: Optional[BaseLLMClient] = None,
               task_type_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Build a complete task record.

    Routing:
      anchor images → _build_anchor_task (place_lookup / geocode_resolution)
                       → fallback to LLM / rule-based
      context images → LLM (place_filter_rank / place_discovery)
                        → fallback to rule-based
    """
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]
    utility_class = parsed.get("image_utility_class", "context")
    counts = task_type_counts or {}

    nearby_pois = maps.nearby_search(lat=seed_lat, lng=seed_lng, radius_m=1000, max_results=15)
    if len(nearby_pois) < 2:
        raise ValueError(f"Not enough POIs near ({seed_lat}, {seed_lng})")

    # ── anchor images: try image-critical tasks first ──
    if utility_class == "anchor":
        try:
            return _build_anchor_task(parsed, nearby_pois, maps, schema_loader, counts)
        except Exception as e:
            print(f"    [anchor] failed ({e}), falling back to LLM/rule path")

    # ── LLM path (place_filter_rank / place_discovery) ──
    if llm and llm.available:
        try:
            return _llm_build_task(parsed, nearby_pois, maps, schema_loader, llm, counts)
        except Exception as e:
            print(f"    [LLM] failed, falling back to rule-based: {e}")

    # ── rule-based fallback ──
    return _rule_based_build_task(parsed, nearby_pois, maps)


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    maps = GoogleMapsClient(config)
    schema_loader = SchemaLoader(SCHEMA_PATH)
    llm = ClientFactory.for_stage(STAGE_TASK_DESIGN, config)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    parsed_records = read_jsonl(PARSED_PATH)
    if not parsed_records:
        print("[Module 3] No parsed records found. Run module 2 first.")
        return TASKS_PATH

    results: List[Dict[str, Any]] = []
    task_type_counts: Dict[str, int] = Counter()
    method_counts: Dict[str, int] = Counter()

    for i, parsed in enumerate(parsed_records, 1):
        image_id = parsed.get("image_id", "")
        utility = parsed.get("image_utility_class", "?")
        print(f"[Module 3] Building task {i}/{len(parsed_records)}: {image_id} ({utility})")

        try:
            task = build_task(parsed, maps, schema_loader, llm=llm,
                              task_type_counts=dict(task_type_counts))
            results.append(task)

            tt = task["task_type"]
            method = task.get("construction_metadata", {}).get("method", "unknown")
            task_type_counts[tt] += 1
            method_counts[method] += 1

            extra = ""
            if tt in ("place_filter_rank", "place_discovery"):
                extra = f", category={task['task_dimensions'].get('place_category_canonical')}"
            elif tt == "place_lookup":
                extra = f", matched={task.get('construction_metadata', {}).get('matched_sign', '')}"
            elif tt == "geocode_resolution":
                extra = f", addr={task['answer_data'].get('expected_address', '')[:30]}"

            print(f"  -> [{method}] {tt}{extra}")
        except Exception as e:
            append_error(ERRORS_PATH, {"image_id": image_id, "error": str(e)})
            print(f"  -> FAILED: {e}")
            continue

    count = write_jsonl(TASKS_PATH, results)
    print(f"\n[Module 3] Done. {count} tasks")
    print(f"  Type distribution: {dict(task_type_counts)}")
    print(f"  Method distribution: {dict(method_counts)}")
    return TASKS_PATH


if __name__ == "__main__":
    main()
