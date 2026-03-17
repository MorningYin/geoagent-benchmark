"""Module 3: Task Builder — reverse-construct tasks from real images + POI data.

This is the most complex module. Steps:
1. Query real nearby POIs (Google Maps Nearby Search)
2. Decide task type (initially fixed to place_filter_rank)
3. Reverse-construct task_dimensions from real POI data
4. Create information gap plan (what info comes from image vs query vs GPS)
5. Fill standard answer + constraint_check_records using real API data
"""

from __future__ import annotations

import copy
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.config import PipelineConfig
from shared.maps_client import GoogleMapsClient
from shared.llm_client import DashScopeClient
from shared.schema_loader import SchemaLoader
from shared.answer_builder import AnswerTemplateBuilder
from shared.field_utils import ValueRangeUtils
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "02_parsed"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "03_tasks"
PARSED_PATH = DATA_IN / "parsed.jsonl"
TASKS_PATH = DATA_OUT / "tasks.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.json"

# Category mapping: POI primary_type -> schema place_category_canonical
_POI_TYPE_TO_CATEGORY = {
    "restaurant": "restaurant",
    "cafe": "cafe",
    "shopping_mall": "shopping_mall",
    "supermarket": "supermarket",
    "pharmacy": "pharmacy",
    "convenience_store": "convenience_store",
    "park": "park",
    "museum": "museum",
    "tourist_attraction": "tourist_attraction",
    "transit_station": "transit_station",
    "bus_station": "bus_station",
    "train_station": "train_station",
    "university": "university",
    "library": "library",
    "book_store": "book_store",
    "hospital": "hospital",
    "hotel": "hotel",
    "gas_station": "gas_station",
    "bank": "bank",
    "post_office": "post_office",
}


def _infer_category(poi: Dict[str, Any]) -> Optional[str]:
    """Infer a schema-compatible category from POI types."""
    primary = poi.get("primaryType", "")
    if primary in _POI_TYPE_TO_CATEGORY:
        return _POI_TYPE_TO_CATEGORY[primary]
    for t in poi.get("types", []):
        if t in _POI_TYPE_TO_CATEGORY:
            return _POI_TYPE_TO_CATEGORY[t]
    return None


def _build_global_context(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Build global_context from image parse data + randomized elements."""
    vp = parsed.get("vision_parse", {})
    time_hint = vp.get("time_hint", "unknown")
    weather_hint = vp.get("weather_hint", "unknown")

    # Generate a plausible request_time
    now = datetime.now(timezone.utc)
    offset = random.randint(-7, 7)
    hour = random.randint(8, 21)
    request_time = (now + timedelta(days=offset)).replace(
        hour=hour, minute=random.choice([0, 15, 30, 45]), second=0, microsecond=0
    )

    weather_map = {"clear": "clear", "cloudy": None, "rain": "rain",
                   "snow": "snow", "foggy": "foggy", "unknown": None}
    weather_ctx = weather_map.get(weather_hint)

    return {
        "request_time": request_time.isoformat(),
        "time_semantics": "now" if time_hint == "daytime" else "tonight" if time_hint == "nighttime" else None,
        "weather_context": weather_ctx,
        "user_party_composition": random.choice([None, "solo", "couple", "friends", "family_with_kids"]),
        "mobility_context": random.choice([None, "normal", "with_luggage"]),
        "urgency_level": random.choice(["low", "medium", "high"]),
        "familiarity_with_area": random.choice(["first_time_visitor", "occasional_visitor", "local_resident"]),
        "trip_state": random.choice(["exploring", "commuting", "just_arrived", "touring"]),
        "language_preference": "zh-CN",
    }


def _build_task_dimensions(parsed: Dict[str, Any],
                           nearby_pois: List[Dict[str, Any]],
                           maps: GoogleMapsClient) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reverse-construct task_dimensions from real POI data.

    Returns (task_dimensions, answer_data).
    """
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    # Pick a category from available POIs
    categorized_pois = [(poi, _infer_category(poi)) for poi in nearby_pois]
    categorized_pois = [(p, c) for p, c in categorized_pois if c is not None]

    if not categorized_pois:
        raise ValueError("No categorizable POIs found")

    # Group by category and pick one with enough POIs
    from collections import Counter
    cat_counts = Counter(c for _, c in categorized_pois)
    # Prefer categories with 3+ POIs for ranking tasks
    good_cats = [c for c, n in cat_counts.items() if n >= 2]
    if not good_cats:
        good_cats = list(cat_counts.keys())
    chosen_category = random.choice(good_cats)

    target_pois = [p for p, c in categorized_pois if c == chosen_category]

    # Compute real distances
    for poi in target_pois:
        loc = poi.get("location", {})
        poi_lat = loc.get("latitude", seed_lat)
        poi_lng = loc.get("longitude", seed_lng)
        poi["_distance_m"] = maps.haversine_distance(seed_lat, seed_lng, poi_lat, poi_lng)

    # Sort by distance for ranking
    target_pois.sort(key=lambda p: p.get("_distance_m", 99999))

    # --- Reverse-construct constraints ---
    max_distance = max(p["_distance_m"] for p in target_pois)
    distance_limit = int(max_distance * random.uniform(1.1, 1.5))
    distance_limit = max(distance_limit, 500)

    # Rating constraint: set slightly below best rating so best passes
    ratings = [p.get("rating", 0) for p in target_pois if p.get("rating")]
    min_rating = None
    if ratings:
        best_rating = max(ratings)
        min_rating = round(best_rating - random.uniform(0.3, 1.0), 1)
        min_rating = max(min_rating, 1.0)

    ranking_objective = random.choice(["nearest", "highest_rated"])

    task_dimensions = {
        "place_category_canonical": chosen_category,
        "place_category_surface_form": chosen_category,  # will be aliased in query writer
        "search_region_type": "center_radius",
        "search_region_value": {
            "center_lat": seed_lat,
            "center_lng": seed_lng,
            "radius_m": distance_limit,
        },
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

    # --- Build answer from real data ---
    if ranking_objective == "nearest":
        answer_poi = target_pois[0]
    else:
        rated = [p for p in target_pois if p.get("rating")]
        answer_poi = max(rated, key=lambda p: p["rating"]) if rated else target_pois[0]

    answer_loc = answer_poi.get("location", {})
    answer_data = {
        "expected_place_name": answer_poi.get("displayName", {}).get("text", ""),
        "expected_place_id": answer_poi.get("id", ""),
        "expected_address": answer_poi.get("formattedAddress", ""),
        "expected_lat": answer_loc.get("latitude"),
        "expected_lng": answer_loc.get("longitude"),
        "expected_rating": answer_poi.get("rating"),
        "expected_distance_m": round(answer_poi.get("_distance_m", 0)),
        "candidate_pois": [
            {
                "name": p.get("displayName", {}).get("text", ""),
                "place_id": p.get("id", ""),
                "distance_m": round(p.get("_distance_m", 0)),
                "rating": p.get("rating"),
            }
            for p in target_pois
        ],
    }

    return task_dimensions, answer_data


def _build_information_gap_plan(parsed: Dict[str, Any],
                                task_dimensions: Dict[str, Any]) -> Dict[str, Any]:
    """Determine what info the image provides vs what must come from query/GPS."""
    vp = parsed.get("vision_parse", {})
    ocr = vp.get("ocr_texts", [])
    scene = vp.get("scene_type", "")
    geo = vp.get("geo_hints", {})

    image_provides = []
    query_must_state = []
    gps_provides = []
    must_not_leak = []

    # Image can provide location context
    if geo.get("level") in ("street", "building", "district"):
        image_provides.append("approximate_location")
    elif geo.get("level") == "city":
        image_provides.append("city_level_location")
    if ocr:
        image_provides.append("visible_text_clues")
    if scene:
        image_provides.append(f"scene_type:{scene}")
    # Visible entities provide environmental context
    entities = vp.get("visible_entities", [])
    if len(entities) >= 2:
        image_provides.append("environmental_context")

    # GPS provides coordinates
    gps_provides.append("exact_coordinates")
    gps_provides.append("search_center")

    # Query must state the category need
    category = task_dimensions.get("place_category_canonical", "")
    query_must_state.append(f"need_category:{category}")

    # Query must state ranking preference
    ranking = task_dimensions.get("ranking_objective", "")
    if ranking:
        query_must_state.append(f"ranking_preference:{ranking}")

    # Must not leak: the answer POI name, exact distance
    must_not_leak.append("answer_place_name")
    must_not_leak.append("exact_distances")
    if task_dimensions.get("min_rating"):
        must_not_leak.append("min_rating_threshold")

    return {
        "image_provides": image_provides,
        "query_must_state": query_must_state,
        "gps_provides": gps_provides,
        "must_not_leak_in_query": must_not_leak,
    }


def build_task(parsed: Dict[str, Any], maps: GoogleMapsClient,
               schema_loader: SchemaLoader) -> Dict[str, Any]:
    """Build a complete task record from a parsed image record."""
    seed_lat = parsed["seed_lat"]
    seed_lng = parsed["seed_lng"]

    # 1. Query real nearby POIs
    nearby_pois = maps.nearby_search(
        lat=seed_lat, lng=seed_lng,
        radius_m=1000,
        max_results=15,
    )

    if len(nearby_pois) < 2:
        raise ValueError(f"Not enough POIs near ({seed_lat}, {seed_lng})")

    # 2. Task type: place_filter_rank (initial version)
    task_type = "place_filter_rank"

    # 3. Reverse-construct task_dimensions
    task_dimensions, answer_data = _build_task_dimensions(parsed, nearby_pois, maps)

    # 4. Build global_context
    global_context = _build_global_context(parsed)

    # 5. Information gap plan
    info_gap = _build_information_gap_plan(parsed, task_dimensions)

    # 6. Build device_context (from image seed)
    device_context = {
        "gps": {"lat": seed_lat, "lng": seed_lng},
        "current_time": global_context["request_time"],
        "device_language": global_context.get("language_preference", "zh-CN"),
    }

    # 7. Assemble task record
    return {
        "image_id": parsed["image_id"],
        "image_path": parsed.get("image_path", ""),
        "task_type": task_type,
        "global_context": global_context,
        "task_dimensions": task_dimensions,
        "vision_parse": parsed.get("vision_parse"),
        "real_world_context": {
            "seed": {
                "city": parsed.get("seed_city"),
                "label": parsed.get("seed_label"),
                "area_type": parsed.get("area_type"),
            },
            "nearby_poi_count": len(nearby_pois),
        },
        "information_gap_plan": info_gap,
        "answer_data": answer_data,
        "device_context": device_context,
    }


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    maps = GoogleMapsClient(config)
    schema_loader = SchemaLoader(SCHEMA_PATH)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    parsed_records = read_jsonl(PARSED_PATH)
    if not parsed_records:
        print("[Module 3] No parsed records found. Run module 2 first.")
        return TASKS_PATH

    results: List[Dict[str, Any]] = []

    for i, parsed in enumerate(parsed_records, 1):
        image_id = parsed.get("image_id", "")
        print(f"[Module 3] Building task {i}/{len(parsed_records)}: {image_id}")

        try:
            task = build_task(parsed, maps, schema_loader)
            results.append(task)
            print(f"  -> task_type={task['task_type']}, "
                  f"category={task['task_dimensions'].get('place_category_canonical')}")
        except Exception as e:
            append_error(ERRORS_PATH, {"image_id": image_id, "error": str(e)})
            print(f"  -> FAILED: {e}")
            continue

    count = write_jsonl(TASKS_PATH, results)
    print(f"[Module 3] Done. {count} tasks written to {TASKS_PATH}")
    return TASKS_PATH


if __name__ == "__main__":
    main()
