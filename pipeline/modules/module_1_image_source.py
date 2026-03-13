"""Module 1: Image Source — download Street View images for seed points via Google Maps APIs."""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.maps_client import GoogleMapsClient
from shared.jsonl_io import write_jsonl, append_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "01_images"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
ERRORS_PATH = DATA_DIR / "errors.jsonl"

# POI types to search per area_type
AREA_TYPE_SEARCH = {
    "commercial":    ["restaurant", "cafe", "shopping_mall", "store"],
    "transport_hub": ["transit_station", "bus_station", "train_station"],
    "residential":   ["supermarket", "pharmacy", "convenience_store", "park"],
    "tourist":       ["tourist_attraction", "museum", "landmark"],
    "university":    ["university", "library", "book_store", "cafe"],
}


def _pick_search_types(area_type: str) -> List[str]:
    types = AREA_TYPE_SEARCH.get(area_type, ["restaurant"])
    return random.sample(types, min(2, len(types)))


def process_seed(seed: Dict[str, Any], maps: GoogleMapsClient,
                 max_pois: int = 3) -> List[Dict[str, Any]]:
    """For one seed point, find nearby POIs and download street view images. Returns manifest records."""
    records: List[Dict[str, Any]] = []
    search_types = _pick_search_types(seed.get("area_type", "commercial"))

    try:
        pois = maps.nearby_search(
            lat=seed["lat"], lng=seed["lng"],
            radius_m=800,
            included_types=search_types,
            max_results=max_pois * 2,
        )
    except Exception as e:
        append_error(ERRORS_PATH, {"seed": seed, "stage": "nearby_search", "error": str(e)})
        return records

    selected = pois[:max_pois]

    for poi in selected:
        poi_loc = poi.get("location", {})
        poi_lat = poi_loc.get("latitude", seed["lat"])
        poi_lng = poi_loc.get("longitude", seed["lng"])

        heading = maps.compute_heading(seed["lat"], seed["lng"], poi_lat, poi_lng)

        image_id = uuid.uuid4().hex[:12]
        image_filename = f"{image_id}.jpg"
        image_path = IMAGES_DIR / image_filename

        try:
            img_bytes = maps.download_street_view(
                lat=seed["lat"], lng=seed["lng"],
                heading=heading, pitch=0, fov=90,
            )
            image_path.write_bytes(img_bytes)
        except Exception as e:
            append_error(ERRORS_PATH, {"seed": seed, "poi_id": poi.get("id"), "stage": "street_view_download", "error": str(e)})
            continue

        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "seed_city": seed.get("city"),
            "seed_label": seed.get("label"),
            "seed_lat": seed["lat"],
            "seed_lng": seed["lng"],
            "area_type": seed.get("area_type"),
            "poi_id": poi.get("id"),
            "poi_name": poi.get("displayName", {}).get("text", ""),
            "poi_address": poi.get("formattedAddress", ""),
            "poi_lat": poi_lat,
            "poi_lng": poi_lng,
            "poi_types": poi.get("types", []),
            "poi_rating": poi.get("rating"),
            "poi_rating_count": poi.get("userRatingCount"),
            "heading": heading,
        }
        records.append(record)

    return records


def main(config: Optional[PipelineConfig] = None, seed_limit: Optional[int] = None) -> Path:
    """Run module 1: download images for all seed points.

    Args:
        config: Pipeline configuration.
        seed_limit: If set, only process this many seeds (for testing).

    Returns:
        Path to the manifest.jsonl file.
    """
    config = config or PipelineConfig()
    maps = GoogleMapsClient(config)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    seeds_path = Path(__file__).resolve().parent.parent / "seeds" / "city_seeds.json"
    with open(seeds_path, encoding="utf-8") as f:
        seeds: List[Dict[str, Any]] = json.load(f)

    if seed_limit is not None:
        seeds = seeds[:seed_limit]

    all_records: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds, 1):
        print(f"[Module 1] Processing seed {i}/{len(seeds)}: {seed.get('label', '')}")
        records = process_seed(seed, maps)
        all_records.extend(records)
        print(f"  -> got {len(records)} images")

    count = write_jsonl(MANIFEST_PATH, all_records)
    print(f"[Module 1] Done. {count} images written to {MANIFEST_PATH}")
    return MANIFEST_PATH


if __name__ == "__main__":
    main()
