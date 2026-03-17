"""Module 1 (GAEA variant): Build image manifest from HuggingFace GAEA-Train dataset.

Replaces the original module_1_image_source.py which downloads Street View via Google Maps API.
This module loads images + coordinates from ucf-crcv/GAEA-Train via the HuggingFace Dataset
Viewer API (/rows endpoint), fetching rows on demand and downloading images from CDN URLs.

Usage:
    python -m modules.module_1_gaea_source              # default: 100 samples
    python -m modules.module_1_gaea_source --seed-limit 10  # for testing

Prerequisites:
    pip install requests Pillow
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from shared.config import PipelineConfig
from shared.jsonl_io import write_jsonl, append_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "01_images"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
ERRORS_PATH = DATA_DIR / "errors.jsonl"

# HuggingFace Dataset Viewer API
# 默认用官方源；国内如果连不上，设置 HF_ROWS_API 环境变量指向可用镜像
ROWS_API = os.environ.get(
    "HF_ROWS_API",
    "https://datasets-server.huggingface.co/rows"
)
DATASET_ID = "ucf-crcv/GAEA-Train"
PAGE_SIZE = 20  # keep small to avoid 429 rate limits on mirrors

PREFERRED_SUBSETS = {"Conversational"}


def _parse_location(location_str: str) -> tuple[str, str]:
    """Parse 'City, Country' into (city, country). Handles edge cases."""
    parts = [p.strip() for p in location_str.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return parts[0], ""


def _is_valid_coord(lat: float, lng: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lng <= 180


def _fetch_rows(offset: int, length: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Fetch a page of rows from the Dataset Viewer API with retry on 429."""
    import time
    for attempt in range(max_retries):
        resp = requests.get(ROWS_API, params={
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }, timeout=60)
        if resp.status_code == 429:
            wait = 2 ** attempt + 1  # 2, 3, 5 seconds
            print(f"[Module 1 / GAEA]   429 rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()  # raise the last 429
    data = resp.json()
    return [item["row"] for item in data.get("rows", [])]


def _download_image(url: str) -> bytes:
    """Download image bytes from a HuggingFace cached-assets URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def load_gaea_samples(seed_limit: int = 100) -> List[Dict[str, Any]]:
    """Load samples from GAEA-Train via the Dataset Viewer /rows API.

    Fetches rows in pages of 100, filters for Conversational subset,
    deduplicates by location, and downloads images individually from CDN.
    """
    samples: List[Dict[str, Any]] = []
    seen_locations: set[str] = set()
    offset = 0
    consecutive_empty = 0

    while len(samples) < seed_limit:
        print(f"[Module 1 / GAEA] Fetching rows {offset}–{offset + PAGE_SIZE}...")
        try:
            rows = _fetch_rows(offset, PAGE_SIZE)
        except Exception as e:
            print(f"[Module 1 / GAEA] API error at offset {offset}: {e}")
            break

        if not rows:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("[Module 1 / GAEA] No more rows available.")
                break
            offset += PAGE_SIZE
            continue
        consecutive_empty = 0

        for row in rows:
            if len(samples) >= seed_limit:
                break

            # Filter by subset
            if row.get("subset") not in PREFERRED_SUBSETS:
                continue

            # Parse coordinates
            try:
                lat = float(row["lat"])
                lng = float(row["lon"])
            except (ValueError, TypeError, KeyError):
                continue
            if not _is_valid_coord(lat, lng):
                continue

            # Location dedup
            location = row.get("location", "")
            if not location or location in seen_locations:
                continue
            seen_locations.add(location)

            # Get image URL: file_name field is {"src": url, "height": ..., "width": ...}
            file_name_field = row.get("file_name")
            if isinstance(file_name_field, dict):
                image_url = file_name_field.get("src", "")
            else:
                image_url = ""
            if not image_url:
                continue

            # Download image
            try:
                img_bytes = _download_image(image_url)
            except Exception as e:
                append_error(ERRORS_PATH, {
                    "location": location, "stage": "image_download", "error": str(e),
                })
                continue

            samples.append({
                "image_bytes": img_bytes,
                "lat": lat,
                "lng": lng,
                "location": location,
                "dataset": row.get("dataset", ""),
            })
            print(f"[Module 1 / GAEA]   [{len(samples)}/{seed_limit}] {location}")

        offset += PAGE_SIZE

    return samples


def process_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Save images to disk and build manifest records."""
    from PIL import Image as PILImage

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for sample in samples:
        image_id = uuid.uuid4().hex[:12]
        image_filename = f"{image_id}.jpg"
        image_path = IMAGES_DIR / image_filename

        try:
            pil_image = PILImage.open(io.BytesIO(sample["image_bytes"]))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(str(image_path), "JPEG", quality=90)
        except Exception as e:
            append_error(ERRORS_PATH, {
                "location": sample["location"],
                "stage": "image_save",
                "error": str(e),
            })
            continue

        city, country = _parse_location(sample["location"])

        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "seed_city": city,
            "seed_country": country,
            "seed_label": sample["location"],
            "seed_lat": sample["lat"],
            "seed_lng": sample["lng"],
            "area_type": None,           # will be inferred by Module 2 from scene_type
            "source_dataset": sample.get("dataset", ""),
        }
        records.append(record)

    return records


def main(config: Optional[PipelineConfig] = None,
         seed_limit: Optional[int] = None) -> Path:
    """Run module 1 (GAEA variant): load images from HuggingFace dataset.

    Args:
        config: Pipeline configuration (not heavily used in this module).
        seed_limit: Number of samples to collect. Defaults to 100.

    Returns:
        Path to the manifest.jsonl file.
    """
    limit = seed_limit or 100

    print(f"[Module 1 / GAEA] Loading up to {limit} samples from {DATASET_ID}...")
    samples = load_gaea_samples(seed_limit=limit)
    print(f"[Module 1 / GAEA] Collected {len(samples)} valid samples")

    print(f"[Module 1 / GAEA] Saving images to {IMAGES_DIR}...")
    records = process_samples(samples)

    count = write_jsonl(MANIFEST_PATH, records)
    print(f"[Module 1 / GAEA] Done. {count} images written to {MANIFEST_PATH}")
    return MANIFEST_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1: GAEA Image Source")
    parser.add_argument("--seed-limit", type=int, default=100,
                        help="Number of samples to collect")
    args = parser.parse_args()

    main(seed_limit=args.seed_limit)
