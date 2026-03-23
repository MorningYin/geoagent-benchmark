"""Module 5: Ground Truth Anchor — API-based fact verification.

Pipeline v4: replaces v3's NL query writer with programmatic API verification.
No LLM calls. Uses Google Maps API to verify that facts referenced in the
generated task actually hold in reality.

Checks performed:
  - Referenced place names exist near the stated coordinates
  - Coordinates are valid
  - answer_reference fields can be corroborated by API data

Does NOT hard-reject: marks tasks as verified=true/false for human review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.maps_client import GoogleMapsClient
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = PIPELINE_ROOT / "data" / "04_judged"
DATA_OUT = PIPELINE_ROOT / "data" / "05_verified"
APPROVED_PATH = DATA_IN / "approved.jsonl"
OUTPUT_PATH = DATA_OUT / "verified.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"


def _check_coordinates_valid(location: Dict[str, float]) -> Dict[str, Any]:
    """Check that coordinates are within valid ranges."""
    lat = location.get("lat", 0)
    lng = location.get("lng", 0)
    passed = -90 <= lat <= 90 and -180 <= lng <= 180
    return {
        "check": "valid_coordinates",
        "passed": passed,
        "detail": f"lat={lat}, lng={lng}",
    }


def _check_place_exists(
    place_name: str,
    lat: float,
    lng: float,
    maps: GoogleMapsClient,
    radius_m: int = 2000,
) -> Dict[str, Any]:
    """Check if a named place exists near the given coordinates."""
    try:
        results = maps.nearby_search(lat, lng, radius_m=radius_m)
        # Fuzzy match: check if place_name is a substring of any result name
        found_names = []
        for r in results:
            name = r.get("displayName", {}).get("text", "")
            found_names.append(name)
            if place_name in name or name in place_name:
                return {
                    "check": "place_exists",
                    "passed": True,
                    "detail": f"Found '{name}' matching '{place_name}'",
                }
        return {
            "check": "place_exists",
            "passed": False,
            "detail": (
                f"'{place_name}' not found among nearby POIs: "
                f"{found_names[:5]}"
            ),
        }
    except Exception as e:
        return {
            "check": "place_exists",
            "passed": False,
            "detail": f"API error: {e}",
        }


def verify_task(task: Dict[str, Any], maps: GoogleMapsClient) -> Dict[str, Any]:
    """Run ground truth verification on a single task.

    Attaches a ground_truth field with check results.
    Never raises — errors are captured in check results.
    """
    checks: List[Dict[str, Any]] = []
    provenance = task.get("provenance", {})
    location = provenance.get("location", {"lat": 0, "lng": 0})
    answer_ref = task.get("verification", {}).get("answer_reference", {})

    # Check 1: Valid coordinates
    checks.append(_check_coordinates_valid(location))

    # Check 2: If answer_reference mentions a place_name, verify it exists
    place_name = answer_ref.get("place_name") or answer_ref.get("name")
    if place_name and isinstance(place_name, str):
        checks.append(
            _check_place_exists(place_name, location["lat"], location["lng"], maps)
        )

    # Check 3: If answer_reference mentions an address, verify via reverse geocode
    expected_address = answer_ref.get("address")
    if expected_address and isinstance(expected_address, str) and len(expected_address) > 3:
        try:
            geo_result = maps.reverse_geocode(location["lat"], location["lng"])
            actual_address = geo_result.get("formatted_address", "")
            # Loose match: check if any part of expected appears in actual
            match = any(
                part in actual_address
                for part in expected_address.split()
                if len(part) >= 2
            )
            checks.append({
                "check": "address_match",
                "passed": match,
                "detail": f"Expected: {expected_address}, Got: {actual_address}",
            })
        except Exception as e:
            checks.append({
                "check": "address_match",
                "passed": False,
                "detail": f"Geocode API error: {e}",
            })

    task["ground_truth"] = {
        "verified": all(c["passed"] for c in checks) if checks else False,
        "checks": checks,
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return task


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    maps = GoogleMapsClient(config)

    tasks = read_jsonl(APPROVED_PATH)
    if not tasks:
        print("[Module 5] No approved tasks found. Run module 4 first.")
        return OUTPUT_PATH

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    verified_count = 0

    for i, task in enumerate(tasks, 1):
        image_id = task.get("provenance", {}).get("image_id", "unknown")
        print(f"[Module 5] Verifying task {i}/{len(tasks)}: {image_id}")

        try:
            verified_task = verify_task(task, maps)
            gt = verified_task["ground_truth"]
            results.append(verified_task)

            if gt["verified"]:
                verified_count += 1
                print(f"  -> VERIFIED ({len(gt['checks'])} checks passed)")
            else:
                failed = [c for c in gt["checks"] if not c["passed"]]
                print(f"  -> UNVERIFIED ({len(failed)} checks failed: "
                      f"{[c['check'] for c in failed]})")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            append_error(ERRORS_PATH, {
                "image_id": image_id, "error": str(e),
            })

    count = write_jsonl(OUTPUT_PATH, results)
    print(f"\n[Module 5] Done. {count} tasks processed, "
          f"{verified_count} fully verified.")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
