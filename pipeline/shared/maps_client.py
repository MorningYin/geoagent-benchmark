"""Google Maps API real client — Nearby Search, Place Details, Street View URL."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from .config import PipelineConfig

_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
_PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
_STREET_VIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


class GoogleMapsClient:
    """Thin wrapper around Google Maps Platform APIs (Places API New + Street View Static)."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._api_key = self.config.google_maps_api_key
        self._session = requests.Session()

    # ── Nearby Search (New) ──
    def nearby_search(self, lat: float, lng: float, radius_m: int = 500,
                      included_types: Optional[List[str]] = None,
                      max_results: int = 10,
                      language: str = "zh-CN") -> List[Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.types,places.rating,"
                "places.userRatingCount,places.currentOpeningHours,"
                "places.regularOpeningHours,places.primaryType"
            ),
        }
        body: Dict[str, Any] = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(radius_m),
                }
            },
            "maxResultCount": max_results,
            "languageCode": language,
        }
        if included_types:
            body["includedTypes"] = included_types

        resp = self._session.post(_NEARBY_SEARCH_URL, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("places", [])

    # ── Place Details (New) ──
    def place_details(self, place_id: str,
                      fields: Optional[List[str]] = None,
                      language: str = "zh-CN") -> Dict[str, Any]:
        if fields is None:
            fields = [
                "id", "displayName", "formattedAddress", "location",
                "types", "rating", "userRatingCount",
                "currentOpeningHours", "regularOpeningHours",
                "primaryType", "websiteUri", "nationalPhoneNumber",
            ]
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": ",".join(fields),
        }
        url = _PLACE_DETAILS_URL.format(place_id=place_id)
        resp = self._session.get(url, headers=headers, params={"languageCode": language}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Street View Static API ──
    def street_view_url(self, lat: float, lng: float,
                        heading: int = 0, pitch: int = 0,
                        fov: int = 90, size: str = "640x640") -> str:
        params = {
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": size,
            "key": self._api_key,
        }
        return f"{_STREET_VIEW_URL}?{urlencode(params)}"

    def download_street_view(self, lat: float, lng: float,
                             heading: int = 0, pitch: int = 0,
                             fov: int = 90, size: str = "640x640") -> bytes:
        url = self.street_view_url(lat, lng, heading, pitch, fov, size)
        resp = self._session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content

    # ── Reverse Geocoding ──
    def reverse_geocode(self, lat: float, lng: float,
                        language: str = "zh-CN") -> Dict[str, Any]:
        """Reverse geocode coordinates to a structured address.

        Returns the first (most specific) result from Google Geocoding API.
        """
        params = {
            "latlng": f"{lat},{lng}",
            "language": language,
            "key": self._api_key,
        }
        resp = self._session.get(_GEOCODE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return {"formatted_address": "", "address_components": [], "place_id": ""}
        first = results[0]
        return {
            "formatted_address": first.get("formatted_address", ""),
            "address_components": first.get("address_components", []),
            "place_id": first.get("place_id", ""),
            "types": first.get("types", []),
        }

    # ── Utility: compute heading from seed to POI ──
    @staticmethod
    def compute_heading(from_lat: float, from_lng: float,
                        to_lat: float, to_lng: float) -> int:
        lat1, lng1 = math.radians(from_lat), math.radians(from_lng)
        lat2, lng2 = math.radians(to_lat), math.radians(to_lng)
        d_lng = lng2 - lng1
        x = math.sin(d_lng) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
        heading = math.degrees(math.atan2(x, y))
        return int((heading + 360) % 360)

    @staticmethod
    def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
