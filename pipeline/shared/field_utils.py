"""Value range parsing, sampling, and validation — extracted from pipeline.py:321-645."""

from __future__ import annotations

import random
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schema_loader import SchemaLoader


# ──────────────────────────────────────────────────────────────
# ValueRangeUtils
# ──────────────────────────────────────────────────────────────

class ValueRangeUtils:
    INT_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
    NULL_OR_INT_RANGE_RE = re.compile(r"^null\s+or\s+(\d+)-(\d+)$")
    NULL_OR_FLOAT_RANGE_RE = re.compile(r"^null\s+or\s+([0-9.]+)-([0-9.]+)$")
    CHARS_RE = re.compile(r"^(\d+)-(\d+)\s+chars\s+natural\s+language$")

    @staticmethod
    def parse_simple_range(expr: Any) -> Optional[Tuple[float, float, bool]]:
        if isinstance(expr, list):
            return None
        if not isinstance(expr, str):
            return None
        text = expr.strip()
        m = ValueRangeUtils.INT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), False
        m = ValueRangeUtils.NULL_OR_INT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), True
        m = ValueRangeUtils.NULL_OR_FLOAT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), True
        return None

    @staticmethod
    def parse_char_length(expr: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(expr, str):
            return None
        m = ValueRangeUtils.CHARS_RE.match(expr.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    @staticmethod
    def random_string(min_len: int = 4, max_len: int = 20) -> str:
        length = random.randint(min_len, max_len)
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    @staticmethod
    def random_iso_datetime() -> str:
        base = datetime.now(timezone.utc) + timedelta(days=random.randint(-30, 30))
        dt = base.replace(
            hour=random.randint(0, 23),
            minute=random.choice([0, 10, 15, 20, 30, 40, 45, 50]),
            second=0, microsecond=0,
        )
        return dt.isoformat()

    @staticmethod
    def random_language_code() -> str:
        return random.choice(["zh-CN", "en", "ja", "fr", "de", "es", "ko"])

    @staticmethod
    def random_coordinate(lat_range: Tuple[float, float] = (-60, 60),
                          lng_range: Tuple[float, float] = (-170, 170)) -> Dict[str, float]:
        return {"lat": round(random.uniform(*lat_range), 6), "lng": round(random.uniform(*lng_range), 6)}

    @staticmethod
    def random_location_object() -> Dict[str, Any]:
        input_type = random.choice(["place_id", "latlng", "address", "text_query"])
        if input_type == "place_id":
            value = f"place_{ValueRangeUtils.random_string(8, 12)}"
        elif input_type == "latlng":
            value = ValueRangeUtils.random_coordinate()
        elif input_type == "address":
            value = f"{random.randint(1, 999)} {ValueRangeUtils.random_string(5, 10)} street"
        else:
            value = ValueRangeUtils.random_string(6, 16)
        return {"input_type": input_type, "value": value}

    @staticmethod
    def random_bbox() -> Dict[str, float]:
        c = ValueRangeUtils.random_coordinate()
        return {
            "south": round(c["lat"] - 0.05, 6), "west": round(c["lng"] - 0.05, 6),
            "north": round(c["lat"] + 0.05, 6), "east": round(c["lng"] + 0.05, 6),
        }

    @staticmethod
    def random_center_radius() -> Dict[str, float]:
        c = ValueRangeUtils.random_coordinate()
        return {"center_lat": c["lat"], "center_lng": c["lng"], "radius_m": random.randint(100, 10000)}

    @staticmethod
    def is_null_like(value: Any) -> bool:
        return value is None or value == [] or value == {} or value == ""


# ──────────────────────────────────────────────────────────────
# ValueSampler
# ──────────────────────────────────────────────────────────────

class ValueSampler:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def sample_from_value_range(self, field_schema: Dict[str, Any],
                                task_type: Optional[str] = None,
                                context: Optional[Dict[str, Any]] = None) -> Any:
        value_range = self.schema_loader.get_resolved_value_range(field_schema)
        data_type = field_schema["data_type"]
        name = field_schema["name"]
        context = context or {}

        if isinstance(value_range, list):
            return random.choice(value_range)

        simple_range = ValueRangeUtils.parse_simple_range(value_range)
        if data_type in {"integer", "integer_or_null", "number_or_null"} and simple_range is not None:
            low, high, allow_null = simple_range
            if allow_null and random.random() < 0.35:
                return None
            if data_type.startswith("integer"):
                return int(random.randint(int(low), int(high)))
            return round(random.uniform(low, high), 2)

        char_range = ValueRangeUtils.parse_char_length(value_range)
        if data_type == "string" and char_range is not None:
            min_len, max_len = char_range
            return ValueRangeUtils.random_string(min_len=min_len, max_len=min(24, max_len))

        if isinstance(value_range, str):
            text = value_range.strip()

            if data_type == "string_or_null":
                if random.random() < 0.3:
                    return None
                if "ISO-8601 datetime" in text:
                    return ValueRangeUtils.random_iso_datetime()
                if "BCP-47 language code" in text:
                    return ValueRangeUtils.random_language_code()
                if "brand/business chain name" in text:
                    return random.choice(["BrandA", "BrandB", "BrandC"])
                if "ccTLD" in text or "country code" in text:
                    return random.choice(["cn", "us", "jp", "de", "fr"])
                return ValueRangeUtils.random_string(4, 12)

            if data_type == "object":
                return self._sample_object_field(name, value_range, context)
            if data_type == "object_or_null":
                if random.random() < 0.35:
                    return None
                return self._sample_object_field(name, value_range, context)
            if data_type == "string_or_object":
                return self._sample_string_or_object(name, value_range)
            if data_type == "array<object>":
                return self._sample_array_object_field(name, value_range, context)
            if data_type == "array<string>":
                return self._sample_array_string_field(name, value_range)
            if data_type == "string":
                if "ISO-8601 datetime" in text:
                    return ValueRangeUtils.random_iso_datetime()
                if "snake_case identifier" in text:
                    return f"sample_{ValueRangeUtils.random_string(5, 8)}"
                return ValueRangeUtils.random_string(4, 16)

        if data_type == "boolean":
            return random.choice([True, False])
        if data_type == "enum":
            raise ValueError(f"enum 字段必须是 list value_range，字段={name}")
        raise ValueError(f"无法为字段采样: name={name}, data_type={data_type}, value_range={value_range}")

    def _sample_object_field(self, field_name: str, value_range: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text = value_range.strip()
        if text == "{input_type, value}":
            return ValueRangeUtils.random_location_object()
        if "{center_lat, center_lng, radius_m}" in text or "location object" in text:
            return ValueRangeUtils.random_center_radius()
        if "bbox" in text:
            if field_name in {"search_region_value", "meeting_region_value", "spatial_bias_value"}:
                if random.random() < 0.4:
                    return ValueRangeUtils.random_bbox()
                return ValueRangeUtils.random_center_radius()
        if "user_location" in text:
            return {"user_location": ValueRangeUtils.random_coordinate(), "radius_m": random.randint(100, 5000)}
        if "route_polyline_ref" in text:
            return {"route_polyline_ref": f"poly_{ValueRangeUtils.random_string(8, 12)}"}
        if "region object" in text:
            if random.random() < 0.5:
                return {"city_name": random.choice(["Beijing", "Shanghai", "Tokyo", "Paris", "New York"])}
            return ValueRangeUtils.random_bbox()
        return {"value": ValueRangeUtils.random_string(6, 12)}

    def _sample_string_or_object(self, field_name: str, value_range: str) -> Any:
        text = value_range.strip()
        if "address string" in text or "place_id string" in text:
            choice = random.choice(["address", "latlng", "place_id"])
            if choice == "address":
                return f"{random.randint(1, 999)} {ValueRangeUtils.random_string(5, 10)} road"
            if choice == "place_id":
                return f"place_{ValueRangeUtils.random_string(8, 12)}"
            return ValueRangeUtils.random_coordinate()
        return ValueRangeUtils.random_string(5, 12)

    def _sample_array_object_field(self, field_name: str, value_range: str,
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = value_range.strip()
        m = re.match(r"^(\d+)-(\d+)\s+.*objects$", text)
        count = 2 if not m else random.randint(int(m.group(1)), min(int(m.group(2)), int(m.group(1)) + 3))
        items: List[Dict[str, Any]] = []
        for idx in range(count):
            if field_name in {"candidate_destinations", "origins"}:
                items.append(ValueRangeUtils.random_location_object())
            elif field_name == "candidate_goals":
                items.append({"goal_id": f"goal_{idx+1}", "goal_type": random.choice(["place", "category"]),
                              "value": ValueRangeUtils.random_location_object()})
            else:
                items.append({"value": ValueRangeUtils.random_string(5, 10)})
        return items

    def _sample_array_string_field(self, field_name: str, value_range: str) -> List[str]:
        text = value_range.strip()
        m = re.match(r"^(\d+)-(\d+)\s+goal\s+ids$", text)
        if m:
            low, high = int(m.group(1)), int(m.group(2))
            count = random.randint(low, min(high, 3))
            return [f"goal_{i+1}" for i in range(count)]
        return [ValueRangeUtils.random_string(5, 8) for _ in range(random.randint(1, 3))]


# ──────────────────────────────────────────────────────────────
# FieldValidator
# ──────────────────────────────────────────────────────────────

class FieldValidator:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def validate_value(self, field_schema: Dict[str, Any], value: Any) -> bool:
        data_type = field_schema["data_type"]
        value_range = self.schema_loader.get_resolved_value_range(field_schema)

        if isinstance(value_range, list):
            return value in value_range
        if data_type == "boolean":
            return isinstance(value, bool)
        if data_type == "string":
            if not isinstance(value, str):
                return False
            char_range = ValueRangeUtils.parse_char_length(value_range)
            if char_range is not None:
                return char_range[0] <= len(value) <= char_range[1]
            if isinstance(value_range, str) and "ISO-8601 datetime" in value_range:
                return self._is_iso_datetime(value)
            return True
        if data_type == "string_or_null":
            if value is None:
                return True
            if not isinstance(value, str):
                return False
            if isinstance(value_range, str) and "ISO-8601 datetime" in value_range:
                return self._is_iso_datetime(value)
            return True
        if data_type == "integer":
            return isinstance(value, int) and self._validate_numeric_range(value_range, value)
        if data_type == "integer_or_null":
            return value is None or (isinstance(value, int) and self._validate_numeric_range(value_range, value))
        if data_type == "number_or_null":
            return value is None or (isinstance(value, (int, float)) and self._validate_numeric_range(value_range, float(value)))
        if data_type in {"enum", "enum_or_null", "boolean_or_null"}:
            return value in value_range
        if data_type == "object":
            return isinstance(value, dict)
        if data_type == "object_or_null":
            return value is None or isinstance(value, dict)
        if data_type == "string_or_object":
            return isinstance(value, (str, dict))
        if data_type == "array<object>":
            return isinstance(value, list) and all(isinstance(i, dict) for i in value)
        if data_type == "array<string>":
            return isinstance(value, list) and all(isinstance(i, str) for i in value)
        return True

    def _is_iso_datetime(self, text: str) -> bool:
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return True
        except Exception:
            return False

    def _validate_numeric_range(self, expr: Any, value: float) -> bool:
        simple_range = ValueRangeUtils.parse_simple_range(expr)
        if simple_range is None:
            return True
        low, high, _allow_null = simple_range
        return low <= value <= high

    def validate_record(self, field_schemas: List[Dict[str, Any]],
                        record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for field_schema in field_schemas:
            name = field_schema["name"]
            required = field_schema.get("required", False)
            if required and name not in record:
                errors.append(f"缺少必填字段: {name}")
                continue
            if name in record and not self.validate_value(field_schema, record[name]):
                errors.append(f"字段不合法: {name}, value={record[name]!r}")
        return len(errors) == 0, errors
