"""Schema loader — extracted from pipeline.py:123-241."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SchemaLoader:
    def __init__(self, schema_path: Union[str, Path]):
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"schema 文件不存在: {self.schema_path}")
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self._task_category_map = {item["task_type"]: item for item in self.schema["task_categories"]}
        self._scenario_frame_map = {item["scenario_id"]: item for item in self.schema["scenario_frames"]}
        self._global_context_map = {item["name"]: item for item in self.schema["global_context_schema"]}
        self._input_dimension_map = {
            task_type: {item["name"]: item for item in items}
            for task_type, items in self.schema["input_dimensions"].items()
        }
        self._controlled_vocabs = self.schema.get("controlled_vocabularies", {})

    @property
    def benchmark_name(self) -> str:
        return self.schema["benchmark_name"]

    @property
    def version(self) -> str:
        return self.schema["version"]

    @property
    def task_types(self) -> List[str]:
        return list(self._task_category_map.keys())

    def task_category(self, task_type: str) -> Dict[str, Any]:
        return self._task_category_map[task_type]

    def scenario_frames(self) -> List[Dict[str, Any]]:
        return self.schema["scenario_frames"]

    def scenario_frame(self, scenario_id: str) -> Dict[str, Any]:
        return self._scenario_frame_map[scenario_id]

    def global_context_fields(self) -> List[Dict[str, Any]]:
        return self.schema["global_context_schema"]

    def global_context_field(self, name: str) -> Dict[str, Any]:
        return self._global_context_map[name]

    def input_dimensions(self, task_type: str) -> List[Dict[str, Any]]:
        return self.schema["input_dimensions"][task_type]

    def input_dimension(self, task_type: str, name: str) -> Dict[str, Any]:
        return self._input_dimension_map[task_type][name]

    def answer_template(self, task_type: str) -> Dict[str, Any]:
        return self.schema["standard_answer_templates"][task_type]

    def classification_rules(self) -> Dict[str, Any]:
        return self.schema["classification_rules"]

    def coherence_rules(self) -> Dict[str, Any]:
        return self.schema["coherence_rules"]

    def constraint_check_schema(self) -> Dict[str, Any]:
        return self.schema["constraint_check_record_schema"]

    def resolve_value_range_ref(self, value_range_ref: str) -> Any:
        current: Any = self.schema
        parts = value_range_ref.split(".")
        for part in parts:
            match = re.match(r"^(\w+)\[(\d+)\]$", part)
            if match:
                key = match.group(1)
                idx = int(match.group(2))
                current = current[key][idx]
            else:
                current = current[part]
        return current

    def get_resolved_value_range(self, field_schema: Dict[str, Any]) -> Any:
        if "value_range_ref" in field_schema:
            return self.resolve_value_range_ref(field_schema["value_range_ref"])
        return field_schema.get("value_range")

    def get_surface_aliases(self, canonical: str) -> List[str]:
        alias_map = self._controlled_vocabs.get("surface_alias_examples", {})
        aliases = alias_map.get(canonical)
        if aliases:
            return aliases
        return [canonical]

    def agent_input_schema(self) -> Dict[str, Any]:
        return self.schema.get("agent_input_schema", {})

    def attachment_schema(self) -> Dict[str, Any]:
        return self.schema.get("attachment_schema", {})

    def plausibility_validation_schema(self) -> Dict[str, Any]:
        return self.schema.get("plausibility_validation_schema", {})

    def image_policy(self, task_type: str) -> Dict[str, Any]:
        cat = self._task_category_map.get(task_type, {})
        return cat.get("image_policy", {"may_have_image": False, "image_probability": 0.0})

    def scenario_frame_input_modes(self, scenario_id: str) -> List[str]:
        frame = self._scenario_frame_map.get(scenario_id, {})
        return frame.get("suggested_input_modes", ["text_only"])

    def scenario_frame_attachment_types(self, scenario_id: str) -> List[str]:
        frame = self._scenario_frame_map.get(scenario_id, {})
        return frame.get("allowed_attachment_types", [])
