"""Answer template builder — extracted from pipeline.py:1517-1561."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .data_models import ConstraintCheckRecord, GeneratedSample
from .schema_loader import SchemaLoader


class AnswerTemplateBuilder:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def build_empty_answer(self, task_type: str) -> Dict[str, Any]:
        template = self.schema_loader.answer_template(task_type)
        return self._build_fields(template.get("fields", []))

    def _build_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field in fields:
            result[field["name"]] = self._empty_value_for_field(field)
        return result

    def _empty_value_for_field(self, field: Dict[str, Any]) -> Any:
        field_type = field["type"]
        if field_type.startswith("array"):
            return []
        if field_type.startswith("object"):
            subfields = field.get("subfields")
            if subfields:
                return self._build_fields(subfields)
            return {}
        return None

    def generate_constraint_check_records(self, task_type: str,
                                          structured_sample: GeneratedSample,
                                          observed_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        field_map = {item["name"]: item for item in self.schema_loader.input_dimensions(task_type)}
        for field_name, field_schema in field_map.items():
            expected = structured_sample.task_dimensions.get(field_name)
            if expected is None:
                continue
            constraint_kind = field_schema.get("constraint_strength", "background")
            context_role = field_schema.get("context_role")
            observed = observed_values.get(field_name)
            is_satisfied = None
            if observed is not None:
                is_satisfied = (observed == expected)
            records.append(asdict(ConstraintCheckRecord(
                constraint_name=field_name, constraint_kind=constraint_kind,
                context_role=context_role, is_satisfied=is_satisfied,
                observed_value=observed, expected_value=expected, notes=None)))
        return records
