"""Classification + condition evaluation + coherence engine — extracted from pipeline.py:1344-1495."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .data_models import CoherenceViolation, CoherenceValidation
from .field_utils import ValueRangeUtils
from .schema_loader import SchemaLoader


class ClassificationEngine:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def compute_non_empty_filter_dimension_count(self, task_type: str, task_dimensions: Dict[str, Any]) -> int:
        rules = self.schema_loader.classification_rules()
        counted = rules.get("place_filter_rank", {}).get("counted_filter_dimensions", [])
        count = 0
        for key in counted:
            if key in task_dimensions and not ValueRangeUtils.is_null_like(task_dimensions.get(key)):
                count += 1
        return count

    def validate(self, task_type: str, task_dimensions: Dict[str, Any]) -> Tuple[bool, List[CoherenceViolation]]:
        violations: List[CoherenceViolation] = []
        rules = self.schema_loader.classification_rules()
        non_empty_count = self.compute_non_empty_filter_dimension_count(task_type, task_dimensions)

        if task_type == "place_discovery":
            rule = rules["place_discovery"]["deterministic_rule"]
            ranking_objective_present = task_dimensions.get("ranking_objective") is not None
            comparison = task_dimensions.get("needs_candidate_comparison")
            if ranking_objective_present or non_empty_count > rule["max_non_empty_filter_dimensions"] or comparison != rule["needs_candidate_comparison"]:
                violations.append(CoherenceViolation(
                    rule_id="classification_place_discovery_mismatch",
                    rule_family="classification_rules", severity="hard",
                    message="place_discovery 的结构化字段不满足其分类规则"))

        if task_type == "place_filter_rank":
            rule = rules["place_filter_rank"]["deterministic_rule"]
            ranking_objective_present = task_dimensions.get("ranking_objective") is not None
            comparison = task_dimensions.get("needs_candidate_comparison") is True
            if not (ranking_objective_present or non_empty_count >= rule["min_non_empty_filter_dimensions"] or comparison):
                violations.append(CoherenceViolation(
                    rule_id="classification_place_filter_rank_mismatch",
                    rule_family="classification_rules", severity="hard",
                    message="place_filter_rank 的结构化字段不满足其分类规则"))

        return len(violations) == 0, violations


class ConditionEvaluator:
    def __init__(self, schema_loader: SchemaLoader, classification_engine: ClassificationEngine):
        self.schema_loader = schema_loader
        self.classification_engine = classification_engine

    def compute_derived(self, task_type: str, scenario_frame: Dict[str, Any],
                        global_context: Dict[str, Any], task_dimensions: Dict[str, Any]) -> Dict[str, Any]:
        default_context = scenario_frame.get("default_context", {}) if scenario_frame else {}
        unlikely_values = scenario_frame.get("unlikely_values", {}) if scenario_frame else {}
        compatible_task_types = scenario_frame.get("compatible_task_types", []) if scenario_frame else []

        hits_unlikely = False
        for field_name, bad_values in unlikely_values.items():
            if global_context.get(field_name) in bad_values:
                hits_unlikely = True
                break

        STRONG_SEMANTIC_FIELDS = {"trip_state", "mobility_context", "familiarity_with_area"}
        conflict_default = False
        for field_name, default_value in default_context.items():
            if field_name not in STRONG_SEMANTIC_FIELDS:
                continue
            current_value = global_context.get(field_name)
            if current_value is not None and current_value != default_value:
                conflict_default = True
                break

        return {
            "non_empty_filter_dimension_count": self.classification_engine.compute_non_empty_filter_dimension_count(task_type, task_dimensions),
            "task_type_in_frame_compatible_task_types": task_type in compatible_task_types if scenario_frame else True,
            "global_context_hits_unlikely_values": hits_unlikely,
            "conflicts_with_frame_default_context": conflict_default,
        }

    def get_value(self, path: str, state: Dict[str, Any]) -> Any:
        parts = path.split(".")
        current = state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def eval_condition(self, condition: Dict[str, Any], state: Dict[str, Any]) -> bool:
        if "all" in condition:
            return all(self.eval_condition(item, state) for item in condition["all"])
        if "any" in condition:
            return any(self.eval_condition(item, state) for item in condition["any"])
        field_path = condition["field"]
        op = condition["op"]
        expected = condition.get("value")
        actual = self.get_value(field_path, state)
        if op == "eq":
            return actual == expected
        if op == "gt":
            return actual is not None and actual > expected
        if op == "lt":
            return actual is not None and actual < expected
        if op == "gte":
            return actual is not None and actual >= expected
        if op == "lte":
            return actual is not None and actual <= expected
        if op == "not_null":
            return actual is not None
        if op == "is_null":
            return actual is None
        if op == "count_lt":
            return isinstance(actual, list) and len(actual) < expected
        if op == "count_gte":
            return isinstance(actual, list) and len(actual) >= expected
        raise ValueError(f"未知 condition op: {op}")


class CoherenceEngine:
    def __init__(self, schema_loader: SchemaLoader, classification_engine: ClassificationEngine):
        self.schema_loader = schema_loader
        self.classification_engine = classification_engine
        self.evaluator = ConditionEvaluator(schema_loader, classification_engine)

    def validate(self, task_type: str, scenario_frame: Dict[str, Any],
                 global_context: Dict[str, Any], task_dimensions: Dict[str, Any]) -> CoherenceValidation:
        rules_config = self.schema_loader.coherence_rules()
        violations: List[CoherenceViolation] = []
        derived = self.evaluator.compute_derived(task_type, scenario_frame, global_context, task_dimensions)
        state = {
            "task_type": task_type, "scenario_frame": scenario_frame,
            "global_context": global_context, "task_dimensions": task_dimensions, "derived": derived,
        }
        for family in rules_config["rule_families"]:
            family_name = family["rule_family"]
            for rule in family["rules"]:
                applies_to = rule.get("applies_to_task_types", ["*"])
                if applies_to != ["*"] and task_type not in applies_to:
                    continue
                if self.evaluator.eval_condition(rule["condition"], state):
                    violations.append(CoherenceViolation(
                        rule_id=rule["rule_id"], rule_family=family_name,
                        severity=rule["severity"], message=rule["message"]))

        hard = [v for v in violations if v.severity == "hard"]
        soft = [v for v in violations if v.severity == "soft"]
        if hard:
            return CoherenceValidation(is_valid=False, severity="hard", violations=violations,
                                       repair_action=rules_config["execution_policy"]["hard_violation_action"])
        if soft:
            return CoherenceValidation(is_valid=False, severity="soft", violations=violations, repair_action="resample_both")
        return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")
