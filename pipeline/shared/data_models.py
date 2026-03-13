"""Core data classes for the pipeline — extracted from pipeline.py with v3 extensions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ConstraintCheckRecord:
    constraint_name: str
    constraint_kind: str
    context_role: Optional[str]
    is_satisfied: Optional[bool]
    observed_value: Any
    expected_value: Any
    notes: Optional[str] = None


@dataclass
class CoherenceViolation:
    rule_id: str
    rule_family: str
    severity: str
    message: str


@dataclass
class CoherenceValidation:
    is_valid: bool
    severity: str
    violations: List[CoherenceViolation] = field(default_factory=list)
    repair_action: str = "accept"


@dataclass
class GeneratedSample:
    benchmark_name: str
    schema_version: str
    task_type: str
    scenario_frame_id: Optional[str]
    global_context: Dict[str, Any]
    task_dimensions: Dict[str, Any]
    agent_input: Optional[Dict[str, Any]]
    natural_language_query: Optional[str]
    coherence_validation: CoherenceValidation
    plausibility_validation: Optional[CoherenceValidation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # --- v3 extensions ---
    image_id: Optional[str] = None
    vision_parse: Optional[Dict[str, Any]] = None
    real_world_context: Optional[Dict[str, Any]] = None
    information_gap_plan: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
