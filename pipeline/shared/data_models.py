"""Core data models for pipeline v4.

Defines the GeoTask output format with 3 control coordinates,
free-form scenario, expected behavior chain, and verification rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Coordinates:
    """3-dimensional control coordinates for distribution management."""
    task_type: str      # locate | search | route | plan | judge | adapt
    complexity: str     # simple | moderate | complex
    image_role: str     # primary | supporting | trigger


@dataclass
class ScenarioTurn:
    """A single turn in the user-agent conversation."""
    role: str           # "user"
    content: str
    images: List[str] = field(default_factory=list)


@dataclass
class WriteAction:
    """An expected write operation on an application."""
    app: str            # calendar, favorites, notes, etc.
    action: str         # create, update, delete
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedBehavior:
    """The expected agent behavior chain for solving the task."""
    reasoning: str      # step-by-step reasoning chain
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    write_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Verification:
    """How to verify the agent's answer is correct."""
    must_pass: List[str] = field(default_factory=list)
    answer_reference: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    """Source metadata for traceability."""
    image_id: str
    image_path: str
    location: Dict[str, float]          # {"lat": ..., "lng": ...}
    nearby_pois: List[Dict[str, Any]] = field(default_factory=list)
    scene_description: str = ""
    generated_by: str = ""
    timestamp: str = ""


@dataclass
class GeoTask:
    """Complete benchmark task — the pipeline's primary output unit."""
    coordinates: Coordinates
    scenario: Dict[str, Any]            # narrative + turns
    expected_behavior: ExpectedBehavior
    verification: Verification
    provenance: Provenance
    judge_report: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
