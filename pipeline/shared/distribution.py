"""Distribution controller for pipeline v4.

Tracks coordinate value counts across generated samples
and produces steering prompts to guide LLM toward underrepresented areas.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class DistributionController:
    """Tracks sample coordinate distributions and suggests generation targets."""

    def __init__(self, task_spec: Dict[str, Any]):
        self.targets: Dict[str, Dict[str, float]] = {}
        for coord_name, coord_spec in task_spec["coordinates"].items():
            self.targets[coord_name] = coord_spec["target_distribution"]
        self.counts: Dict[str, Counter] = {
            coord: Counter() for coord in self.targets
        }

    def record(self, coordinates: Dict[str, str]) -> None:
        """Record a generated sample's coordinates."""
        for coord, value in coordinates.items():
            if coord in self.counts:
                self.counts[coord][value] += 1

    def total(self) -> int:
        """Total number of recorded samples."""
        return max(sum(self.counts["task_type"].values()), 1)

    def steering_prompt(self) -> str:
        """Build a Chinese-language steering block for LLM prompt injection.

        Returns guidance on which coordinate values are under-represented.
        """
        total = self.total()
        lines: List[str] = []
        for coord, target_dist in self.targets.items():
            current = self.counts[coord]
            gaps: List[str] = []
            for val, target_pct in target_dist.items():
                actual_pct = current.get(val, 0) / total
                if actual_pct < target_pct * 0.7:  # >30% below target
                    gaps.append(
                        f"{val}(当前{actual_pct:.0%}, 目标{target_pct:.0%})"
                    )
            if gaps:
                lines.append(f"  {coord}: 缺少 {', '.join(gaps)}")
        if not lines:
            return "当前分布均衡，自由创作。"
        return (
            "分布引导（优先往这些方向创作，但不要牺牲自然性）:\n"
            + "\n".join(lines)
        )

    def suggest_target(self) -> Dict[str, str]:
        """Suggest coordinate values for the next sample (pick most underserved)."""
        suggestion: Dict[str, str] = {}
        total = self.total()
        for coord, target_dist in self.targets.items():
            current = self.counts[coord]
            max_deficit = -1.0
            best_val = list(target_dist.keys())[0]
            for val, target_pct in target_dist.items():
                actual_pct = current.get(val, 0) / total
                deficit = target_pct - actual_pct
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_val = val
            suggestion[coord] = best_val
        return suggestion

    def summary(self) -> str:
        """Return a compact distribution summary string."""
        total = self.total()
        parts: List[str] = []
        for coord, counts in self.counts.items():
            dist_str = ", ".join(
                f"{v}={counts.get(v, 0)}" for v in self.targets[coord]
            )
            parts.append(f"  {coord}: {dist_str}")
        return f"Total samples: {total}\n" + "\n".join(parts)
