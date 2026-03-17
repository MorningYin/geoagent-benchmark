"""PlausibilityChecker — extracted from pipeline.py:1227-1338, uses new LLM client."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .data_models import CoherenceViolation, CoherenceValidation
from .llm_client import BaseLLMClient


class PlausibilityChecker:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def validate(self, task_type: str, scenario_frame: Dict[str, Any],
                 global_context: Dict[str, Any], task_dimensions: Dict[str, Any],
                 agent_input: Dict[str, Any]) -> CoherenceValidation:
        programmatic_violations = self._check_observation_alignment(agent_input)
        if programmatic_violations:
            return CoherenceValidation(
                is_valid=False, severity="hard",
                violations=programmatic_violations,
                repair_action="resample_both"
            )

        if not self.llm_client.available:
            return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")

        try:
            return self._llm_check(task_type, global_context, task_dimensions, agent_input)
        except Exception:
            return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")

    def _check_observation_alignment(self, agent_input: Dict[str, Any]) -> List[CoherenceViolation]:
        violations: List[CoherenceViolation] = []
        attachments = agent_input.get("attachments", [])
        for att in attachments:
            relevance = att.get("task_relevance", {})
            if relevance.get("is_required", False):
                supports = relevance.get("supports_fields", [])
                if not supports:
                    violations.append(CoherenceViolation(
                        rule_id="required_attachment_without_supported_fields",
                        rule_family="observation_task_alignment",
                        severity="hard",
                        message="标记为必需的附件没有支撑任何任务字段"
                    ))
            caption = att.get("caption", "")
            if not caption or not caption.strip():
                violations.append(CoherenceViolation(
                    rule_id="attachment_caption_empty",
                    rule_family="observation_task_alignment",
                    severity="hard",
                    message="附件的 caption 不能为空"
                ))
        return violations

    def _llm_check(self, task_type: str, global_context: Dict[str, Any],
                    task_dimensions: Dict[str, Any], agent_input: Dict[str, Any]) -> CoherenceValidation:
        td_summary = {k: v for k, v in task_dimensions.items() if v is not None}
        prompt = {
            "task_type": task_type,
            "global_context": global_context,
            "task_dimensions_summary": td_summary,
            "has_attachments": len(agent_input.get("attachments", [])) > 0,
            "input_mode": agent_input.get("input_mode", "text_only"),
            "instruction": (
                "你是一个生活常识审核员。下面是一个地理任务的结构化描述。\n"
                "请判断这个场景作为一个整体是否在现实生活中合理。\n\n"
                "不要检查单个字段是否合法（这已经由程序完成），\n"
                "你只需要判断所有字段组合在一起是否像一个真实的人会遇到的情况。\n\n"
                "常见的不合理组合示例：\n"
                "- 带着婴儿车骑自行车\n"
                "- 带小孩+下大雨+步行去很多景点\n"
                "- 紧急就医却慢慢挑评分最高的\n"
                "- 凌晨三点找公园野餐\n"
                "- 带着大量行李骑自行车\n\n"
                "返回 JSON：{\"plausible\": true/false, \"reason\": \"简要说明\"}"
            ),
        }
        messages = [
            {"role": "system", "content": "你是严格的生活常识审核员，只判断场景是否合理，不做其他事。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        result = self.llm_client.json_completion(messages, temperature=0.3)

        if result.get("plausible", True):
            return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")
        else:
            return CoherenceValidation(
                is_valid=False, severity="hard",
                violations=[CoherenceViolation(
                    rule_id="llm_plausibility_check_failed",
                    rule_family="joint_plausibility",
                    severity="hard",
                    message=result.get("reason", "LLM 判定场景不合理")
                )],
                repair_action="resample_both"
            )
