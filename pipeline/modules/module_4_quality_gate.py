"""Module 4: Quality Judge — LLM-as-Judge multi-dimensional quality review.

Pipeline v4: replaces v3's 4-layer programmatic quality gate with a single
LLM judge call that scores tasks across 5 dimensions and calibrates coordinates.

Judge dimensions:
  1. Realism        — would a real user ask this?
  2. Image necessity — is the image essential or decorative?
  3. Behavior chain — is the reasoning/API chain logical?
  4. Verifiability  — are the must_pass checks concrete?
  5. Coordinate accuracy — do self-labeled coordinates match?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import ClientFactory, STAGE_TASK_JUDGE
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = PIPELINE_ROOT / "data" / "03_tasks"
DATA_OUT = PIPELINE_ROOT / "data" / "04_judged"
TASKS_PATH = DATA_IN / "tasks.jsonl"
APPROVED_PATH = DATA_OUT / "approved.jsonl"
REJECTED_PATH = DATA_OUT / "rejected.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"


JUDGE_PROMPT_TEMPLATE = """\
你是一个严格的地理 Agent 基准测试审稿人。请认真评审以下任务样本，尽量找出问题。

## 待评审任务

%s

## 原始图像描述

%s

## 评审维度（每项 1-5 分）

1. **场景自然度 (realism)**: 真实用户会在这种情况下提出这种需求吗？
   - 1-2: 场景生硬、刻意、不自然
   - 3: 可能但不太常见
   - 4-5: 非常自然，生活中常见

2. **图像必要性 (image_necessity)**: 去掉图像，Agent 还能完成任务吗？
   - 1: 图像完全是摆设，不影响解题
   - 2: 去掉图只是稍微变难
   - 3: 去掉图明显变难
   - 4-5: 去掉图根本无法完成任务

3. **行为链合理性 (behavior_chain)**: Agent 的推理步骤和 API 调用逻辑通顺吗？
   - 有没有跳步、多余步骤、或技术上不可能的操作？
   - API 调用的参数合理吗？

4. **可验证性 (verifiability)**: must_pass 里的检查条件够具体吗？
   - 每条能客观判断对错吗？还是太模糊（如"做得好"）？

5. **坐标准确性 (coordinate_accuracy)**: task_type / complexity / image_role 标注准确吗？
   - 任务实际复杂度和标注的 complexity 匹配吗？
   - 图像在任务中的实际角色和标注的 image_role 匹配吗？

## 输出格式（严格 JSON）

{"scores": {"realism": N, "image_necessity": N, "behavior_chain": N, "verifiability": N, "coordinate_accuracy": N}, "overall_pass": true/false, "corrected_coordinates": {"task_type": "...", "complexity": "...", "image_role": "..."}, "rejection_reason": "..." 或 null, "review_priority": "low|medium|high", "suggestions": "..."}

## 判断标准

- 所有分 >= 3 → overall_pass = true
- 任一分 <= 1 → overall_pass = false
- 有分为 2 → overall_pass = true，但 review_priority = "high"
- 如果坐标标注不准确，在 corrected_coordinates 给出你认为正确的值

## 长度要求
suggestions 字段请控制在 100 字以内，只列出最关键的 1-2 个问题，不要逐条展开分析。"""


def _build_judge_input(task: Dict[str, Any]) -> str:
    """Build the judge prompt with task context."""
    task_for_judge = {
        "coordinates": task.get("coordinates", {}),
        "scenario": task.get("scenario", {}),
        "expected_behavior": task.get("expected_behavior", {}),
        "verification": task.get("verification", {}),
        "coordinate_reasoning": task.get("coordinate_reasoning", ""),
    }

    scene_desc = task.get("provenance", {}).get("scene_description", "（无描述）")
    task_json_str = json.dumps(task_for_judge, ensure_ascii=False, indent=2)

    return JUDGE_PROMPT_TEMPLATE % (task_json_str, scene_desc)


def _determine_pass(scores: Dict[str, int]) -> tuple[bool, str]:
    """Determine pass/fail and review priority from scores."""
    values = list(scores.values())

    if any(v <= 1 for v in values):
        return False, "high"

    if any(v <= 2 for v in values):
        return True, "high"

    if all(v >= 4 for v in values):
        return True, "low"

    return True, "medium"


def judge_task(task: Dict[str, Any],
               llm) -> Dict[str, Any]:
    """Run LLM judge on a single task. Returns task with judge_report attached."""
    prompt = _build_judge_input(task)

    judge_result = llm.json_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # Extract and validate scores
    scores = judge_result.get("scores", {})
    expected_keys = {"realism", "image_necessity", "behavior_chain",
                     "verifiability", "coordinate_accuracy"}
    for key in expected_keys:
        if key not in scores:
            scores[key] = 3  # default to neutral if missing

    # Determine pass/fail
    overall_pass, review_priority = _determine_pass(scores)

    # Override with LLM's own judgment if it explicitly failed
    if judge_result.get("overall_pass") is False:
        overall_pass = False

    # Apply coordinate corrections if provided
    corrected = judge_result.get("corrected_coordinates", {})
    if corrected and corrected != task.get("coordinates"):
        task["coordinates_original"] = task["coordinates"]
        task["coordinates"] = corrected

    # Attach judge report
    task["judge_report"] = {
        "scores": scores,
        "overall_pass": overall_pass,
        "review_priority": judge_result.get("review_priority", review_priority),
        "rejection_reason": judge_result.get("rejection_reason"),
        "suggestions": judge_result.get("suggestions", ""),
        "coordinates_corrected": corrected != task.get("coordinates_original", task["coordinates"]),
    }

    return task


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = ClientFactory.for_stage(STAGE_TASK_JUDGE, config)

    tasks = read_jsonl(TASKS_PATH)
    if not tasks:
        print("[Module 4] No tasks found. Run module 3 first.")
        return APPROVED_PATH

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    score_sums: Dict[str, float] = {}

    for i, task in enumerate(tasks, 1):
        image_id = task.get("provenance", {}).get("image_id", "unknown")
        coords = task.get("coordinates", {})
        print(
            f"[Module 4] Judging task {i}/{len(tasks)}: {image_id} "
            f"({coords.get('task_type', '?')})"
        )

        try:
            judged = judge_task(task, llm)
            report = judged["judge_report"]
            scores = report["scores"]

            # Accumulate scores for summary
            for k, v in scores.items():
                score_sums[k] = score_sums.get(k, 0) + v

            if report["overall_pass"]:
                approved.append(judged)
                print(
                    f"  -> PASS (priority={report['review_priority']}) "
                    f"scores={scores}"
                )
            else:
                rejected.append(judged)
                print(
                    f"  -> REJECT: {report.get('rejection_reason', 'N/A')} "
                    f"scores={scores}"
                )

        except Exception as e:
            print(f"  -> ERROR: {e}")
            append_error(ERRORS_PATH, {
                "image_id": image_id, "error": str(e),
            })

    # Write results
    approved_count = write_jsonl(APPROVED_PATH, approved)
    rejected_count = write_jsonl(REJECTED_PATH, rejected)

    # Summary
    total = len(approved) + len(rejected)
    print(f"\n[Module 4] Done. {approved_count} approved, {rejected_count} rejected.")
    if total > 0:
        print("[Module 4] Average scores:")
        for k, v in score_sums.items():
            print(f"  {k}: {v / total:.1f}")
        pass_rate = len(approved) / total * 100
        print(f"  Pass rate: {pass_rate:.0f}%")

    return APPROVED_PATH


if __name__ == "__main__":
    main()
