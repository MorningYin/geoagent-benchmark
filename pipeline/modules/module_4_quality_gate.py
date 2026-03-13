"""Module 4: Quality Gate — three-layer quality check.

Layer 1: Programmatic rules (CoherenceEngine)
Layer 2: Image necessity check (information_gap_plan validity)
Layer 3: LLM plausibility review (qwen-plus)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import DashScopeClient
from shared.schema_loader import SchemaLoader
from shared.coherence import ClassificationEngine, CoherenceEngine
from shared.plausibility import PlausibilityChecker
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "03_tasks"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "04_quality"
TASKS_PATH = DATA_IN / "tasks.jsonl"
QUALITY_PATH = DATA_OUT / "quality.jsonl"
REJECTED_PATH = DATA_OUT / "rejected.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.json"


def _check_image_necessity(task: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 2: Check that the information_gap_plan makes the image actually necessary."""
    info_gap = task.get("information_gap_plan", {})
    image_provides = info_gap.get("image_provides", [])
    query_must_state = info_gap.get("query_must_state", [])

    issues = []

    if not image_provides:
        issues.append("image_provides is empty — image adds no information")

    if len(image_provides) < 2:
        issues.append("image provides too little unique info (< 2 items)")

    if not query_must_state:
        issues.append("query_must_state is empty — task might be trivially solvable without any query")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def _llm_plausibility_check(task: Dict[str, Any], llm: DashScopeClient) -> Dict[str, Any]:
    """Layer 3: LLM checks overall task plausibility."""
    prompt = {
        "task_type": task.get("task_type"),
        "global_context": task.get("global_context"),
        "task_dimensions_summary": {k: v for k, v in task.get("task_dimensions", {}).items() if v is not None},
        "image_scene": task.get("vision_parse", {}).get("scene_type"),
        "information_gap": task.get("information_gap_plan"),
        "instruction": (
            "你是一个 benchmark 质检员。请判断以下地理任务样本是否合理：\n"
            "1. 任务场景是否在现实中可能发生？\n"
            "2. 图像场景和任务类型是否匹配？\n"
            "3. 信息缺口计划是否合理（图像确实能提供某些信息）？\n"
            "4. 约束条件是否有明显矛盾？\n\n"
            "返回 JSON: {\"plausible\": true/false, \"reason\": \"简要说明\", \"review_priority\": 1-5}"
        ),
    }
    messages = [
        {"role": "system", "content": "你是严格的 benchmark 质检员。"},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    result = llm.json_completion(messages, temperature=0.3)
    return result


def check_task(task: Dict[str, Any], schema_loader: SchemaLoader,
               coherence_engine: CoherenceEngine,
               classification_engine: ClassificationEngine,
               llm: DashScopeClient) -> Dict[str, Any]:
    """Run all three quality layers. Returns quality report."""
    report: Dict[str, Any] = {
        "image_id": task.get("image_id"),
        "passed": True,
        "rejection_reasons": [],
        "review_priority": 1,
    }

    # Layer 1: Programmatic coherence
    task_type = task.get("task_type", "place_filter_rank")
    global_context = task.get("global_context", {})
    task_dimensions = task.get("task_dimensions", {})

    cls_ok, cls_violations = classification_engine.validate(task_type, task_dimensions)
    if not cls_ok:
        report["passed"] = False
        report["rejection_reasons"].extend([v.message for v in cls_violations])

    # Use a minimal scenario_frame for coherence check
    scenario_frame = {"scenario_id": "generated", "default_context": {},
                      "likely_overrides": {}, "unlikely_values": {},
                      "compatible_task_types": [task_type]}
    coherence = coherence_engine.validate(task_type, scenario_frame, global_context, task_dimensions)
    if not coherence.is_valid:
        report["passed"] = False
        report["rejection_reasons"].extend([v.message for v in coherence.violations])

    # Layer 2: Image necessity
    img_check = _check_image_necessity(task)
    if not img_check["passed"]:
        report["passed"] = False
        report["rejection_reasons"].extend(img_check["issues"])

    # Layer 3: LLM plausibility (only if previous layers passed, to save API calls)
    if report["passed"] and llm.available:
        try:
            llm_result = _llm_plausibility_check(task, llm)
            if not llm_result.get("plausible", True):
                report["passed"] = False
                report["rejection_reasons"].append(llm_result.get("reason", "LLM 判定不合理"))
            report["review_priority"] = llm_result.get("review_priority", 3)
        except Exception as e:
            # LLM failure doesn't block
            report["review_priority"] = 3

    return report


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = DashScopeClient(config)
    schema_loader = SchemaLoader(SCHEMA_PATH)
    classification_engine = ClassificationEngine(schema_loader)
    coherence_engine = CoherenceEngine(schema_loader, classification_engine)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(TASKS_PATH)
    if not tasks:
        print("[Module 4] No tasks found. Run module 3 first.")
        return QUALITY_PATH

    passed_tasks: List[Dict[str, Any]] = []
    rejected_tasks: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        image_id = task.get("image_id", "")
        print(f"[Module 4] Checking task {i}/{len(tasks)}: {image_id}")

        try:
            report = check_task(task, schema_loader, coherence_engine,
                                classification_engine, llm)
            task["quality_report"] = report

            if report["passed"]:
                passed_tasks.append(task)
                print(f"  -> PASSED (priority={report['review_priority']})")
            else:
                rejected_tasks.append(task)
                print(f"  -> REJECTED: {report['rejection_reasons'][:2]}")
        except Exception as e:
            append_error(ERRORS_PATH, {"image_id": image_id, "error": str(e)})
            print(f"  -> ERROR: {e}")

    write_jsonl(QUALITY_PATH, passed_tasks)
    write_jsonl(REJECTED_PATH, rejected_tasks)
    print(f"[Module 4] Done. {len(passed_tasks)} passed, {len(rejected_tasks)} rejected")
    return QUALITY_PATH


if __name__ == "__main__":
    main()
