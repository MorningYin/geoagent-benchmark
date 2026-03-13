"""Module 5: Query Writer — natural language rewriting + reverse verification.

Uses qwen-max for rewriting, qwen-plus for reverse verification.
Injects information_gap_plan so the query doesn't leak image-provided info.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import DashScopeClient
from shared.jsonl_io import read_jsonl, write_jsonl, append_error

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "04_quality"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "05_queries"
QUALITY_PATH = DATA_IN / "quality.jsonl"
FINAL_PATH = DATA_OUT / "final.jsonl"
ERRORS_PATH = DATA_OUT / "errors.jsonl"

QUERY_REWRITE_FEW_SHOTS = """以下是几个好的转写示例，请学习这种风格：

示例1:
输入: task_type=place_filter_rank, trip_state=just_landed, weather=rain, mobility=with_luggage,
      place_category=pharmacy, must_be_open_now=true, distance_limit=2000, ranking_objective=nearest
转写: 我刚下飞机，外面在下雨，拖着行李不想走太远，附近两公里内有没有现在还开着的药店？最近的那家在哪？

示例2:
输入: task_type=route_planning, trip_state=after_event, party=friends, travel_mode=DRIVE,
      avoid_highways=true, departure_time_type=arrive_by, arrival_time=23:00
转写: 我们刚看完演出，想晚上十一点前开车回酒店，尽量别走高速，有什么推荐的路线吗？

示例3:
输入: task_type=place_filter_rank, trip_state=exploring, familiarity=first_time_visitor,
      place_category=restaurant, ranking_objective=highest_rated, distance_limit=1000
转写: 第一次来这边，想找附近一公里内评价最好的餐厅吃饭，有推荐吗？

注意以下是差的转写风格，请避免：
- "我要找一个药店，要求营业中，距离2000米以内" → 太像填表
- 把所有字段平铺列出来 → 缺乏叙事感"""


def _build_rewrite_prompt(task: Dict[str, Any]) -> str:
    """Build the rewrite prompt with information gap awareness."""
    info_gap = task.get("information_gap_plan", {})
    must_not_leak = info_gap.get("must_not_leak_in_query", [])
    image_provides = info_gap.get("image_provides", [])

    leak_instruction = ""
    if must_not_leak:
        leak_instruction = (
            "\n## 信息泄漏禁区（绝对不能在 query 中提到）\n"
            f"以下信息不能出现在 query 中：{must_not_leak}\n"
        )

    image_instruction = ""
    if image_provides:
        image_instruction = (
            "\n## 图像上下文\n"
            "用户同时发了一张街景照片。写 query 时可以用'看看这张照片'、'这附近'等口吻，\n"
            f"但不要重复图片已提供的信息：{image_provides}\n"
        )

    prompt = (
        "你是一个自然语言转写器，擅长把结构化地理任务写成真实用户会说的话。\n\n"
        "## 写法规则\n"
        "- verifiable 约束写成明确需求\n"
        "- query_style_only 写成背景叙事\n"
        "- latent_preference 写成偏好暗示\n"
        "- planner_hint 自然融入环境描述\n\n"
        "## 关键要求\n"
        "1. 用中文口语，像真人说话\n"
        "2. 不要列字段名\n"
        "3. 多个约束自然融合\n"
        "4. 只输出一段自然语言 query\n"
        f"{image_instruction}"
        f"{leak_instruction}\n"
        f"## 参考示例\n{QUERY_REWRITE_FEW_SHOTS}\n\n"
        "## 当前样本\n"
        + json.dumps({
            "task_type": task.get("task_type"),
            "global_context": task.get("global_context"),
            "task_dimensions": task.get("task_dimensions"),
        }, ensure_ascii=False, indent=2)
    )
    return prompt


def _reverse_verify(query: str, task: Dict[str, Any], llm: DashScopeClient) -> bool:
    """Use qwen-plus to check if the query leaks answer information."""
    info_gap = task.get("information_gap_plan", {})
    must_not_leak = info_gap.get("must_not_leak_in_query", [])

    prompt = {
        "query": query,
        "must_not_leak": must_not_leak,
        "task_type": task.get("task_type"),
        "instruction": (
            "你是一个信息泄漏检查员。\n"
            "下面是一条用户 query 和不允许出现的信息列表。\n"
            "请检查 query 是否泄漏了这些信息。\n"
            "同时检查 query 是否仅凭文字就能得到完整答案（不需要图片辅助）。\n\n"
            "返回 JSON: {\"leaked\": true/false, \"leak_details\": \"说明\"}"
        ),
    }
    messages = [
        {"role": "system", "content": "你是一个严格的信息泄漏检查员。"},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    result = llm.json_completion(messages, temperature=0.2)
    return not result.get("leaked", False)


def rewrite_query(task: Dict[str, Any], llm: DashScopeClient, max_retries: int = 2) -> str:
    """Rewrite task to natural language, with reverse verification."""
    prompt_text = _build_rewrite_prompt(task)
    messages = [{"role": "user", "content": prompt_text}]

    for attempt in range(max_retries + 1):
        query = llm.rewrite_completion(messages, temperature=0.8)

        if not query.strip():
            continue

        # Reverse verify
        if _reverse_verify(query, task, llm):
            return query.strip()

        # If leaked, add feedback and retry
        messages.append({"role": "assistant", "content": query})
        messages.append({"role": "user", "content": "这个 query 泄漏了答案信息，请重写。不要提及具体的店名、距离数字等。"})

    # Return last attempt even if verification failed
    return query.strip()


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm = DashScopeClient(config)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(QUALITY_PATH)
    if not tasks:
        print("[Module 5] No quality-passed tasks found. Run module 4 first.")
        return FINAL_PATH

    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        image_id = task.get("image_id", "")
        print(f"[Module 5] Rewriting query {i}/{len(tasks)}: {image_id}")

        try:
            query = rewrite_query(task, llm)
            task["natural_language_query"] = query

            # Build agent_input
            task["agent_input"] = {
                "input_mode": "text_plus_photo",
                "user_message": {
                    "primary_text": query,
                    "language": task.get("global_context", {}).get("language_preference", "zh-CN"),
                },
                "device_context": task.get("device_context", {}),
                "attachments": [{
                    "attachment_id": task.get("image_id"),
                    "modality": "photo",
                    "source_type": "street_view_photo",
                    "file_ref": task.get("image_path"),
                    "caption": task.get("vision_parse", {}).get("overall_description", ""),
                    "ocr_texts": task.get("vision_parse", {}).get("ocr_texts", []),
                    "geo_hint": task.get("vision_parse", {}).get("geo_hints"),
                    "task_relevance": {
                        "is_required": True,
                        "supports_fields": ["search_region_value"],
                    },
                }],
            }

            results.append(task)
            print(f"  -> query: {query[:60]}...")
        except Exception as e:
            append_error(ERRORS_PATH, {"image_id": image_id, "error": str(e)})
            print(f"  -> ERROR: {e}")

    count = write_jsonl(FINAL_PATH, results)
    print(f"[Module 5] Done. {count} final samples written to {FINAL_PATH}")
    return FINAL_PATH


if __name__ == "__main__":
    main()
