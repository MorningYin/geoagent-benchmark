"""Module 5: Query Writer — natural language rewriting + reverse verification.

Uses qwen-max for rewriting, qwen-plus for reverse verification.
Injects information_gap_plan so the query doesn't leak image-provided info.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import PipelineConfig
from shared.llm_client import BaseLLMClient, ClientFactory, STAGE_QUERY_REWRITE, STAGE_QUERY_VERIFY
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

示例4（图片是核心线索 — place_lookup）:
输入: task_type=place_lookup, 图片中可见文字=["星巴克臻选", "南京西路1266号"],
      familiarity=first_time_visitor, query_source=image_ocr
转写: 我刚拍了张照片，上面好像写着"星巴克臻选"，在南京西路上。这是哪家店？能帮我确认一下具体位置吗？

示例5（图片是核心线索 — geocode_resolution）:
输入: task_type=geocode_resolution, 图片中可见文字=["建国门外大街", "永安里"],
      familiarity=uncertain, query_source=image_geo_clue
转写: 我看到路牌上写着"建国门外大街"和"永安里"，这里是北京哪个区？具体地址是什么？

注意以下是差的转写风格，请避免：
- "我要找一个药店，要求营业中，距离2000米以内" → 太像填表
- 把所有字段平铺列出来 → 缺乏叙事感
- "看看这张照片" → 如果图片里有明确文字，要引用具体文字而不是说"看看这张照片"
- 对于 place_lookup / geocode_resolution，必须引用图片中的可见文字作为查询核心"""


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

    # ── Image instruction: varies by utility class ──
    utility_class = task.get("image_utility_class", "context")
    anchor_evidence = task.get("anchor_evidence")
    image_instruction = ""

    if utility_class == "anchor" and anchor_evidence:
        signs = anchor_evidence.get("readable_signs", [])
        geo_hint = anchor_evidence.get("geo_hint", "")
        signs_str = "、".join(signs[:3])
        image_instruction = (
            "\n## 图像上下文（anchor 图 — 图片是核心线索）\n"
            f"用户拍了一张照片，图中可以看到以下文字/标识：{signs_str}\n"
            f"图片暗示的地理位置：{geo_hint}\n"
            "写 query 时请**自然地引用图中可见的文字**（如'我看到路牌上写着XX'、'对面有个XX的招牌'），\n"
            "让图片成为 query 的核心信息来源，而不是可有可无的附件。\n"
            "**不要**用'看看这张照片'这种空洞说法。\n"
        )
    elif image_provides:
        image_instruction = (
            "\n## 图像上下文（context 图 — 图片是补充上下文）\n"
            "用户同时发了一张街景照片。写 query 时可以用'我在这附近'、'看看我周围的环境'等口吻，\n"
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


def _reverse_verify(query: str, task: Dict[str, Any], llm: BaseLLMClient) -> bool:
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


def rewrite_query(task: Dict[str, Any],
                  llm_rewrite: BaseLLMClient,
                  llm_verify: Optional[BaseLLMClient] = None,
                  max_retries: int = 2) -> str:
    """Rewrite task to natural language, with reverse verification.

    Args:
        llm_rewrite: model used for NL rewriting (STAGE_QUERY_REWRITE)
        llm_verify:  model used for leak detection (STAGE_QUERY_VERIFY).
                     Falls back to llm_rewrite if not provided.
    """
    verifier = llm_verify or llm_rewrite
    prompt_text = _build_rewrite_prompt(task)
    messages = [{"role": "user", "content": prompt_text}]
    query = ""

    for attempt in range(max_retries + 1):
        query = llm_rewrite.rewrite_completion(messages, temperature=0.8)

        if not query.strip():
            continue

        # Reverse verify with the designated verifier model
        if _reverse_verify(query, task, verifier):
            return query.strip()

        # If leaked, add feedback and retry
        messages.append({"role": "assistant", "content": query})
        messages.append({"role": "user", "content": "这个 query 泄漏了答案信息，请重写。不要提及具体的店名、距离数字等。"})

    # Return last attempt even if verification failed
    return query.strip()


def main(config: Optional[PipelineConfig] = None) -> Path:
    config = config or PipelineConfig()
    llm_rewrite = ClientFactory.for_stage(STAGE_QUERY_REWRITE, config)
    llm_verify  = ClientFactory.for_stage(STAGE_QUERY_VERIFY, config)

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
            query = rewrite_query(task, llm_rewrite, llm_verify)
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
