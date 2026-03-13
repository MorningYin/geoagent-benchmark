"""
GeoAgentBench Pipeline — 地理任务 Agent Benchmark 数据生成管线

整体流程：
  1. 采样 task_type 和 scenario_frame
  2. 生成 global_context（优先 LLM 联合生成，失败则程序化 fallback）
  3. 生成 task_dimensions（程序预填枚举字段 + LLM 补充数值/对象字段，失败则程序化 fallback）
  4. 程序化校验 classification_rules + coherence_rules（不调 LLM）
  5. 通过校验后，LLM 转写为自然语言 query（失败则程序模板兜底）

依赖：
  - openai 包（用于调用 DeepSeek API，非必须——没有时全部走程序化 fallback）
  - schema JSON 文件（v1.2 GeoAgentBench-TaskSchema）

模型分工：
  - global_context 生成 + task_dimensions 补全 → deepseek-chat（结构化填充，不需要深度推理）
  - 自然语言 query 转写 → deepseek-reasoner（需要语言能力和场景理解）
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    # 没装 openai 包也能跑，只是 LLM 功能不可用，全部走程序化 fallback
    OpenAI = None


# ============================================================
# 一、基础数据结构
# 这些 dataclass 是整个 pipeline 的数据载体，不含任何业务逻辑
# ============================================================

@dataclass
class ConstraintCheckRecord:
    """
    统一的约束核验记录，对应 schema 中的 constraint_check_record_schema。
    每个字段的含义：
      - constraint_name: 被检查的约束维度名（如 must_be_open_now）
      - constraint_kind: hard / soft / background
      - context_role: verifiable / query_style_only / latent_preference / planner_hint
      - is_satisfied: 该约束是否被满足（None 表示尚未核验）
      - observed_value: API 实际返回的值
      - expected_value: 结构化实例中定义的期望值
    """
    constraint_name: str
    constraint_kind: str
    context_role: Optional[str]
    is_satisfied: Optional[bool]
    observed_value: Any
    expected_value: Any
    notes: Optional[str] = None


@dataclass
class CoherenceViolation:
    """单条一致性违规记录：记录哪条规则被触发、严重程度、以及人类可读的说明"""
    rule_id: str       # 规则 ID，如 walk_distance_upper_bound
    rule_family: str   # 所属规则族：physical_feasibility / context_task_semantic_consistency / scenario_frame_compatibility
    severity: str      # soft（建议重采样）或 hard（必须重采样）
    message: str       # 人类可读的违规说明


@dataclass
class CoherenceValidation:
    """
    一致性校验的汇总结果。
    is_valid=True 表示该样本通过了所有校验，可以进入自然语言转写。
    repair_action 指示失败时的修复策略（如 resample_both / manual_review）。
    """
    is_valid: bool
    severity: str
    violations: List[CoherenceViolation] = field(default_factory=list)
    repair_action: str = "accept"


@dataclass
class GeneratedSample:
    """
    pipeline 的最终产物：一个完整的 benchmark 样本。
    包含三层数据：
      - 潜在任务定义层：global_context + task_dimensions（benchmark 内部真值，agent 看不到）
      - Agent 输入层：agent_input（agent 实际收到的输入，包含文本、环境数据、附件）
      - 校验与元数据层：coherence_validation + plausibility_validation + metadata
    """
    benchmark_name: str
    schema_version: str
    task_type: str
    scenario_frame_id: Optional[str]
    global_context: Dict[str, Any]
    task_dimensions: Dict[str, Any]
    agent_input: Optional[Dict[str, Any]]             # agent 实际收到的输入（device_context + attachments + user_message）
    natural_language_query: Optional[str]               # 兼容字段，内容与 agent_input.user_message.primary_text 一致
    coherence_validation: CoherenceValidation           # 程序化字段规则校验结果
    plausibility_validation: Optional[CoherenceValidation]  # LLM 联合合理性审核结果（复用同一数据结构）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # asdict() 会递归把所有嵌套 dataclass（包括 violations 里的 CoherenceViolation）
        # 全部转成 dict，所以不需要再手动对 violations 调 asdict
        return asdict(self)


# ============================================================
# 二、Schema 读取与解析
# 从 v1.2 JSON 文件加载 schema，并提供各种快捷查询方法。
# 所有下游模块都通过 SchemaLoader 读取 schema 数据，
# 而不是自己去解析 JSON，保证单一数据源。
# ============================================================

class SchemaLoader:
    """
    Schema 读取器：加载 v1.2 JSON 并建立各种查询索引。
    核心职责：
      - 提供 task_types、scenario_frames、input_dimensions 等的快捷访问
      - 解析 value_range_ref 引用（如 controlled_vocabularies.place_category_schema.fields[0].value_range）
      - 提供 surface alias 查询（canonical type → 自然语言别名列表）
    """
    def __init__(self, schema_path: Union[str, Path]):
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"schema 文件不存在: {self.schema_path}")
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self._task_category_map = {item["task_type"]: item for item in self.schema["task_categories"]}
        self._scenario_frame_map = {item["scenario_id"]: item for item in self.schema["scenario_frames"]}
        self._global_context_map = {item["name"]: item for item in self.schema["global_context_schema"]}
        self._input_dimension_map = {
            task_type: {item["name"]: item for item in items}
            for task_type, items in self.schema["input_dimensions"].items()
        }
        self._controlled_vocabs = self.schema.get("controlled_vocabularies", {})

    @property
    def benchmark_name(self) -> str:
        return self.schema["benchmark_name"]

    @property
    def version(self) -> str:
        return self.schema["version"]

    @property
    def task_types(self) -> List[str]:
        return list(self._task_category_map.keys())

    def task_category(self, task_type: str) -> Dict[str, Any]:
        return self._task_category_map[task_type]

    def scenario_frames(self) -> List[Dict[str, Any]]:
        return self.schema["scenario_frames"]

    def scenario_frame(self, scenario_id: str) -> Dict[str, Any]:
        return self._scenario_frame_map[scenario_id]

    def global_context_fields(self) -> List[Dict[str, Any]]:
        return self.schema["global_context_schema"]

    def global_context_field(self, name: str) -> Dict[str, Any]:
        return self._global_context_map[name]

    def input_dimensions(self, task_type: str) -> List[Dict[str, Any]]:
        return self.schema["input_dimensions"][task_type]

    def input_dimension(self, task_type: str, name: str) -> Dict[str, Any]:
        return self._input_dimension_map[task_type][name]

    def answer_template(self, task_type: str) -> Dict[str, Any]:
        return self.schema["standard_answer_templates"][task_type]

    def classification_rules(self) -> Dict[str, Any]:
        return self.schema["classification_rules"]

    def coherence_rules(self) -> Dict[str, Any]:
        return self.schema["coherence_rules"]

    def constraint_check_schema(self) -> Dict[str, Any]:
        return self.schema["constraint_check_record_schema"]

    def resolve_value_range_ref(self, value_range_ref: str) -> Any:
        current: Any = self.schema
        parts = value_range_ref.split(".")
        for part in parts:
            match = re.match(r"^(\w+)\[(\d+)\]$", part)
            if match:
                key = match.group(1)
                idx = int(match.group(2))
                current = current[key][idx]
            else:
                current = current[part]
        return current

    def get_resolved_value_range(self, field_schema: Dict[str, Any]) -> Any:
        if "value_range_ref" in field_schema:
            return self.resolve_value_range_ref(field_schema["value_range_ref"])
        return field_schema.get("value_range")

    def get_surface_aliases(self, canonical: str) -> List[str]:
        alias_map = self._controlled_vocabs.get("surface_alias_examples", {})
        aliases = alias_map.get(canonical)
        if aliases:
            return aliases
        return [canonical]

    def agent_input_schema(self) -> Dict[str, Any]:
        """获取 agent_input 的 schema 定义"""
        return self.schema.get("agent_input_schema", {})

    def attachment_schema(self) -> Dict[str, Any]:
        """获取 attachment 的 schema 定义"""
        return self.schema.get("attachment_schema", {})

    def plausibility_validation_schema(self) -> Dict[str, Any]:
        """获取 plausibility_validation 的 schema 定义"""
        return self.schema.get("plausibility_validation_schema", {})

    def image_policy(self, task_type: str) -> Dict[str, Any]:
        """获取指定任务类型的 image_policy"""
        cat = self._task_category_map.get(task_type, {})
        return cat.get("image_policy", {"may_have_image": False, "image_probability": 0.0})

    def scenario_frame_input_modes(self, scenario_id: str) -> List[str]:
        """获取指定 scenario_frame 的 suggested_input_modes"""
        frame = self._scenario_frame_map.get(scenario_id, {})
        return frame.get("suggested_input_modes", ["text_only"])

    def scenario_frame_attachment_types(self, scenario_id: str) -> List[str]:
        """获取指定 scenario_frame 的 allowed_attachment_types"""
        frame = self._scenario_frame_map.get(scenario_id, {})
        return frame.get("allowed_attachment_types", [])


# ============================================================
# 三、DeepSeek LLM 封装
# 支持双模型：
#   - generation_model (默认 deepseek-chat): 用于结构化数据生成（global_context、task_dimensions）
#   - rewrite_model (默认 deepseek-reasoner): 用于自然语言 query 转写
# deepseek-reasoner 是 think 模型，会先内部推理再输出，
# 适合需要语言能力和场景理解的转写任务。
# ============================================================

class DeepSeekChatClient:
    """
    DeepSeek API 客户端封装，支持按环节选择不同模型。
    generation_model: 结构化生成用，便宜快速
    rewrite_model: 自然语言转写用，质量更高
    """
    def __init__(
        self,
        generation_model: str = "deepseek-chat",
        rewrite_model: str = "deepseek-reasoner",
        api_key_env: str = "DEEPSEEK_API_KEY",
    ):
        self.generation_model = generation_model
        self.rewrite_model = rewrite_model
        self.api_key = os.getenv(api_key_env)
        self.client = None
        if self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    @property
    def available(self) -> bool:
        return self.client is not None

    def json_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """结构化数据生成：用 generation_model，强制 JSON 输出"""
        if not self.available:
            raise RuntimeError("DeepSeek 客户端不可用：请设置 DEEPSEEK_API_KEY 且安装 openai 包")
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def text_completion(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> str:
        """普通文本生成：用 generation_model"""
        if not self.available:
            raise RuntimeError("DeepSeek 客户端不可用：请设置 DEEPSEEK_API_KEY 且安装 openai 包")
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def rewrite_completion(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> str:
        """
        自然语言转写：用 rewrite_model（默认 deepseek-reasoner）。
        think 模型会先内部推理"这个场景下用户会怎么说话"，再输出自然语言，
        产出的口语化程度和信息密度会明显更好。
        """
        if not self.available:
            raise RuntimeError("DeepSeek 客户端不可用：请设置 DEEPSEEK_API_KEY 且安装 openai 包")
        response = self.client.chat.completions.create(
            model=self.rewrite_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ============================================================
# 四、通用工具
# ValueRangeUtils：解析 schema 中各种格式的 value_range 表达式，
#   并提供随机值生成（坐标、时间、字符串等）
# ============================================================

class ValueRangeUtils:
    """
    解析 schema 中的 value_range 表达式。
    schema 里的 value_range 格式多样，例如：
      - 纯列表：["DRIVE", "WALK", "BICYCLE"]  → 直接枚举
      - 整数范围字符串："1-20"  → 随机整数
      - 带 null 的范围："null or 50-100000"  → 可能返回 None
      - 字符长度："1-120 chars natural language"  → 随机字符串
      - 语义描述："ISO-8601 datetime"  → 随机时间
    这个类负责把这些格式统一解析，供 ValueSampler 和 FieldValidator 使用。
    """
    INT_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
    NULL_OR_INT_RANGE_RE = re.compile(r"^null\s+or\s+(\d+)-(\d+)$")
    NULL_OR_FLOAT_RANGE_RE = re.compile(r"^null\s+or\s+([0-9.]+)-([0-9.]+)$")
    CHARS_RE = re.compile(r"^(\d+)-(\d+)\s+chars\s+natural\s+language$")

    @staticmethod
    def parse_simple_range(expr: Any) -> Optional[Tuple[float, float, bool]]:
        if isinstance(expr, list):
            return None
        if not isinstance(expr, str):
            return None
        text = expr.strip()
        m = ValueRangeUtils.INT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), False
        m = ValueRangeUtils.NULL_OR_INT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), True
        m = ValueRangeUtils.NULL_OR_FLOAT_RANGE_RE.match(text)
        if m:
            return float(m.group(1)), float(m.group(2)), True
        return None

    @staticmethod
    def parse_char_length(expr: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(expr, str):
            return None
        m = ValueRangeUtils.CHARS_RE.match(expr.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    @staticmethod
    def random_string(min_len: int = 4, max_len: int = 20) -> str:
        length = random.randint(min_len, max_len)
        alphabet = string.ascii_lowercase
        return "".join(random.choice(alphabet) for _ in range(length))

    @staticmethod
    def random_iso_datetime() -> str:
        base = datetime.now(timezone.utc) + timedelta(days=random.randint(-30, 30))
        dt = base.replace(
            hour=random.randint(0, 23),
            minute=random.choice([0, 10, 15, 20, 30, 40, 45, 50]),
            second=0,
            microsecond=0,
        )
        return dt.isoformat()

    @staticmethod
    def random_language_code() -> str:
        return random.choice(["zh-CN", "en", "ja", "fr", "de", "es", "ko"])

    @staticmethod
    def random_coordinate(lat_range: Tuple[float, float] = (-60, 60), lng_range: Tuple[float, float] = (-170, 170)) -> Dict[str, float]:
        return {
            "lat": round(random.uniform(*lat_range), 6),
            "lng": round(random.uniform(*lng_range), 6),
        }

    @staticmethod
    def random_location_object() -> Dict[str, Any]:
        input_type = random.choice(["place_id", "latlng", "address", "text_query"])
        if input_type == "place_id":
            value = f"place_{ValueRangeUtils.random_string(8, 12)}"
        elif input_type == "latlng":
            value = ValueRangeUtils.random_coordinate()
        elif input_type == "address":
            value = f"{random.randint(1, 999)} {ValueRangeUtils.random_string(5, 10)} street"
        else:
            value = ValueRangeUtils.random_string(6, 16)
        return {"input_type": input_type, "value": value}

    @staticmethod
    def random_bbox() -> Dict[str, float]:
        c = ValueRangeUtils.random_coordinate()
        return {
            "south": round(c["lat"] - 0.05, 6),
            "west": round(c["lng"] - 0.05, 6),
            "north": round(c["lat"] + 0.05, 6),
            "east": round(c["lng"] + 0.05, 6),
        }

    @staticmethod
    def random_center_radius() -> Dict[str, float]:
        c = ValueRangeUtils.random_coordinate()
        return {"center_lat": c["lat"], "center_lng": c["lng"], "radius_m": random.randint(100, 10000)}

    @staticmethod
    def is_null_like(value: Any) -> bool:
        return value is None or value == [] or value == {} or value == ""


# ============================================================
# 五、字段采样与校验
# ============================================================

class ValueSampler:
    """
    通用字段采样器：根据 schema 中的 data_type 和 value_range，
    为任意字段生成一个合法的随机值。
    """
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def sample_from_value_range(self, field_schema: Dict[str, Any], task_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Any:
        value_range = self.schema_loader.get_resolved_value_range(field_schema)
        data_type = field_schema["data_type"]
        name = field_schema["name"]
        context = context or {}

        if isinstance(value_range, list):
            return random.choice(value_range)

        simple_range = ValueRangeUtils.parse_simple_range(value_range)
        if data_type in {"integer", "integer_or_null", "number_or_null"} and simple_range is not None:
            low, high, allow_null = simple_range
            if allow_null and random.random() < 0.35:
                return None
            if data_type.startswith("integer"):
                return int(random.randint(int(low), int(high)))
            return round(random.uniform(low, high), 2)

        char_range = ValueRangeUtils.parse_char_length(value_range)
        if data_type == "string" and char_range is not None:
            min_len, max_len = char_range
            return ValueRangeUtils.random_string(min_len=min_len, max_len=min(24, max_len))

        if isinstance(value_range, str):
            text = value_range.strip()

            if data_type == "string_or_null":
                if random.random() < 0.3:
                    return None
                if "ISO-8601 datetime" in text:
                    return ValueRangeUtils.random_iso_datetime()
                if "BCP-47 language code" in text:
                    return ValueRangeUtils.random_language_code()
                if "brand/business chain name" in text:
                    return random.choice(["BrandA", "BrandB", "BrandC"])
                if "ccTLD" in text or "country code" in text:
                    return random.choice(["cn", "us", "jp", "de", "fr"])
                return ValueRangeUtils.random_string(4, 12)

            if data_type == "object":
                return self._sample_object_field(name, value_range, context)

            if data_type == "object_or_null":
                if random.random() < 0.35:
                    return None
                return self._sample_object_field(name, value_range, context)

            if data_type == "string_or_object":
                return self._sample_string_or_object(name, value_range)

            if data_type == "array<object>":
                return self._sample_array_object_field(name, value_range, context)

            if data_type == "array<string>":
                return self._sample_array_string_field(name, value_range)

            if data_type == "string":
                if "ISO-8601 datetime" in text:
                    return ValueRangeUtils.random_iso_datetime()
                if "snake_case identifier" in text:
                    return f"sample_{ValueRangeUtils.random_string(5, 8)}"
                return ValueRangeUtils.random_string(4, 16)

        if data_type == "boolean":
            return random.choice([True, False])

        if data_type == "enum":
            raise ValueError(f"enum 字段必须是 list value_range，字段={name}")

        raise ValueError(f"无法为字段采样: name={name}, data_type={data_type}, value_range={value_range}")

    def _sample_object_field(self, field_name: str, value_range: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text = value_range.strip()
        if text == "{input_type, value}":
            return ValueRangeUtils.random_location_object()
        if "{center_lat, center_lng, radius_m}" in text or "location object" in text:
            return ValueRangeUtils.random_center_radius()
        if "bbox" in text:
            if field_name in {"search_region_value", "meeting_region_value", "spatial_bias_value"}:
                if random.random() < 0.4:
                    return ValueRangeUtils.random_bbox()
                return ValueRangeUtils.random_center_radius()
        if "user_location" in text:
            return {"user_location": ValueRangeUtils.random_coordinate(), "radius_m": random.randint(100, 5000)}
        if "route_polyline_ref" in text:
            return {"route_polyline_ref": f"poly_{ValueRangeUtils.random_string(8, 12)}"}
        if "region object" in text:
            if random.random() < 0.5:
                return {"city_name": random.choice(["Beijing", "Shanghai", "Tokyo", "Paris", "New York"])}
            return ValueRangeUtils.random_bbox()
        return {"value": ValueRangeUtils.random_string(6, 12)}

    def _sample_string_or_object(self, field_name: str, value_range: str) -> Any:
        text = value_range.strip()
        if "address string" in text or "place_id string" in text:
            choice = random.choice(["address", "latlng", "place_id"])
            if choice == "address":
                return f"{random.randint(1, 999)} {ValueRangeUtils.random_string(5, 10)} road"
            if choice == "place_id":
                return f"place_{ValueRangeUtils.random_string(8, 12)}"
            return ValueRangeUtils.random_coordinate()
        return ValueRangeUtils.random_string(5, 12)

    def _sample_array_object_field(self, field_name: str, value_range: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = value_range.strip()
        m = re.match(r"^(\d+)-(\d+)\s+.*objects$", text)
        if not m:
            count = 2
        else:
            low, high = int(m.group(1)), int(m.group(2))
            count = random.randint(low, min(high, low + 3))
        items: List[Dict[str, Any]] = []
        for idx in range(count):
            if field_name in {"candidate_destinations", "origins"}:
                items.append(ValueRangeUtils.random_location_object())
            elif field_name == "candidate_goals":
                items.append({"goal_id": f"goal_{idx+1}", "goal_type": random.choice(["place", "category"]), "value": ValueRangeUtils.random_location_object()})
            else:
                items.append({"value": ValueRangeUtils.random_string(5, 10)})
        return items

    def _sample_array_string_field(self, field_name: str, value_range: str) -> List[str]:
        text = value_range.strip()
        m = re.match(r"^(\d+)-(\d+)\s+goal\s+ids$", text)
        if m:
            low, high = int(m.group(1)), int(m.group(2))
            count = random.randint(low, min(high, 3))
            return [f"goal_{i+1}" for i in range(count)]
        return [ValueRangeUtils.random_string(5, 8) for _ in range(random.randint(1, 3))]


class FieldValidator:
    """
    字段校验器：检查一个具体值是否符合 schema 定义的 data_type 和 value_range。
    """
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def validate_value(self, field_schema: Dict[str, Any], value: Any) -> bool:
        data_type = field_schema["data_type"]
        value_range = self.schema_loader.get_resolved_value_range(field_schema)

        if isinstance(value_range, list):
            return value in value_range
        if data_type == "boolean":
            return isinstance(value, bool)
        if data_type == "string":
            if not isinstance(value, str):
                return False
            char_range = ValueRangeUtils.parse_char_length(value_range)
            if char_range is not None:
                return char_range[0] <= len(value) <= char_range[1]
            if isinstance(value_range, str) and "ISO-8601 datetime" in value_range:
                return self._is_iso_datetime(value)
            return True
        if data_type == "string_or_null":
            if value is None:
                return True
            if not isinstance(value, str):
                return False
            if isinstance(value_range, str) and "ISO-8601 datetime" in value_range:
                return self._is_iso_datetime(value)
            return True
        if data_type == "integer":
            return isinstance(value, int) and self._validate_numeric_range(value_range, value)
        if data_type == "integer_or_null":
            return value is None or (isinstance(value, int) and self._validate_numeric_range(value_range, value))
        if data_type == "number_or_null":
            return value is None or (isinstance(value, (int, float)) and self._validate_numeric_range(value_range, float(value)))
        if data_type in {"enum", "enum_or_null", "boolean_or_null"}:
            return value in value_range
        if data_type == "object":
            return isinstance(value, dict)
        if data_type == "object_or_null":
            return value is None or isinstance(value, dict)
        if data_type == "string_or_object":
            return isinstance(value, (str, dict))
        if data_type == "array<object>":
            return isinstance(value, list) and all(isinstance(i, dict) for i in value)
        if data_type == "array<string>":
            return isinstance(value, list) and all(isinstance(i, str) for i in value)
        return True

    def _is_iso_datetime(self, text: str) -> bool:
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return True
        except Exception:
            return False

    def _validate_numeric_range(self, expr: Any, value: float) -> bool:
        simple_range = ValueRangeUtils.parse_simple_range(expr)
        if simple_range is None:
            return True
        low, high, _allow_null = simple_range
        return low <= value <= high

    def validate_record(self, field_schemas: List[Dict[str, Any]], record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for field_schema in field_schemas:
            name = field_schema["name"]
            required = field_schema.get("required", False)
            if required and name not in record:
                errors.append(f"缺少必填字段: {name}")
                continue
            if name in record and not self.validate_value(field_schema, record[name]):
                errors.append(f"字段不合法: {name}, value={record[name]!r}")
        return len(errors) == 0, errors


# ============================================================
# 六、LLM 生成器
# 三个环节各自的 prompt 策略：
#   - GlobalContextLLMGenerator: 简洁 prompt，deepseek-chat 够用
#   - TaskDimensionsLLMGenerator: 加了"使用真实地点"和"数值与上下文语义一致"约束
#   - QueryRewriter: 用 deepseek-reasoner + few-shot 示例 + context_role 写法指导
# ============================================================

class GlobalContextLLMGenerator:
    """用 LLM 联合生成 global_context。用 deepseek-chat，不需要深度推理。"""
    def __init__(self, schema_loader: SchemaLoader, llm_client: DeepSeekChatClient):
        self.schema_loader = schema_loader
        self.llm_client = llm_client

    def generate(self, task_type: str, scenario_frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm_client.available:
            raise RuntimeError("LLM 不可用")
        fields = self.schema_loader.global_context_fields()
        prompt = {
            "task_type": task_type,
            "scenario_frame": scenario_frame,
            "global_context_schema": fields,
            "instruction": (
                "你要基于 scenario_frame 联合生成 global_context。"
                "必须返回一个 JSON 对象，键必须覆盖 schema 里的所有 global_context 字段。"
                "优先尊重 default_context，并从 likely_overrides 中采样更自然的组合。"
                "不要输出解释，只输出 JSON。"
            ),
        }
        messages = [
            {"role": "system", "content": "你是严格遵守 schema 的 benchmark 数据生成器。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        return self.llm_client.json_completion(messages, temperature=0.8)


class TaskDimensionsLLMGenerator:
    """
    用 LLM 补全 task_dimensions 中的数值和对象字段。
    prompt 加了三条关键约束：
      1. 使用真实存在的地点名称和坐标，不要编造
      2. 数值字段要和 global_context 语义一致（如 urgency=high 时 distance_limit 应放宽）
      3. search_region_value 使用真实城市坐标
    """
    def __init__(self, schema_loader: SchemaLoader, llm_client: DeepSeekChatClient):
        self.schema_loader = schema_loader
        self.llm_client = llm_client

    def generate_partial(self, task_type: str, global_context: Dict[str, Any], base_dimensions: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm_client.available:
            raise RuntimeError("LLM 不可用")
        field_schemas = self.schema_loader.input_dimensions(task_type)
        prompt = {
            "task_type": task_type,
            "global_context": global_context,
            "task_dimensions_schema": field_schemas,
            "pre_filled_dimensions": base_dimensions,
            "instruction": (
                "你需要在已预采样的枚举字段基础上，补全 task_dimensions 中的数值、对象、可选字段。\n"
                "必须返回完整的 task_dimensions JSON 对象。不允许改动已预填的枚举值。\n\n"
                "关键要求：\n"
                "1. 所有地点名称、地址、坐标必须是真实存在的，不要编造。"
                "比如 origin/destination 用真实的机场名、火车站名、景点名、街道地址。\n"
                "2. 数值字段必须和 global_context 语义一致：\n"
                "   - urgency=high 时，distance_limit 和 travel_time_limit 应该放宽而不是收紧\n"
                "   - mobility=with_luggage/stroller 时，距离应偏短\n"
                "   - weather=rain/snow 时，步行距离应更短\n"
                "3. search_region_value 必须使用真实城市的坐标（如北京39.9/116.4、上海31.2/121.5），不要随机坐标。\n"
                "4. candidate_goals / candidate_destinations / origins 里的每个地点都必须是真实的。\n\n"
                "只输出 JSON。"
            ),
        }
        messages = [
            {"role": "system", "content": "你是严格遵守 schema 的结构化字段填充器。你对全球主要城市的地理信息非常熟悉。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        return self.llm_client.json_completion(messages, temperature=0.8)


# --- 自然语言转写的 few-shot 示例 ---
# 这些示例教 LLM "好的转写"长什么样：
#   - 叙事自然，像真人说话而不是填表
#   - verifiable 约束写成明确需求
#   - query_style_only 写成背景叙事
#   - latent_preference 写成偏好暗示
QUERY_REWRITE_FEW_SHOTS = """
以下是几个好的转写示例，请学习这种风格：

示例1:
输入: task_type=place_filter_rank, trip_state=just_landed, weather=rain, mobility=with_luggage,
      place_category=pharmacy, must_be_open_now=true, distance_limit=2000, ranking_objective=nearest
转写: 我刚下飞机，外面在下雨，拖着行李不想走太远，附近两公里内有没有现在还开着的药店？最近的那家在哪？

示例2:
输入: task_type=route_planning, trip_state=after_event, party=friends, travel_mode=DRIVE,
      avoid_highways=true, departure_time_type=arrive_by, arrival_time=23:00
转写: 我们刚看完演出，想晚上十一点前开车回酒店，尽量别走高速，有什么推荐的路线吗？

示例3:
输入: task_type=meeting_point, party=friends, origins=[浦东机场, 虹桥火车站],
      place_category=restaurant, fairness=balanced_duration_variance
转写: 我们几个人一个在浦东机场，一个在虹桥火车站，想找个大家过去都差不多远的餐厅碰头，有推荐吗？

示例4:
输入: task_type=itinerary_planning, trip_state=touring, familiarity=first_time_visitor,
      candidate_goals=[故宫, 天安门, 天坛, 颐和园], time_budget=480min, travel_mode=WALK
转写: 第一次来北京，今天想走路逛一圈，故宫、天安门、天坛、颐和园这几个地方，八个小时能安排过来吗？

注意以下是差的转写风格，请避免：
- "我要找一个药店，要求营业中，距离2000米以内" → 太像填表，不像人话
- "请帮我规划路线，起点A终点B，模式驾车" → 太机械
- 把所有字段平铺列出来 → 缺乏叙事感
""".strip()


class QueryRewriter:
    """
    自然语言转写器：把结构化实例转成真实用户口语。
    使用 deepseek-reasoner（think 模型），配合 few-shot 示例和 context_role 写法指导。
    """
    def __init__(self, schema_loader: SchemaLoader, llm_client: DeepSeekChatClient):
        self.schema_loader = schema_loader
        self.llm_client = llm_client

    def rewrite(self, task_type: str, global_context: Dict[str, Any],
                task_dimensions: Dict[str, Any], agent_input: Optional[Dict[str, Any]] = None) -> str:
        if not self.llm_client.available:
            raise RuntimeError("LLM 不可用，无法转写自然语言 query")

        task_meta = self.schema_loader.task_category(task_type)

        # --- 根据是否有 attachment 调整 prompt ---
        attachments = (agent_input or {}).get("attachments", [])
        has_attachment = len(attachments) > 0

        attachment_instruction = ""
        if has_attachment:
            att = attachments[0]
            caption = att.get("caption", "")
            ocr = att.get("ocr_texts", [])
            is_required = att.get("task_relevance", {}).get("is_required", False)
            attachment_instruction = (
                "\n## 多模态信息\n"
                f"用户同时发了一张图片。图片内容：{caption}\n"
                f"图中可读文字：{ocr}\n"
            )
            if is_required:
                attachment_instruction += (
                    "这张图是解题的关键线索。query 里不要重复图片已经包含的信息，\n"
                    "而是说类似'看看这张照片'、'这是哪'、'这附近有没有...'的口吻。\n"
                )
            else:
                attachment_instruction += (
                    "这张图提供了位置上下文。query 里可以说'我就在这附近'、'我拍了张照片'，\n"
                    "但主要需求还是要用文字说清楚。\n"
                )

        instruction = (
            "你是一个自然语言转写器，擅长把结构化地理任务写成真实用户会说的话。\n\n"
            "## 写法规则\n"
            "根据每个字段的 context_role 决定怎么写进 query：\n"
            "- verifiable（如 must_be_open_now, distance_limit_m, min_rating）→ 写成明确的需求或条件\n"
            "- query_style_only（如 trip_state, user_party_composition）→ 写成叙事背景，自然带出\n"
            "- latent_preference（如 mobility_context, urgency_level）→ 写成偏好暗示，不要直接说字段名\n"
            "- planner_hint（如 time_semantics, weather_context）→ 自然融入时间和环境描述\n\n"
            "## 关键要求\n"
            "1. 用中文口语，像真人在跟朋友或助手说话\n"
            "2. 不要列举字段名，不要像填表\n"
            "3. 把多个约束自然地融合进一两句话里\n"
            "4. 只输出一段自然语言 query，不要解释\n"
            f"{attachment_instruction}\n"
            f"## 参考示例\n{QUERY_REWRITE_FEW_SHOTS}\n\n"
            "## 当前样本\n"
        )

        sample_data = {
            "task_type": task_type,
            "task_definition": task_meta.get("definition", ""),
            "global_context": global_context,
            "task_dimensions": task_dimensions,
        }

        messages = [
            {"role": "user", "content": instruction + json.dumps(sample_data, ensure_ascii=False, indent=2)},
        ]

        return self.llm_client.rewrite_completion(messages, temperature=0.8)


# ============================================================
# 七、程序化 Fallback 生成器
# ============================================================

QUERY_TEXT_POOL = [
    "故宫博物院", "东京塔", "Eiffel Tower", "Central Park", "大英博物馆",
    "北京南站", "浦东国际机场", "西湖", "华山", "泰山",
    "星巴克万达广场店", "全家便利店人民路店", "海底捞火锅",
    "上海迪士尼", "Sydney Opera House", "Statue of Liberty",
    "天安门广场", "外滩", "三里屯", "南京路步行街",
    "首都国际机场T3航站楼", "广州塔", "成都大熊猫繁育研究基地",
    "杭州西溪湿地", "武汉黄鹤楼", "西安钟楼",
]

class ProgrammaticFallbackGenerator:
    """程序化 fallback 生成器。"""
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader
        self.sampler = ValueSampler(schema_loader)

    def generate_global_context(self, task_type: str, scenario_frame: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        default_context = copy.deepcopy(scenario_frame.get("default_context", {}))
        likely_overrides = copy.deepcopy(scenario_frame.get("likely_overrides", {}))
        for field_schema in self.schema_loader.global_context_fields():
            name = field_schema["name"]
            if name in default_context:
                result[name] = default_context[name]
                continue
            if name in likely_overrides and isinstance(likely_overrides[name], list) and likely_overrides[name]:
                result[name] = random.choice(likely_overrides[name])
                continue
            result[name] = self.sampler.sample_from_value_range(field_schema, task_type=task_type)
        return result

    def prefill_task_dimensions_explicit(self, task_type: str, global_context: Dict[str, Any],
                                          scenario_frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        likely_categories = None
        if scenario_frame:
            likely_categories = scenario_frame.get("likely_place_categories")
        counted_filter_dims = self.schema_loader.classification_rules().get(
            "place_filter_rank", {}
        ).get("counted_filter_dimensions", [])
        discovery_filter_count = 0

        for field_schema in self.schema_loader.input_dimensions(task_type):
            value_range = self.schema_loader.get_resolved_value_range(field_schema)
            name = field_schema["name"]
            if not isinstance(value_range, list):
                continue
            if name == "place_category_canonical" and likely_categories:
                valid_choices = [c for c in likely_categories if c in value_range]
                if valid_choices:
                    result[name] = random.choice(valid_choices)
                    continue
            if name == "place_category_surface_form":
                continue
            if task_type == "place_filter_rank" and name == "needs_candidate_comparison":
                result[name] = True
                continue
            if task_type == "place_discovery" and name == "needs_candidate_comparison":
                result[name] = False
                continue
            if task_type == "place_discovery" and name in counted_filter_dims:
                if discovery_filter_count >= 1:
                    result[name] = None if None in value_range else value_range[0]
                    continue
                sampled = random.choice(value_range)
                if sampled is not None and sampled != "none":
                    discovery_filter_count += 1
                result[name] = sampled
                continue
            result[name] = random.choice(value_range)

        if "place_category_canonical" in result:
            aliases = self.schema_loader.get_surface_aliases(result["place_category_canonical"])
            result["place_category_surface_form"] = random.choice(aliases)
        return result

    def fill_missing_task_dimensions(self, task_type: str, current: Dict[str, Any], global_context: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(current)
        for field_schema in self.schema_loader.input_dimensions(task_type):
            name = field_schema["name"]
            if name in result:
                continue
            if name == "query_text":
                result[name] = random.choice(QUERY_TEXT_POOL)
                continue
            if name == "place_category_surface_form" and "place_category_canonical" in result:
                aliases = self.schema_loader.get_surface_aliases(result["place_category_canonical"])
                result[name] = random.choice(aliases)
                continue
            result[name] = self.sampler.sample_from_value_range(field_schema, task_type=task_type, context={"global_context": global_context, "task_dimensions": result})
        return result


# ============================================================
# 7.5 Agent Input 生成器
# 从 latent task（global_context + task_dimensions）生成 agent 实际收到的输入。
# 包括三部分：
#   1. device_context：GPS 定位、系统时间、设备语言（程序化映射）
#   2. attachments：伪多模态图像描述（根据 image_policy 决定是否生成）
#   3. input_mode：根据是否有 attachments 决定
# ============================================================

class AgentInputGenerator:
    """
    从 latent task 生成 agent_input。
    device_context 完全程序化（从 global_context 和 task_dimensions 提取）。
    attachments 由 AttachmentStubGenerator 生成（可能调 LLM）。
    """
    def __init__(self, schema_loader: SchemaLoader, attachment_generator: 'AttachmentStubGenerator'):
        self.schema_loader = schema_loader
        self.attachment_generator = attachment_generator

    def generate(self, task_type: str, scenario_frame: Dict[str, Any],
                 global_context: Dict[str, Any], task_dimensions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """返回 (agent_input, metadata)"""
        metadata = {"errors": []}

        # --- 1. 生成 device_context（纯程序化映射）---
        device_context = self._build_device_context(task_type, global_context, task_dimensions)

        # --- 2. 决定是否生成 attachments ---
        image_policy = self.schema_loader.image_policy(task_type)
        scenario_id = scenario_frame.get("scenario_id", "")
        allowed_types = self.schema_loader.scenario_frame_attachment_types(scenario_id)
        suggested_modes = self.schema_loader.scenario_frame_input_modes(scenario_id)

        attachments = []
        has_image = False

        if image_policy.get("may_have_image", False) and random.random() < image_policy.get("image_probability", 0.0):
            # 场景框架也允许图像类输入
            if any(m != "text_only" for m in suggested_modes) and allowed_types:
                try:
                    source_type = random.choice(allowed_types)
                    stub = self.attachment_generator.generate(
                        task_type, scenario_frame, global_context, task_dimensions, source_type
                    )
                    attachments.append(stub)
                    has_image = True
                except Exception as e:
                    metadata["errors"].append(f"attachment 生成失败: {e}")

        # --- 3. 决定 input_mode ---
        if has_image:
            source = attachments[0].get("source_type", "")
            if "screenshot" in source:
                input_mode = "text_plus_screenshot"
            else:
                input_mode = "text_plus_photo"
        else:
            input_mode = "text_only"

        agent_input = {
            "input_mode": input_mode,
            "user_message": {
                "primary_text": None,  # 后续由 QueryRewriter 填充
                "language": global_context.get("language_preference", "zh-CN"),
            },
            "device_context": device_context,
            "attachments": attachments,
        }

        return agent_input, metadata

    def _build_device_context(self, task_type: str, global_context: Dict[str, Any],
                               task_dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 global_context 和 task_dimensions 程序化提取 device_context。
        GPS 坐标按 task_type 从不同字段提取。
        """
        # --- GPS ---
        gps = self._extract_gps(task_type, task_dimensions)

        # --- 系统时间 ---
        current_time = global_context.get("request_time")

        # --- 设备语言 ---
        device_language = global_context.get("language_preference", "zh-CN")

        return {
            "gps": gps,
            "current_time": current_time,
            "device_language": device_language,
        }

    def _extract_gps(self, task_type: str, task_dimensions: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        按 task_type 从 task_dimensions 的不同字段提取 GPS 坐标。
        某些任务可能没有坐标（如 place_lookup 的 spatial_bias_value 为 null），返回 None。
        """
        # place_discovery / place_filter_rank → search_region_value 的中心坐标
        if task_type in ("place_discovery", "place_filter_rank"):
            region = task_dimensions.get("search_region_value")
            return self._extract_coord_from_region(region)

        # route_planning → origin 的坐标
        if task_type == "route_planning":
            return self._extract_coord_from_location_obj(task_dimensions.get("origin"))

        # route_choice → origin 的坐标
        if task_type == "route_choice":
            return self._extract_coord_from_location_obj(task_dimensions.get("origin"))

        # meeting_point → origins 列表里第一个的坐标
        if task_type == "meeting_point":
            origins = task_dimensions.get("origins", [])
            if origins:
                return self._extract_coord_from_location_obj(origins[0])
            return None

        # place_lookup → spatial_bias_value 的坐标（可能为 null）
        if task_type == "place_lookup":
            return self._extract_coord_from_region(task_dimensions.get("spatial_bias_value"))

        # itinerary_planning → origin 的坐标
        if task_type == "itinerary_planning":
            return self._extract_coord_from_location_obj(task_dimensions.get("origin"))

        # geocode_resolution → 通常不需要 GPS
        return None

    @staticmethod
    def _extract_coord_from_region(region: Any) -> Optional[Dict[str, float]]:
        """从搜索区域对象中提取坐标"""
        if region is None:
            return None
        if isinstance(region, dict):
            # {center_lat, center_lng, ...} 格式
            if "center_lat" in region and "center_lng" in region:
                return {"lat": region["center_lat"], "lng": region["center_lng"]}
            # {center: {lat, lng}, ...} 格式（LLM 可能返回这种）
            center = region.get("center")
            if isinstance(center, dict) and "lat" in center:
                return {"lat": center["lat"], "lng": center["lng"]}
            # {lat, lng} 格式
            if "lat" in region and "lng" in region:
                return {"lat": region["lat"], "lng": region["lng"]}
        return None

    @staticmethod
    def _extract_coord_from_location_obj(loc: Any) -> Optional[Dict[str, float]]:
        """从位置对象（{input_type, value} 或其他格式）中提取坐标"""
        if loc is None:
            return None
        if isinstance(loc, dict):
            value = loc.get("value")
            if isinstance(value, dict) and "lat" in value:
                return {"lat": value["lat"], "lng": value["lng"]}
            # 有些 LLM 返回 {coordinates: {lat, lng}} 格式
            coords = loc.get("coordinates")
            if isinstance(coords, dict) and "lat" in coords:
                return {"lat": coords["lat"], "lng": coords["lng"]}
            # 直接 {lat, lng}
            if "lat" in loc and "lng" in loc:
                return {"lat": loc["lat"], "lng": loc["lng"]}
        return None


class AttachmentStubGenerator:
    """
    生成伪多模态的图像结构化描述（attachment stub）。
    当前不生成真实图片，只生成 caption + ocr_texts + geo_hint + task_relevance。
    优先调 LLM 生成自然的描述，fallback 用模板化生成。
    """
    def __init__(self, schema_loader: SchemaLoader, llm_client: DeepSeekChatClient):
        self.schema_loader = schema_loader
        self.llm_client = llm_client

    def generate(self, task_type: str, scenario_frame: Dict[str, Any],
                 global_context: Dict[str, Any], task_dimensions: Dict[str, Any],
                 source_type: str) -> Dict[str, Any]:
        """生成一个 attachment stub"""

        # 先尝试 LLM 生成
        if self.llm_client.available:
            try:
                return self._llm_generate(task_type, scenario_frame, global_context, task_dimensions, source_type)
            except Exception:
                pass

        # fallback: 模板化生成
        return self._fallback_generate(task_type, global_context, task_dimensions, source_type)

    def _llm_generate(self, task_type: str, scenario_frame: Dict[str, Any],
                       global_context: Dict[str, Any], task_dimensions: Dict[str, Any],
                       source_type: str) -> Dict[str, Any]:
        """用 LLM 生成自然的图像描述"""
        image_policy = self.schema_loader.image_policy(task_type)
        prompt = {
            "task_type": task_type,
            "source_type": source_type,
            "image_role": image_policy.get("typical_image_role", "location_context"),
            "global_context": global_context,
            "task_dimensions": task_dimensions,
            "instruction": (
                "你要为一个地理任务 benchmark 生成一个伪多模态的图像描述。\n"
                "想象用户在这个场景下顺手拍了一张照片或截了一张图，描述图里有什么。\n\n"
                "要求：\n"
                "1. caption 要像在描述一张真实照片，包含视觉细节（时间、天气、建筑、标识等）\n"
                "2. ocr_texts 是图里能读出的文字（路牌、店名、站名等），必须是真实存在的\n"
                "3. geo_hint 是这张图暗示的地理位置（level: city/district/street/building, hint: 描述）\n"
                "4. 图的内容必须和任务相关，不能是无关图片\n"
                "5. 图不能直接泄露完整答案（比如不能在图里直接标出最终推荐的 POI）\n\n"
                "返回 JSON，包含 caption, ocr_texts, geo_hint 三个字段。"
            ),
        }
        messages = [
            {"role": "system", "content": "你是一个擅长描述真实场景照片的图像描述生成器。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        result = self.llm_client.json_completion(messages, temperature=0.8)

        # 组装完整的 attachment stub
        attachment_id = f"att_{random.randint(1000, 9999)}"
        modality = "screenshot" if "screenshot" in source_type else "photo"

        # 确定 supports_fields
        supports_fields = self._infer_supports_fields(task_type, image_policy)

        return {
            "attachment_id": attachment_id,
            "modality": modality,
            "source_type": source_type,
            "file_ref": None,
            "caption": result.get("caption", ""),
            "ocr_texts": result.get("ocr_texts", []),
            "geo_hint": result.get("geo_hint"),
            "task_relevance": {
                "is_required": image_policy.get("typical_image_role") == "primary_query",
                "supports_fields": supports_fields,
            }
        }

    def _fallback_generate(self, task_type: str, global_context: Dict[str, Any],
                            task_dimensions: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """模板化 fallback 生成"""
        image_policy = self.schema_loader.image_policy(task_type)
        attachment_id = f"att_{random.randint(1000, 9999)}"
        modality = "screenshot" if "screenshot" in source_type else "photo"

        # 根据场景拼一个基本的 caption
        weather = global_context.get("weather_context", "")
        time_sem = global_context.get("time_semantics", "")
        category = task_dimensions.get("place_category_surface_form") or task_dimensions.get("place_category_canonical", "")

        time_desc = "白天" if time_sem in ("now", "today") else "夜晚" if time_sem in ("tonight",) else ""
        weather_desc = {"rain": "下雨", "snow": "下雪", "clear": "晴天", "hot": "炎热", "cold": "寒冷"}.get(weather, "")

        caption = f"{time_desc}{weather_desc}的街景，能看到路牌和周围的商铺".strip()
        if not caption:
            caption = "一张街景照片，能看到路牌和周围的建筑"

        supports_fields = self._infer_supports_fields(task_type, image_policy)

        return {
            "attachment_id": attachment_id,
            "modality": modality,
            "source_type": source_type,
            "file_ref": None,
            "caption": caption,
            "ocr_texts": [],
            "geo_hint": None,
            "task_relevance": {
                "is_required": image_policy.get("typical_image_role") == "primary_query",
                "supports_fields": supports_fields,
            }
        }

    @staticmethod
    def _infer_supports_fields(task_type: str, image_policy: Dict[str, Any]) -> List[str]:
        """根据任务类型和图像角色推断 supports_fields"""
        role = image_policy.get("typical_image_role", "")
        if role == "primary_query":
            if task_type == "place_lookup":
                return ["query_text", "spatial_bias_value"]
            if task_type == "geocode_resolution":
                return ["input_value"]
        if role == "location_context":
            if task_type in ("place_discovery", "place_filter_rank"):
                return ["search_region_value"]
            if task_type == "route_planning":
                return ["origin"]
        if role == "constraint_supplement":
            return []
        return []


class PlausibilityChecker:
    """
    LLM 联合合理性审核器。
    在程序化 coherence_rules 全部通过之后，调 LLM 判断整个样本
    （global_context + task_dimensions + agent_input）作为一个生活场景是否合理。

    程序规则擅长检查二元字段冲突（如 luggage + bicycle），
    这个模块负责兜底拦截多字段联合的常识级荒谬
    （如 "带小孩 + 下雨 + 婴儿车 + 步行去 12 个景点"）。
    """
    def __init__(self, schema_loader: SchemaLoader, llm_client: DeepSeekChatClient):
        self.schema_loader = schema_loader
        self.llm_client = llm_client

    def validate(self, task_type: str, scenario_frame: Dict[str, Any],
                 global_context: Dict[str, Any], task_dimensions: Dict[str, Any],
                 agent_input: Dict[str, Any]) -> CoherenceValidation:
        """
        返回 CoherenceValidation（复用同一数据结构）。
        LLM 不可用时直接通过（不阻塞 pipeline）。
        """
        # --- 1. 先做程序化的 observation_task_alignment 检查 ---
        programmatic_violations = self._check_observation_alignment(agent_input)
        if programmatic_violations:
            return CoherenceValidation(
                is_valid=False, severity="hard",
                violations=programmatic_violations,
                repair_action="resample_both"
            )

        # --- 2. 调 LLM 做联合合理性审核 ---
        if not self.llm_client.available:
            # LLM 不可用时直接通过
            return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")

        try:
            return self._llm_check(task_type, global_context, task_dimensions, agent_input)
        except Exception as e:
            # LLM 调用失败也直接通过，不阻塞 pipeline
            return CoherenceValidation(is_valid=True, severity="none", violations=[], repair_action="accept")

    def _check_observation_alignment(self, agent_input: Dict[str, Any]) -> List[CoherenceViolation]:
        """程序化检查 attachment 和 task 的对齐"""
        violations = []
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
        """调 LLM 判断场景整体合理性"""
        # 精简 task_dimensions，避免 prompt 过长
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


# ============================================================
# 八、任务分类规则引擎
# ============================================================

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


# ============================================================
# 九、Coherence 规则引擎
# ============================================================

class ConditionEvaluator:
    def __init__(self, schema_loader: SchemaLoader, classification_engine: ClassificationEngine):
        self.schema_loader = schema_loader
        self.classification_engine = classification_engine

    def compute_derived(self, task_type: str, scenario_frame: Dict[str, Any], global_context: Dict[str, Any], task_dimensions: Dict[str, Any]) -> Dict[str, Any]:
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

    def validate(self, task_type: str, scenario_frame: Dict[str, Any], global_context: Dict[str, Any], task_dimensions: Dict[str, Any]) -> CoherenceValidation:
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


# ============================================================
# 十、Maps API 接口与答案层
# ============================================================

class MapsAPIStub:
    """Google Maps API 接口存根，后续替换为真实实现。"""
    def text_search_places(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    def nearby_search_places(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    def place_details(self, place_id: str, fields: List[str]) -> Dict[str, Any]:
        raise NotImplementedError
    def compute_routes(self, origin: Dict[str, Any], destination: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    def compute_route_matrix(self, origins: List[Dict[str, Any]], destinations: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    def geocode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class AnswerTemplateBuilder:
    """答案模板构建器。"""
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

    def generate_constraint_check_records(self, task_type: str, structured_sample: GeneratedSample, observed_values: Dict[str, Any]) -> List[Dict[str, Any]]:
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


# ============================================================
# 十一、主 Pipeline
# ============================================================

class GeoAgentBenchPipeline:
    """
    GeoAgentBench 数据生成管线总控器。
    支持通过参数自定义模型选择：
      pipeline = GeoAgentBenchPipeline(
          schema_path="schema.json",
          llm_client=DeepSeekChatClient(
              generation_model="deepseek-chat",      # 结构化生成
              rewrite_model="deepseek-reasoner",      # 自然语言转写
          )
      )
    """
    def __init__(self, schema_path: Union[str, Path], llm_client: Optional[DeepSeekChatClient] = None, seed: int = 42):
        random.seed(seed)
        self.schema_loader = SchemaLoader(schema_path)
        self.llm_client = llm_client or DeepSeekChatClient()
        self.validator = FieldValidator(self.schema_loader)
        self.programmatic_fallback = ProgrammaticFallbackGenerator(self.schema_loader)
        self.global_context_llm = GlobalContextLLMGenerator(self.schema_loader, self.llm_client)
        self.task_dimensions_llm = TaskDimensionsLLMGenerator(self.schema_loader, self.llm_client)
        self.query_rewriter = QueryRewriter(self.schema_loader, self.llm_client)
        self.classification_engine = ClassificationEngine(self.schema_loader)
        self.coherence_engine = CoherenceEngine(self.schema_loader, self.classification_engine)
        self.answer_builder = AnswerTemplateBuilder(self.schema_loader)
        # v2.0 新增模块
        self.attachment_generator = AttachmentStubGenerator(self.schema_loader, self.llm_client)
        self.agent_input_generator = AgentInputGenerator(self.schema_loader, self.attachment_generator)
        self.plausibility_checker = PlausibilityChecker(self.schema_loader, self.llm_client)

    def sample_task_type(self, task_type: Optional[str] = None) -> str:
        if task_type is not None:
            if task_type not in self.schema_loader.task_types:
                raise ValueError(f"未知 task_type: {task_type}")
            return task_type
        return random.choice(self.schema_loader.task_types)

    def sample_scenario_frame(self, task_type: str) -> Dict[str, Any]:
        compatible = [f for f in self.schema_loader.scenario_frames() if task_type in f["compatible_task_types"]]
        if compatible:
            return copy.deepcopy(random.choice(compatible))
        return {
            "scenario_id": f"generic_{task_type}",
            "default_context": {}, "likely_overrides": {"language_preference": ["zh-CN"]},
            "unlikely_values": {}, "compatible_task_types": [task_type],
        }

    def generate_global_context(self, task_type: str, scenario_frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metadata = {"llm_used": False, "fallback_used": False, "errors": []}
        try:
            global_context = self.global_context_llm.generate(task_type, scenario_frame)
            valid, errors = self.validator.validate_record(self.schema_loader.global_context_fields(), global_context)
            if valid:
                metadata["llm_used"] = True
                return global_context, metadata
            metadata["errors"].extend(errors)
        except Exception as e:
            metadata["errors"].append(f"LLM global_context 失败: {e}")
        global_context = self.programmatic_fallback.generate_global_context(task_type, scenario_frame)
        valid, errors = self.validator.validate_record(self.schema_loader.global_context_fields(), global_context)
        if not valid:
            raise RuntimeError(f"fallback 生成的 global_context 仍不合法: {errors}")
        metadata["fallback_used"] = True
        return global_context, metadata

    def generate_task_dimensions(self, task_type: str, global_context: Dict[str, Any],
                                  scenario_frame: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metadata = {"llm_used": False, "fallback_used": False, "errors": []}
        prefilled = self.programmatic_fallback.prefill_task_dimensions_explicit(task_type, global_context, scenario_frame)
        try:
            completed = self.task_dimensions_llm.generate_partial(task_type, global_context, prefilled)
            merged = copy.deepcopy(prefilled)
            merged.update(completed)
            valid, errors = self.validator.validate_record(self.schema_loader.input_dimensions(task_type), merged)
            if valid:
                metadata["llm_used"] = True
                return merged, metadata
            metadata["errors"].extend(errors)
        except Exception as e:
            metadata["errors"].append(f"LLM task_dimensions 失败: {e}")
        fallback_dims = self.programmatic_fallback.fill_missing_task_dimensions(task_type, prefilled, global_context)
        valid, errors = self.validator.validate_record(self.schema_loader.input_dimensions(task_type), fallback_dims)
        if not valid:
            raise RuntimeError(f"fallback 生成的 task_dimensions 仍不合法: {errors}")
        metadata["fallback_used"] = True
        return fallback_dims, metadata

    def rewrite_query(self, task_type: str, global_context: Dict[str, Any],
                      task_dimensions: Dict[str, Any], agent_input: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        metadata = {"llm_used": False, "fallback_used": False, "errors": []}
        try:
            query = self.query_rewriter.rewrite(task_type, global_context, task_dimensions, agent_input)
            if isinstance(query, str) and query.strip():
                metadata["llm_used"] = True
                return query.strip(), metadata
        except Exception as e:
            metadata["errors"].append(f"LLM query 转写失败: {e}")
        query = self._fallback_rewrite_query(task_type, global_context, task_dimensions, agent_input)
        metadata["fallback_used"] = True
        return query, metadata

    def _fallback_rewrite_query(self, task_type: str, global_context: Dict[str, Any],
                                 task_dimensions: Dict[str, Any], agent_input: Optional[Dict[str, Any]] = None) -> str:
        pieces: List[str] = []

        # 如果有 attachment，先加一句引导
        attachments = (agent_input or {}).get("attachments", [])
        if attachments:
            pieces.append("我拍了张照片，帮我看看")

        trip_state = global_context.get("trip_state")
        weather = global_context.get("weather_context")
        urgency = global_context.get("urgency_level")
        party = global_context.get("user_party_composition")
        if trip_state:
            pieces.append(f"我现在是{trip_state}")
        if weather:
            pieces.append(f"外面天气是{weather}")
        if party:
            pieces.append(f"我是{party}")
        if urgency:
            pieces.append(f"现在紧迫程度是{urgency}")
        if task_type in {"place_discovery", "place_filter_rank", "meeting_point"}:
            category = task_dimensions.get("place_category_surface_form") or task_dimensions.get("place_category_canonical")
            pieces.append(f"想找{category}")
        elif task_type == "place_lookup":
            pieces.append(f"帮我找一下{task_dimensions.get('query_text', '')}")
        elif task_type == "route_planning":
            origin = self._location_to_text(task_dimensions.get("origin"))
            destination = self._location_to_text(task_dimensions.get("destination"))
            pieces.append(f"我想从{origin}去{destination}")
        elif task_type == "route_choice":
            pieces.append("帮我比较这些候选目的地哪个更合适")
        elif task_type == "geocode_resolution":
            input_val = self._location_to_text(task_dimensions.get("input_value"))
            target_form = task_dimensions.get("target_output_form", "标准地址")
            pieces.append(f"帮我把{input_val}解析成{target_form}")
        elif task_type == "itinerary_planning":
            pieces.append("帮我规划一个行程")
        return "，".join([p for p in pieces if p])

    @staticmethod
    def _location_to_text(loc: Any) -> str:
        if loc is None:
            return "某个位置"
        if isinstance(loc, str):
            return loc
        if isinstance(loc, dict):
            value = loc.get("value")
            if value is not None:
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    lat = value.get("lat")
                    lng = value.get("lng")
                    if lat is not None and lng is not None:
                        return f"({lat}, {lng})"
            lat = loc.get("lat")
            lng = loc.get("lng")
            if lat is not None and lng is not None:
                return f"({lat}, {lng})"
            for v in loc.values():
                if isinstance(v, str) and v:
                    return v
        return str(loc)

    def generate_one(self, task_type: Optional[str] = None, max_retries: int = 20) -> GeneratedSample:
        """
        生成一个完整的 benchmark 样本。9 步流程：
        1. 采样 task_type
        2. 采样 scenario_frame
        3. 生成 global_context
        4. 生成 task_dimensions
        5. 程序化质控（classification + coherence）
        6. 生成 agent_input（device_context + attachments）
        7. LLM 联合合理性审核（plausibility）
        8. 转写自然语言 query
        9. 回填 agent_input.user_message.primary_text 和 natural_language_query
        """
        last_errors: List[str] = []
        for attempt in range(1, max_retries + 1):
            current_task_type = self.sample_task_type(task_type)
            scenario_frame = self.sample_scenario_frame(current_task_type)
            try:
                # --- 步骤 3: 生成 global_context ---
                global_context, gc_meta = self.generate_global_context(current_task_type, scenario_frame)

                # --- 步骤 4: 生成 task_dimensions ---
                task_dimensions, td_meta = self.generate_task_dimensions(current_task_type, global_context, scenario_frame)

                # --- 步骤 5: 程序化质控 ---
                cls_ok, cls_violations = self.classification_engine.validate(current_task_type, task_dimensions)
                coherence = self.coherence_engine.validate(current_task_type, scenario_frame, global_context, task_dimensions)
                if not cls_ok:
                    coherence.violations.extend(cls_violations)
                    coherence.is_valid = False
                    coherence.severity = "hard"
                    coherence.repair_action = "resample_both"
                if not coherence.is_valid:
                    last_errors = [v.message for v in coherence.violations]
                    continue

                # --- 步骤 6: 生成 agent_input ---
                agent_input, ai_meta = self.agent_input_generator.generate(
                    current_task_type, scenario_frame, global_context, task_dimensions
                )

                # --- 步骤 7: LLM 联合合理性审核 ---
                plausibility = self.plausibility_checker.validate(
                    current_task_type, scenario_frame, global_context, task_dimensions, agent_input
                )
                if not plausibility.is_valid:
                    last_errors = [v.message for v in plausibility.violations]
                    continue

                # --- 步骤 8: 转写自然语言 query ---
                query, query_meta = self.rewrite_query(current_task_type, global_context, task_dimensions, agent_input)

                # --- 步骤 9: 回填 ---
                agent_input["user_message"]["primary_text"] = query

                return GeneratedSample(
                    benchmark_name=self.schema_loader.benchmark_name,
                    schema_version=self.schema_loader.version,
                    task_type=current_task_type,
                    scenario_frame_id=scenario_frame["scenario_id"],
                    global_context=global_context,
                    task_dimensions=task_dimensions,
                    agent_input=agent_input,
                    natural_language_query=query,
                    coherence_validation=coherence,
                    plausibility_validation=plausibility,
                    metadata={
                        "attempt": attempt,
                        "global_context_generation": gc_meta,
                        "task_dimensions_generation": td_meta,
                        "agent_input_generation": ai_meta,
                        "query_rewrite": query_meta,
                    },
                )
            except Exception as e:
                last_errors.append(str(e))
                continue
        raise RuntimeError(f"在 {max_retries} 次尝试内未能生成合法样本。最近错误: {last_errors[-5:]}")

    def generate_batch(self, n: int, task_type: Optional[str] = None) -> List[GeneratedSample]:
        samples: List[GeneratedSample] = []
        for _ in range(n):
            samples.append(self.generate_one(task_type=task_type))
        return samples


# ============================================================
# 十二、演示入口
# ============================================================

def main() -> None:
    schema_path = Path(os.getenv("GEOAGENTBENCH_SCHEMA_PATH", Path(__file__).parent / "schema.json"))
    pipeline = GeoAgentBenchPipeline(schema_path=schema_path)

    demo_task_types = [
        "place_lookup", "place_discovery", "place_filter_rank", "route_planning",
        "route_choice", "meeting_point", "geocode_resolution", "itinerary_planning",
    ]

    print("=" * 100)
    print("GeoAgentBench Pipeline v2.0 Demo")
    print(f"Schema: {schema_path} (v{pipeline.schema_loader.version})")
    print(f"Generation model: {pipeline.llm_client.generation_model}")
    print(f"Rewrite model: {pipeline.llm_client.rewrite_model}")
    print(f"LLM available: {pipeline.llm_client.available}")
    print(f"Scenario frames: {len(pipeline.schema_loader.scenario_frames())}")
    print("=" * 100)

    for task_type in demo_task_types:
        print("\n" + "-" * 100)
        print(f"任务类型: {task_type}")
        sample = pipeline.generate_one(task_type=task_type)

        # 打印 agent_input（这是 agent 实际收到的输入）
        print("[Agent Input]")
        ai = sample.agent_input
        print(f"  input_mode: {ai.get('input_mode')}")
        print(f"  device_context.gps: {ai.get('device_context', {}).get('gps')}")
        print(f"  device_context.current_time: {ai.get('device_context', {}).get('current_time')}")
        attachments = ai.get("attachments", [])
        if attachments:
            att = attachments[0]
            print(f"  attachment: modality={att.get('modality')}, source={att.get('source_type')}")
            print(f"    caption: {att.get('caption', '')[:80]}...")
            print(f"    ocr_texts: {att.get('ocr_texts', [])}")
            print(f"    task_relevance: required={att.get('task_relevance', {}).get('is_required')}")
        else:
            print("  attachments: []")

        print(f"[自然语言 Query]")
        print(f"  {sample.natural_language_query}")
        print(f"[Plausibility] valid={sample.plausibility_validation.is_valid if sample.plausibility_validation else 'N/A'}")
        print(f"[Metadata] attempt={sample.metadata.get('attempt')}")

    print("\n" + "=" * 100)
    print("批量生成示例：随机生成 3 条样本")
    batch = pipeline.generate_batch(3)
    for idx, sample in enumerate(batch, 1):
        print(f"\n--- Batch #{idx} / {sample.task_type} / mode={sample.agent_input.get('input_mode')} ---")
        print(f"  Query: {sample.natural_language_query}")
        atts = sample.agent_input.get("attachments", [])
        if atts:
            print(f"  Attachment: {atts[0].get('caption', '')[:60]}...")


if __name__ == "__main__":
    main()
