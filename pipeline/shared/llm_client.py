"""LLM client abstraction layer.

Architecture
───────────────────────────────────────────────────────────────
BaseLLMClient           ← abstract base, all modules type-hint against this
  OpenAICompatibleClient  ← OpenAI SDK; covers DashScope, any relay, OpenAI itself
  AnthropicNativeClient   ← native anthropic SDK (optional dep)

ClientFactory.for_stage(stage, config)
  → reads config.yaml  stage_models + model_endpoints
  → returns the right BaseLLMClient for that pipeline stage

Stage constants (use these in modules, never hardcode endpoint names):
  STAGE_VISION_PARSE    Module 2  image parsing
  STAGE_TASK_DESIGN     Module 3  scenario + dimension design
  STAGE_QUALITY_LLM     Module 4  LLM plausibility check
  STAGE_QUERY_REWRITE   Module 5  NL rewriting
  STAGE_QUERY_VERIFY    Module 5  leak detection

Switching a model for a stage:
  1. Add / edit an endpoint block in config.yaml → model_endpoints
  2. Change the value in config.yaml → stage_models
  Code is untouched.
───────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Stage name constants ────────────────────────────────────────────────────
STAGE_VISION_PARSE  = "vision_parse"   # Module 2
STAGE_TASK_DESIGN   = "task_design"    # Module 3
STAGE_QUALITY_LLM   = "quality_llm"   # Module 4
STAGE_QUERY_REWRITE = "query_rewrite"  # Module 5
STAGE_QUERY_VERIFY  = "query_verify"   # Module 5


# ── JSON extraction helper ──────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from an LLM text response.

    Handles: pure JSON, ```json ... ``` blocks, leading/trailing prose,
    and single-line inline JSON.
    """
    raw = raw.strip()

    # 1. Strip markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if fence:
        return json.loads(fence.group(1))

    # 2. Try parsing the whole response directly
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3. Find the first {...} block (handles leading/trailing prose)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")


# ── Abstract base ────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Every LLM client in this pipeline must implement this interface.

    Modules should type-hint against BaseLLMClient, never against a concrete
    subclass, so that swapping the underlying model requires zero code changes.
    """

    # ── identity ──

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier string used in API calls."""

    @property
    def available(self) -> bool:
        """False if the client failed to initialise (bad key, missing dep, etc.)."""
        return True

    @property
    def supports_vision(self) -> bool:
        """True if this client can process image inputs."""
        return False

    # ── core methods (modules call these) ──

    @abstractmethod
    def text_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat messages list, return plain text response."""

    def json_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Send messages, parse and return JSON response.

        Default implementation calls text_completion and extracts JSON.
        Subclasses may override to use native JSON mode (e.g. response_format).
        """
        raw = self.text_completion(messages, temperature=temperature)
        return _extract_json(raw)

    def rewrite_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.8,
    ) -> str:
        """Alias for text_completion.

        Kept for call-site readability in Module 5; modules that want
        to call the "strong rewriter model" should use
        ClientFactory.for_stage(STAGE_QUERY_REWRITE, config).
        """
        return self.text_completion(messages, temperature=temperature)

    def vision_json_completion(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Parse an image + prompt, return JSON.

        Raises NotImplementedError if the client does not support vision.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} (model={self.model!r}) does not support vision. "
            "Set supports_vision: true in this endpoint's config block and use a "
            "vision-capable model."
        )


# ── OpenAI-compatible client ─────────────────────────────────────────────────

class OpenAICompatibleClient(BaseLLMClient):
    """LLM client for any OpenAI-compatible API.

    Works with:
    - Alibaba DashScope  (Qwen models)
    - Third-party relay stations
    - OpenAI itself
    - Any other service that speaks the OpenAI chat completions protocol

    Vision is enabled by passing supports_vision=True in the endpoint config.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        supports_vision_flag: bool = False,
    ):
        self._model = model_name
        self._supports_vision = supports_vision_flag
        self._client = None
        try:
            from openai import OpenAI  # type: ignore
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            import warnings
            warnings.warn(f"OpenAICompatibleClient init failed for {model_name!r}: {exc}")

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def supports_vision(self) -> bool:
        return self._supports_vision

    def _ensure(self):
        if self._client is None:
            raise RuntimeError(
                f"OpenAICompatibleClient is not available for model {self._model!r}. "
                "Check api_key / base_url in config.yaml."
            )
        return self._client

    def text_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def json_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Use response_format=json_object when available, else fallback to text+parse."""
        client = self._ensure()
        try:
            resp = client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            # Some relay stations / older endpoints don't support json_object mode
            raw = self.text_completion(messages, temperature=temperature)
            return _extract_json(raw)

    def vision_json_completion(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        if not self._supports_vision:
            raise NotImplementedError(
                f"Model {self._model!r} endpoint is not configured with supports_vision: true"
            )
        client = self._ensure()
        image_path = Path(image_path)
        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp",
                "gif": "image/gif"}.get(suffix, "image/jpeg")
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        data_url = f"data:{mime};base64,{b64}"

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }]
        resp = client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content.strip()
        return _extract_json(raw)


# ── Anthropic native client ──────────────────────────────────────────────────

class AnthropicNativeClient(BaseLLMClient):
    """Native Anthropic SDK client (Claude models).

    Requires: pip install anthropic
    Works with: Anthropic's official API.
    For third-party Claude relays, use OpenAICompatibleClient instead
    (most relays expose the OpenAI protocol).
    """

    def __init__(self, model_name: str, api_key: str):
        self._model = model_name
        self._client = None
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            import warnings
            warnings.warn(
                "AnthropicNativeClient: 'anthropic' package not installed. "
                "Run: pip install anthropic"
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"AnthropicNativeClient init failed: {exc}")

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def supports_vision(self) -> bool:
        return True  # All Claude models support vision

    def _ensure(self):
        if self._client is None:
            raise RuntimeError(
                "AnthropicNativeClient is not available. "
                "Check api_key and that 'anthropic' is installed."
            )
        return self._client

    def text_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        client = self._ensure()
        # Convert OpenAI-style messages to Anthropic format
        system_text = ""
        anthropic_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                anthropic_msgs.append({"role": m["role"], "content": m["content"]})

        kwargs: Dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=anthropic_msgs,
        )
        if system_text:
            kwargs["system"] = system_text

        resp = client.messages.create(**kwargs)
        return resp.content[0].text.strip()

    def vision_json_completion(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        client = self._ensure()
        image_path = Path(image_path)
        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp",
                "gif": "image/gif"}.get(suffix, "image/jpeg")
        b64 = base64.b64encode(image_path.read_bytes()).decode()

        resp = client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return _extract_json(resp.content[0].text)


# ── Client factory ───────────────────────────────────────────────────────────

class ClientFactory:
    """Build LLM clients from config.

    Usage in modules:
        from shared.llm_client import ClientFactory, STAGE_TASK_DESIGN
        llm = ClientFactory.for_stage(STAGE_TASK_DESIGN, config)
        result = llm.json_completion(messages)
    """

    _PROVIDERS = {
        "openai_compatible": "_build_openai_compatible",
        "anthropic":         "_build_anthropic",
    }

    @classmethod
    def for_stage(cls, stage: str, config) -> BaseLLMClient:
        """Return the configured client for a pipeline stage.

        Args:
            stage: one of the STAGE_* constants
            config: PipelineConfig instance
        """
        endpoint_name = config.stage_model(stage)
        endpoint_cfg  = config.model_endpoint(endpoint_name)
        return cls._build(endpoint_cfg, endpoint_name)

    @classmethod
    def _build(cls, cfg: Dict[str, Any], name: str = "") -> BaseLLMClient:
        provider = cfg.get("provider", "openai_compatible")
        builder  = cls._PROVIDERS.get(provider)
        if builder is None:
            raise ValueError(
                f"Unknown provider {provider!r} in endpoint {name!r}. "
                f"Supported: {list(cls._PROVIDERS)}"
            )
        return getattr(cls, builder)(cfg)

    @classmethod
    def _build_openai_compatible(cls, cfg: Dict[str, Any]) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            model_name=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            supports_vision_flag=cfg.get("supports_vision", False),
        )

    @classmethod
    def _build_anthropic(cls, cfg: Dict[str, Any]) -> AnthropicNativeClient:
        return AnthropicNativeClient(
            model_name=cfg["model"],
            api_key=cfg["api_key"],
        )


# ── Backward-compat alias ────────────────────────────────────────────────────
# Existing code that does `from shared.llm_client import DashScopeClient`
# will keep working. The alias builds a client from the config exactly as
# before, but through the new abstraction layer.

class DashScopeClient(OpenAICompatibleClient):
    """Legacy alias. Prefer ClientFactory.for_stage() in new code."""

    def __init__(self, config=None):
        from .config import PipelineConfig
        cfg = config or PipelineConfig()
        super().__init__(
            model_name=cfg.model_text,
            api_key=cfg.dashscope_api_key,
            base_url=cfg.dashscope_base_url,
            supports_vision_flag=True,
        )
        self._config = cfg

    # ── Keep old method signatures so existing callers don't break ──

    def json_completion(self, messages, temperature=0.3, model=None):
        if model and model != self._model:
            tmp = OpenAICompatibleClient(model, self._config.dashscope_api_key,
                                         self._config.dashscope_base_url)
            return tmp.json_completion(messages, temperature=temperature)
        return super().json_completion(messages, temperature=temperature)

    def text_completion(self, messages, temperature=0.7, max_tokens=2048, model=None):
        if model and model != self._model:
            tmp = OpenAICompatibleClient(model, self._config.dashscope_api_key,
                                         self._config.dashscope_base_url)
            return tmp.text_completion(messages, temperature=temperature)
        return super().text_completion(messages, temperature=temperature,
                                        max_tokens=max_tokens)

    def rewrite_completion(self, messages, temperature=0.8, model=None):
        target = model or self._config.model_text_strong
        if target != self._model:
            tmp = OpenAICompatibleClient(target, self._config.dashscope_api_key,
                                          self._config.dashscope_base_url)
            return tmp.text_completion(messages, temperature=temperature)
        return super().text_completion(messages, temperature=temperature)

    def vision_json_completion(self, image_path, prompt, temperature=0.3, model=None):
        if model and model != self._model:
            tmp = OpenAICompatibleClient(model, self._config.dashscope_api_key,
                                          self._config.dashscope_base_url,
                                          supports_vision_flag=True)
            return tmp.vision_json_completion(image_path, prompt, temperature=temperature)
        return super().vision_json_completion(image_path, prompt, temperature=temperature)
