"""DashScope OpenAI-compatible LLM client for Qwen models."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import PipelineConfig


class DashScopeClient:
    """Unified client for DashScope (Qwen) models via OpenAI-compatible API.

    Supports:
    - json_completion: structured JSON output (text model)
    - text_completion: plain text output (text model)
    - rewrite_completion: natural language rewriting (strong text model)
    - vision_completion: image understanding (vision model)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._client: Optional[OpenAI] = None
        try:
            self._client = OpenAI(
                api_key=self.config.dashscope_api_key,
                base_url=self.config.dashscope_base_url,
            )
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError("DashScope 客户端不可用：请检查 config.yaml 中的 dashscope.api_key")
        return self._client

    # ── structured JSON ──
    def json_completion(self, messages: List[Dict[str, str]],
                        temperature: float = 0.7,
                        model: Optional[str] = None) -> Dict[str, Any]:
        client = self._ensure_client()
        model = model or self.config.model_text
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    # ── plain text ──
    def text_completion(self, messages: List[Dict[str, str]],
                        temperature: float = 0.8,
                        model: Optional[str] = None) -> str:
        client = self._ensure_client()
        model = model or self.config.model_text
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # ── rewrite (uses strong text model) ──
    def rewrite_completion(self, messages: List[Dict[str, str]],
                           temperature: float = 0.8,
                           model: Optional[str] = None) -> str:
        client = self._ensure_client()
        model = model or self.config.model_text_strong
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # ── vision (image + text) ──
    def vision_completion(self, image_path: str | Path,
                          prompt: str,
                          temperature: float = 0.5,
                          model: Optional[str] = None) -> str:
        client = self._ensure_client()
        model = model or self.config.model_vision
        image_path = Path(image_path)

        suffix = image_path.suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}.get(suffix.lstrip("."), "image/jpeg")
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        data_url = f"data:{mime};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def vision_json_completion(self, image_path: str | Path,
                               prompt: str,
                               temperature: float = 0.5,
                               model: Optional[str] = None) -> Dict[str, Any]:
        raw = self.vision_completion(image_path, prompt, temperature, model)
        # Try to extract JSON from the response
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]  # remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        return json.loads(raw)
