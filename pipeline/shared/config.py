"""Load and provide typed access to config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class PipelineConfig:
    """Typed wrapper around config.yaml."""

    def __init__(self, path: Path | str = _DEFAULT_CONFIG_PATH):
        self._path = Path(path)
        with open(self._path, encoding="utf-8") as f:
            self._raw: Dict[str, Any] = yaml.safe_load(f)

    # --- Google Maps ---
    @property
    def google_maps_api_key(self) -> str:
        return self._raw["google_maps"]["api_key"]

    # --- DashScope ---
    @property
    def dashscope_api_key(self) -> str:
        return self._raw["dashscope"]["api_key"]

    @property
    def dashscope_base_url(self) -> str:
        return self._raw["dashscope"]["base_url"]

    # --- Models ---
    @property
    def model_vision(self) -> str:
        return self._raw["models"]["vision"]

    @property
    def model_text(self) -> str:
        return self._raw["models"]["text"]

    @property
    def model_text_strong(self) -> str:
        return self._raw["models"]["text_strong"]

    # --- raw access ---
    def get(self, dotted_key: str, default: Any = None) -> Any:
        parts = dotted_key.split(".")
        cur = self._raw
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return default
            if cur is None:
                return default
        return cur
