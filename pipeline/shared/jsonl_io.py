"""JSONL read/write utilities + errors.jsonl helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path | str, records: Iterable[Dict[str, Any]], *, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_error(errors_path: Path | str, record: Dict[str, Any]) -> None:
    write_jsonl(errors_path, [record], append=True)
