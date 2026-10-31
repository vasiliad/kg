#!/usr/bin/env python3
"""
Model size resolution: how many parameters does a model actually have?

The operator's rule (2026-10-31): a model with NO size in its id must have
its size determined from the catalogue metadata or the documentation — not
just admitted to the pool. Otherwise the pool accumulates garbage. An id
whose size cannot be resolved from any source is EXCLUDED (fail-closed).

Sources, in order:
  1. the id itself            "nemotron-3-ultra-550b-a55b" -> 550B (TOTAL,
                              the active MoE suffix is ignored)
  2. catalogue metadata       any text field of the /v1/models entry
                              ("...550 billion parameters...")
  3. committed overrides      data/model_sizes.json — operator pins or a
                              previous docs lookup is reused; survives
                              runner recycling
  4. documentation search     one bounded web query per unknown model, the
                              result is cached into the overrides file

Every function is defensive: resolution never raises and never fabricates —
"unknown" is a valid, honest outcome that simply keeps the model out.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_FILE = PROJECT_ROOT / "data" / "model_sizes.json"

_ID_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)(tb|t|b)(?=$|[-_a-z])")

# Unambiguous prose forms: "550 billion parameters", "1.2 trillion", "550 млрд параметров".
_PROSE_SIZE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[- ]?(trillion|trillions|billion|billions|трлн|млрд)\b",
    re.I,
)
# Attached-unit forms tied to the word "parameters": "550B parameters", "70B param",
# "230B parameter model". Bare "550B" without a parameter word nearby is NOT
# accepted — "4B token context" would be a false positive.
_ATTACHED_SIZE_RE = re.compile(
    r"(\d{2,4}(?:\.\d+)?)\s?b(?:illion)?\s*[- ]?\s*"
    r"(parameters|parameter|params|param|параметров|параметра|параметр|млрд)",
    re.I,
)


def parse_size_from_text(text: str) -> Optional[float]:
    """Extracts a parameter count in BILLIONS from free text. None if absent."""
    if not text:
        return None
    best: Optional[float] = None

    for m in _PROSE_SIZE_RE.finditer(text):
        val = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        params_b = val * 1000.0 if unit in ("trillion", "trillions", "трлн") else val
        best = params_b if best is None else max(best, params_b)

    for m in _ATTACHED_SIZE_RE.finditer(text):
        val = float(m.group(1))
        best = val if best is None else max(best, val)

    return best


def parse_size_from_id(model_id: str) -> Optional[float]:
    """Total parameter count encoded in the id itself (billions)."""
    best: Optional[float] = None
    for m in _ID_SIZE_RE.finditer((model_id or "").lower()):
        val = float(m.group(1))
        unit = m.group(2)
        params_b = val * 1000.0 if unit in ("tb", "t") else val
        best = params_b if best is None else max(best, params_b)
    return best


def _entry_text(entry: Any) -> str:
    """Flattens a /v1/models entry (minus the id) into one searchable string."""
    try:
        if not isinstance(entry, dict):
            return ""
        parts = []
        for key, value in entry.items():
            if key in ("id", "created", "object"):
                continue
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                parts.append(" ".join(str(v) for v in value.values() if isinstance(v, str)))
            elif isinstance(value, list):
                parts.append(" ".join(str(v) for v in value if isinstance(v, (str, int, float))))
        return " ".join(parts)
    except Exception:
        return ""


def load_overrides(path: Optional[Path] = None) -> Dict[str, Any]:
    """Reads the committed size overrides/cache (never raises)."""
    p = Path(path) if path else OVERRIDES_FILE
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_overrides(overrides: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Atomically persists the size cache."""
    p = Path(path) if path else OVERRIDES_FILE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(str(tmp), str(p))
    except Exception:
        pass  # cache write failure must never break a tuning cycle


class SizeResolver:
    """
    Resolves model sizes with caching. `searcher` is an optional callable
    (query -> list of text snippets) — the tuner wires the existing
    Keenable/Tavily/DuckDuckGo search stack into it.
    """

    def __init__(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        searcher: Optional[Callable[[str], Any]] = None,
        cache_path: Optional[Path] = None,
        max_doc_lookups: int = 6,
    ):
        self._overrides = dict(overrides) if overrides is not None else load_overrides(cache_path)
        self._cache_path = cache_path
        self._searcher = searcher
        self._doc_lookups_left = int(max_doc_lookups)
        self.sources: Dict[str, str] = {}

    def _cache(self, model_id: str, params_b: float, source: str) -> None:
        self._overrides[model_id] = {"params_b": params_b, "source": source}
        self.sources[model_id] = source
        save_overrides(self._overrides, self._cache_path)

    def resolve(self, model_id: str, entry: Any = None) -> Optional[float]:
        """
        Returns the parameter count in billions, or None = "cannot be
        determined" (the caller must then exclude the model, fail-closed).
        """
        # 1. the id itself
        from_id = parse_size_from_id(model_id)
        if from_id:
            self.sources[model_id] = "id"
            return from_id

        cached = self._overrides.get(model_id)
        if isinstance(cached, dict) and isinstance(cached.get("params_b"), (int, float)):
            self.sources[model_id] = f"cache:{cached.get('source', 'pinned')}"
            return float(cached["params_b"])

        # 2. catalogue metadata of this very entry
        from_meta = parse_size_from_text(_entry_text(entry))
        if from_meta:
            self._cache(model_id, from_meta, "catalog")
            return from_meta

        # 3. documentation search, bounded per cycle and cached
        if self._searcher and self._doc_lookups_left > 0:
            self._doc_lookups_left -= 1
            try:
                snippets = self._searcher(
                    f"{model_id} large language model parameters billion trillion"
                )
                for snippet in snippets or []:
                    size = parse_size_from_text(str(snippet))
                    if size:
                        self._cache(model_id, size, "docs")
                        return size
            except Exception:
                pass

        self.sources[model_id] = "unresolved"
        return None
