#!/usr/bin/env python3
"""
Eternal Memory Engine for Autonomous Complex.
Combines ultra-compact persistent text storage (data/telegram_history.txt)
with state-of-the-art semantic vector retrieval using NVIDIA Nemotron 3.5 Embed.

Storage model — text log is the source of truth, the semantic index is derived:
- data/telegram_history.txt holds the full conversation in plain text. It is
  small, human-readable and committed to Git, so memory survives the runner
  filesystem being wiped between jobs.
- data/.memory_vault.json is a derived cache (int8-quantised embeddings). It
  is NOT committed: it is rebuilt from the text log at startup, which keeps
  the repository from growing by ~37 KB per remembered message.

Features:
1. Zero-bloat text logging: entire conversation history stored in human-readable plain text.
2. Semantic indexing via NVIDIA NIM: nvidia/nemotron-3.5-embed-8b (or 1b).
3. Pure Python vector similarity (cosine) without heavy external database dependencies.
4. Sovereign fallback: falls back to keyword/token overlap if API is unavailable.
5. Context synthesis: builds compact, highly relevant memory envelopes for LLMs to prevent context degradation.
6. Self-restoring index: rebuilds itself from the durable log when missing or stale.
"""

import os
import sys
import json
import math
import re
import time
import threading
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from harness.key_pool import NVIDIAKeyPool
from harness import vector_codec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class MemoryEngine:
    def __init__(
        self,
        history_file: Optional[Path] = None,
        vault_file: Optional[Path] = None,
        nvidia_api_key: Optional[str] = None,
        model_name: str = "nvidia/nemotron-3.5-embed-8b",
        backup_model: str = "nvidia/nemotron-3.5-embed-1b",
        log_format: str = "conversation"
    ):
        """
        log_format selects the durable log protocol:
          - "conversation": blocks of "[timestamp] ROLE: text", separated by a
            blank line (data/telegram_history.txt).
          - "events": one "[timestamp] message" per line (nim_history.log).
        Both formats feed the same semantic index and the same hybrid
        retrieval, so operational events are remembered exactly like the
        conversation is.
        """
        self.log_format = log_format if log_format in ("conversation", "events") else "conversation"
        self.history_file = history_file or (PROJECT_ROOT / "data" / "telegram_history.txt")
        self.vault_file = vault_file or (PROJECT_ROOT / "data" / ".memory_vault.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        # Guards vault mutations across bot worker threads and the test suite.
        self._io_lock = threading.Lock()
        
        self.api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY", "")
        if not self.api_key:
            env_path = PROJECT_ROOT / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("NVIDIA_API_KEY="):
                                self.api_key = line.strip().split("=", 1)[1].strip()
                except Exception:
                    pass

        self.model_name = model_name
        self.backup_model = backup_model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.key_pool = NVIDIAKeyPool(self.api_key)
        self._vault_cache: Optional[List[Dict[str, Any]]] = None
        # Decoded vectors are cached by their base64 payload so repeated
        # searches never re-decode the same sketch.
        self._vec_cache: Dict[str, List[float]] = {}
        # The vault is committed back into Git, so it must stay bounded.
        try:
            self.max_entries = max(1, int(os.getenv("MEMORY_VAULT_MAX_ENTRIES", "300")))
        except ValueError:
            self.max_entries = 300

    def _load_vault(self) -> List[Dict[str, Any]]:
        if self._vault_cache is not None:
            return self._vault_cache

        if self.vault_file.exists():
            try:
                with open(self.vault_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("entries", []) if isinstance(data, dict) else data
                self._vault_cache = self._migrate_entries(entries)
                return self._vault_cache
            except Exception as e:
                print(f"[MemoryEngine] Warning reading vault: {e}")
        self._vault_cache = []
        return self._vault_cache

    @staticmethod
    def _migrate_entries(entries: Any) -> List[Dict[str, Any]]:
        """
        Converts legacy full-dimension float vectors into the compact
        projected/int8 representation. Runs on every load, so a vault written
        by an older revision is rewritten in the new format the first time it
        is saved — no manual migration step, no data loss.
        """
        migrated: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return migrated

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            vec = entry.get("vector")
            if isinstance(vec, list):
                payload, scale = vector_codec.encode_int8(vec)
                entry = dict(entry)
                entry["vector"] = payload or None
                entry["vector_scale"] = scale
                entry["vector_dim"] = len(vec)
            migrated.append(entry)
        return migrated

    def _entry_vector(self, entry: Dict[str, Any]) -> Optional[List[float]]:
        """Returns the decoded vector of a vault entry (cached by payload)."""
        vec = entry.get("vector")
        if isinstance(vec, list):
            # Legacy vector still held in memory: quantize it on the fly.
            payload, scale = vector_codec.encode_int8(vec)
            entry["vector"] = payload or None
            entry["vector_scale"] = scale
            entry["vector_dim"] = len(vec)
            vec = payload

        if isinstance(vec, str) and vec:
            cached = self._vec_cache.get(vec)
            if cached is None:
                cached = vector_codec.decode_int8(vec, entry.get("vector_scale") or 1.0)
                if len(self._vec_cache) > 2048:
                    self._vec_cache.clear()
                self._vec_cache[vec] = cached
            return cached
        return None

    def _save_vault(self) -> None:
        if self._vault_cache is None:
            return
        try:
            # Keep the vault bounded: it is committed back into Git.
            if len(self._vault_cache) > self.max_entries:
                self._vault_cache = self._vault_cache[-self.max_entries:]
            # Atomic replace so a crash mid-write can never corrupt the vault.
            tmp_path = self.vault_file.with_suffix(self.vault_file.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"version": 2, "entries": self._vault_cache},
                    f,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            os.replace(str(tmp_path), str(self.vault_file))
        except Exception as e:
            print(f"[MemoryEngine] Warning saving vault: {e}")

    def get_embedding(self, text: str, input_type: str = "passage") -> Optional[List[float]]:
        """
        Retrieves embedding vector via NVIDIA NIM API using Nemotron.
        Rotates through the key pool on 401/429.
        """
        if not self.key_pool.current():
            return None

        # Clean text
        clean_text = text.strip()[:4000] # Cap input chunk
        if not clean_text:
            return None

        for model in [self.model_name, self.backup_model]:
            payload = {
                "input": [clean_text],
                "model": model,
                "input_type": input_type,
                "encoding_format": "float"
            }
            for _ in range(max(len(self.key_pool), 1)):
                headers = {
                    "Authorization": f"Bearer {self.key_pool.current()}",
                    "Content-Type": "application/json"
                }
                try:
                    resp = requests.post(f"{self.base_url}/embeddings", headers=headers, json=payload, timeout=12)
                except Exception:
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    vec = data["data"][0]["embedding"]
                    # Round floats to 5 decimals to minimize file size by 60%
                    return [round(x, 5) for x in vec]
                if resp.status_code in (401, 403, 429) and len(self.key_pool) > 1:
                    self.key_pool.mark_bad()
                    continue
                break

        return None

    def get_embeddings_batch(
        self,
        texts: List[str],
        input_type: str = "passage"
    ) -> List[Optional[List[float]]]:
        """
        Embeds a batch of texts in one request. Used to rebuild the semantic
        index from the conversation log in a handful of calls instead of one
        call per historical message.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        if not texts or not self.key_pool.current():
            return results

        pending = [i for i, t in enumerate(texts) if t and t.strip()]
        if not pending:
            return results

        for model in [self.model_name, self.backup_model]:
            if not pending:
                break
            payload = {
                "input": [texts[i].strip()[:4000] for i in pending],
                "model": model,
                "input_type": input_type,
                "encoding_format": "float"
            }
            for _ in range(max(len(self.key_pool), 1)):
                headers = {
                    "Authorization": f"Bearer {self.key_pool.current()}",
                    "Content-Type": "application/json"
                }
                try:
                    resp = requests.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                except Exception:
                    break
                if resp.status_code == 200:
                    data = resp.json().get("data", []) or []
                    for pos, item in enumerate(data):
                        idx = item.get("index", pos)
                        if not isinstance(idx, int) or idx < 0 or idx >= len(pending):
                            idx = pos
                        vec = item.get("embedding")
                        if vec and idx < len(pending):
                            results[pending[idx]] = [round(x, 5) for x in vec]
                    break
                if resp.status_code in (401, 403, 429) and len(self.key_pool) > 1:
                    self.key_pool.mark_bad()
                    continue
                break

        return results

    def _parse_history_blocks(self) -> List[Dict[str, Any]]:
        """
        Parses the durable plain-text log back into structured entries.
        The log is the source of truth for memory; the semantic index is a
        derived cache rebuilt from it.
        """
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return []

        if self.log_format == "events":
            raw_blocks = content.splitlines()
        else:
            raw_blocks = content.split("\n\n")

        blocks: List[Dict[str, Any]] = []
        for raw in raw_blocks:
            block = raw.strip()
            if not block:
                continue
            match = re.match(r"^\[(?P<ts>[^\]]+)\]\s*(?P<rest>.*)$", block, re.DOTALL)
            if not match:
                continue
            rest = match.group("rest").strip()
            if not rest:
                continue

            if self.log_format == "events":
                role, text = "event", rest
            else:
                role_match = re.match(r"^(?P<role>[A-Za-z_]+)\s*:\s*(?P<text>.*)$", rest, re.DOTALL)
                if not role_match:
                    continue
                role = role_match.group("role").lower()
                text = role_match.group("text").strip()

            if not text:
                continue
            blocks.append({
                "timestamp": match.group("ts"),
                "role": role,
                "text": text
            })
        return blocks

    def rebuild_vault_from_history(self, max_entries: int = 200, batch_size: int = 32) -> int:
        """
        Rebuilds the semantic index from the durable conversation log.

        The GitHub runner wipes its filesystem between jobs, so the vector
        index cannot be the source of truth. The plain-text log is committed
        to Git and this routine re-derives the index from it, which keeps the
        repository small while preserving recall across runner restarts.
        """
        blocks = self._parse_history_blocks()
        if not blocks:
            return 0
        blocks = blocks[-max_entries:]

        vectors: List[Optional[List[float]]] = []
        for start in range(0, len(blocks), batch_size):
            chunk = blocks[start:start + batch_size]
            vectors.extend(self.get_embeddings_batch([b["text"] for b in chunk]))

        entries: List[Dict[str, Any]] = []
        for block, vec in zip(blocks, vectors):
            payload, scale = vector_codec.encode_int8(vec) if vec else ("", 1.0)
            entries.append({
                "timestamp": block["timestamp"],
                "role": block["role"],
                "text": block["text"][:1500],
                "vector": payload or None,
                "vector_scale": scale,
                "vector_dim": len(vec) if vec else 0,
                "metadata": {"rebuilt_from_history": True}
            })

        with self._io_lock:
            self._vault_cache = entries
            self._save_vault()

        indexed = sum(1 for e in entries if e.get("vector"))
        print(f"[MemoryEngine] Semantic index rebuilt from history: {indexed}/{len(entries)} entries embedded.")
        return len(entries)

    def ensure_vault(self, force: bool = False, max_entries: int = 200) -> bool:
        """
        Makes sure a usable semantic index exists, rebuilding it from the
        conversation log when the vault is missing or older than the log.
        Returns True when a rebuild happened.
        """
        try:
            if not force and self.vault_file.exists() and self.history_file.exists():
                if self.vault_file.stat().st_mtime >= self.history_file.stat().st_mtime:
                    return False
        except OSError:
            pass

        try:
            self.rebuild_vault_from_history(max_entries=max_entries)
        except Exception as e:
            print(f"[MemoryEngine] Warning rebuilding vault from history: {e}")
            return False
        return True

    def append_message(self, role: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Appends a new turn to the plain text file and semantic vault.
        Reloads the vault from disk before mutating so concurrent writers
        (bot workers, test suite, backup job) never lose each other's entries.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        clean_text = text.strip()
        if not clean_text:
            return

        with self._io_lock:
            # 1. Plain Text Append (Zero overhead)
            if self.log_format == "events":
                log_entry = f"[{timestamp}] {clean_text}\n"
            else:
                log_entry = f"[{timestamp}] {role.upper()}: {clean_text}\n\n"
            try:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"[MemoryEngine] Error writing to history file: {e}")

            # 2. Semantic Embedding & Vault Indexing
            vec = self.get_embedding(clean_text, input_type="passage")
            payload, scale = vector_codec.encode_int8(vec) if vec else ("", 1.0)
            # Drop the in-memory cache so the vault is re-read from disk (multi-writer safe).
            self._vault_cache = None
            vault = self._load_vault()
            entry = {
                "timestamp": timestamp,
                "role": role,
                "text": clean_text[:1500], # Keep indexed chunk concise
                "vector": payload or None,
                "vector_scale": scale,
                "vector_dim": len(vec) if vec else 0,
                "metadata": metadata or {}
            }
            vault.append(entry)
            self._save_vault()

    def get_recent_turns(self, limit: int = 4) -> str:
        """
        Retrieves the last N messages from plain text log for immediate working memory.
        """
        if not self.history_file.exists():
            return ""

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                content = f.read()
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            recent_blocks = blocks[-limit:]
            return "\n\n".join(recent_blocks)
        except Exception:
            return ""

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        return vector_codec.cosine(vec_a, vec_b)

    @staticmethod
    def _tokenize(text: str) -> set:
        """Word tokens (unicode-aware, so Cyrillic works) of a piece of text."""
        return set(re.findall(r"[\w-]+", text.lower()))

    def search_memories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the vault for relevant past memories.

        Hybrid retrieval: semantic cosine similarity when the embedding
        provider is reachable, lexical token overlap always. Either signal on
        its own can surface a memory, so recall survives a provider outage, a
        supplier-side change of the embedding model, or a dimension change
        that would otherwise make every vector comparison meaningless.
        """
        vault = self._load_vault()
        if not vault:
            return []

        q_vec = self.get_embedding(query, input_type="query")
        q_tokens = self._tokenize(query)
        scored_entries = []

        for entry in vault:
            semantic = 0.0
            if q_vec:
                e_vec = self._entry_vector(entry)
                if e_vec:
                    semantic = self._cosine_similarity(q_vec, e_vec)

            lexical = 0.0
            if q_tokens:
                e_tokens = self._tokenize(entry.get("text", ""))
                lexical = len(q_tokens.intersection(e_tokens)) / len(q_tokens)

            score = max(semantic, lexical) if q_vec else lexical
            if score > 0.0:
                scored_entries.append((score, entry))

        # Sort descending by score
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for score, entry in scored_entries[:top_k] if score > 0.35]

    def build_memory_envelope(self, query: str, include_recent: bool = True) -> str:
        """
        Constructs a compact memory context combining relevant past memories
        and the recent conversation turns.
        """
        relevant_memories = self.search_memories(query, top_k=3)
        recent_turns = self.get_recent_turns(limit=4) if include_recent else ""

        parts = []
        if relevant_memories:
            parts.append("--- ETERNAL MEMORY (RELEVANT PAST FACTS) ---")
            for idx, m in enumerate(relevant_memories, 1):
                parts.append(f"[{m.get('timestamp')}] {m.get('role').upper()}: {m.get('text')}")

        if recent_turns:
            parts.append("\n--- IMMEDIATE WORKING CONTEXT (RECENT TURNS) ---")
            parts.append(recent_turns)

        return "\n".join(parts)

if __name__ == "__main__":
    engine = MemoryEngine()
    print("[*] Testing Memory Engine initialization...")
    engine.append_message("system", "Operational context window is 256k tokens. Hard ceiling is 262,144 tokens.")
    engine.append_message("user", "Какой у тебя размер контекстного окна?")
    engine.append_message("arbiter", "Верховный арбитр Qwen2.5-1.5B подтверждает: контекстное окно MiniMax M3.1 Flash равно 256,000 токенов, генерация до 128,000 токенов.")
    
    print("\n[*] Testing Recall for query 'размер контекстного окна'...")
    envelope = engine.build_memory_envelope("размер контекстного окна")
    print(envelope)
