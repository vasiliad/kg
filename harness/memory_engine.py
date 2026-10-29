#!/usr/bin/env python3
"""
Eternal Memory Engine for Autonomous Complex.
Combines ultra-compact persistent text storage (data/telegram_history.txt)
with state-of-the-art semantic vector retrieval using NVIDIA Nemotron 3.5 Embed.

Features:
1. Zero-bloat text logging: entire conversation history stored in human-readable plain text.
2. Semantic indexing via NVIDIA NIM: nvidia/nemotron-3.5-embed-8b (or 1b).
3. Pure Python vector similarity (cosine) without heavy external database dependencies.
4. Sovereign fallback: falls back to keyword/token overlap if API is unavailable.
5. Context synthesis: builds compact, highly relevant memory envelopes for LLMs to prevent context degradation.
"""

import os
import sys
import json
import math
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

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
        backup_model: str = "nvidia/nemotron-3.5-embed-1b"
    ):
        self.history_file = history_file or (PROJECT_ROOT / "data" / "telegram_history.txt")
        self.vault_file = vault_file or (PROJECT_ROOT / "data" / ".memory_vault.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
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
        self._vault_cache: Optional[List[Dict[str, Any]]] = None

    def _load_vault(self) -> List[Dict[str, Any]]:
        if self._vault_cache is not None:
            return self._vault_cache

        if self.vault_file.exists():
            try:
                with open(self.vault_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._vault_cache = data.get("entries", [])
                    return self._vault_cache
            except Exception as e:
                print(f"[MemoryEngine] Warning reading vault: {e}")
        self._vault_cache = []
        return self._vault_cache

    def _save_vault(self) -> None:
        if self._vault_cache is None:
            return
        try:
            # Keep vault compact: max 1000 indexed entries
            if len(self._vault_cache) > 1000:
                self._vault_cache = self._vault_cache[-1000:]
            with open(self.vault_file, "w", encoding="utf-8") as f:
                json.dump({"entries": self._vault_cache}, f, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryEngine] Warning saving vault: {e}")

    def get_embedding(self, text: str, input_type: str = "passage") -> Optional[List[float]]:
        """
        Retrieves embedding vector via NVIDIA NIM API using Nemotron.
        """
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
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
            try:
                resp = requests.post(f"{self.base_url}/embeddings", headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    vec = data["data"][0]["embedding"]
                    # Round floats to 5 decimals to minimize file size by 60%
                    return [round(x, 5) for x in vec]
            except Exception:
                continue

        return None

    def append_message(self, role: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Appends a new turn to the plain text file and semantic vault.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        clean_text = text.strip()
        if not clean_text:
            return

        # 1. Plain Text Append (Zero overhead)
        log_entry = f"[{timestamp}] {role.upper()}: {clean_text}\n\n"
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[MemoryEngine] Error writing to history file: {e}")

        # 2. Semantic Embedding & Vault Indexing
        vec = self.get_embedding(clean_text, input_type="passage")
        vault = self._load_vault()
        entry = {
            "timestamp": timestamp,
            "role": role,
            "text": clean_text[:1500], # Keep indexed chunk concise
            "vector": vec,
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
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search_memories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Semantically searches the vault for relevant past memories.
        Falls back to keyword overlap if embeddings are unavailable.
        """
        vault = self._load_vault()
        if not vault:
            return []

        q_vec = self.get_embedding(query, input_type="query")
        scored_entries = []

        if q_vec:
            # High-precision vector cosine similarity
            for entry in vault:
                e_vec = entry.get("vector")
                if e_vec:
                    score = self._cosine_similarity(q_vec, e_vec)
                    scored_entries.append((score, entry))
        else:
            # Fallback: token overlap score
            q_tokens = set(query.lower().split())
            for entry in vault:
                e_tokens = set(entry.get("text", "").lower().split())
                overlap = len(q_tokens.intersection(e_tokens))
                score = overlap / max(len(q_tokens), 1)
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
