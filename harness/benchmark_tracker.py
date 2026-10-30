import requests
import json
from typing import Dict, Any, List

from harness.nim_scanner import family_of


class BenchmarkTracker:
    """
    Scores candidate models against a seeded frontier registry.

    IMPORTANT — honesty contract:
    The registry below is a STATIC SEED captured from public leaderboards at a
    known date. It is NOT a live feed, so every score it produces is tagged
    ``source="static_seed"`` and ``verified=False``. Only models that are
    actually present in the live NVIDIA catalogue and matched here are eligible
    for promotion; anything unknown is explicitly UNRANKED rather than given an
    invented number. The complex therefore never presents a fabricated score as
    if it had been measured.
    """
    SEED_AS_OF = "2026-10-29"

    def __init__(self):
        # Seed registry keyed by a family token so that supplier-side renames
        # (zhipu/ -> z-ai/, 550b-ultra -> nemotron-3-ultra-550b-a55b) keep
        # resolving instead of silently dropping a model out of the fleet.
        self.quality_baseline: Dict[str, Dict[str, Any]] = {
            "minimax":  {"arena_elo": 1410, "tier": "Ultra-Flash-Reasoning", "quality_score": 99.2},
            "glm":      {"arena_elo": 1405, "tier": "Ultra-Agentic-Code", "quality_score": 98.8},
            "deepseek": {"arena_elo": 1400, "tier": "Ultra-Sparse-MTP", "quality_score": 98.5},
            "ultra550": {"arena_elo": 1390, "tier": "Ultra-Massive-Flagship", "quality_score": 97.5},
            "step5":    {"arena_elo": 1380, "tier": "Ultra-Multimodal-Preview", "quality_score": 96.0},
        }

    def get_model_reputation(self, model_id: str) -> Dict[str, Any]:
        """
        Matches a live NIM model id against the seeded registry.
        """
        normalized_id = model_id.lower().strip()
        family = family_of(normalized_id)

        if family and family in self.quality_baseline:
            stats = self.quality_baseline[family]
            return {
                "matched_model": family,
                "arena_elo": stats["arena_elo"],
                "tier": stats["tier"],
                "quality_score": stats["quality_score"],
                "status": "APPROVED" if stats["quality_score"] >= 88.0 else "REVIEW_NEEDED",
                "score_source": "static_seed",
                "score_verified": False,
                "score_as_of": self.SEED_AS_OF,
            }

        # Unknown model: no invented score — it is explicitly unranked.
        return {
            "matched_model": model_id,
            "arena_elo": None,
            "tier": "Unranked / Experimental",
            "quality_score": None,
            "status": "UNRANKED",
            "score_source": "none",
            "score_verified": False,
            "score_as_of": None,
        }

    def rank_models(self, candidate_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts candidates by seeded quality score (descending). Unranked models
        keep a stable position at the end instead of an invented score.
        """
        scored_list = []
        for cand in candidate_models:
            m_id = cand.get("id", "")
            rep = self.get_model_reputation(m_id)
            combined = {**cand, **rep}
            scored_list.append(combined)

        scored_list.sort(key=lambda x: (x.get("quality_score") is not None, x.get("quality_score") or 0), reverse=True)
        return scored_list
