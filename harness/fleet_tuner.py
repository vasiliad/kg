#!/usr/bin/env python3
"""
Autonomous Fleet Tuner: the complex measures the live catalogue itself and
promotes the models that ARE fast and smart right now.

Why this exists
---------------
The previous selection was a hardcoded keyword allowlist plus a static seed
registry ("score 99.2", source=static_seed, verified=False) printed in reports
as if it were a measurement. That cannot react to a supplier pumping a new
model, and it presented unverified numbers as facts.

What this module does instead — every cycle, without a human:

  1. CANDIDATE DISCOVERY   scan the live /v1/models catalogue and build the
                           candidate pool structurally (exclude embedders,
                           guards, reward/parse/vision-only models) ranked by
                           a prior. The config seeds the prior — it is the
                           starting point, never the verdict.
  2. LIVE MEASUREMENT      ping every candidate (reachability, latency,
                           Russian instruction adherence), then run objective
                           probes (math with a unique correct fraction, code
                           that must parse, JSON that must parse) on the top
                           survivors.
  3. PARAMETER SWEEP       for each selected model compare 2-3 sampling
                           parameter variants on the strict JSON task and
                           keep the measured winner.
  4. COMPOSITION           assign roles (duty_primary / duty_fast /
                           duty_code / duty_flagship / duty_fallback) with
                           vendor diversity so one supplier outage cannot
                           take the whole duty team down.
  5. PERSISTENCE           write active_models.json (fleet + measurements +
                           decision rationale), data/fleet_report.json and a
                           natural-language record into nim_history.log.

Failure contract: if there is no key, no network, or every probe fails, the
previous fleet is KEPT and marked degraded. A bad cycle can never blank the
duty team, and no score is ever invented.
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from harness.nim_pacer import nim_pacer
from harness.nim_scanner import NIMScanner, LOCAL_ONLY_MODEL_IDS
from harness.benchmark_tracker import BenchmarkTracker
from harness.state_manager import StateManager
from harness import fleet_probes

DEFAULT_POLICY = {
    "max_candidates": 18,        # pinged
    "quality_candidates": 6,     # get the full objective probe set
    "quality_tasks": ["reasoning", "code", "json"],
    "param_sweep": True,         # tune sampling parameters by measurement
    "min_score": 55.0,           # below this a model is not duty-eligible
    "min_params_b": 150.0,       # hard floor: models under 150B never enter the pool
    "fast_latency_cap_sec": 12.0,
    "max_fleet_size": 5,
    "probe_timeout_sec": 45,
    "heavy_timeout_sec": 75,
    "max_newcomers": 4,          # guaranteed probe slots for catalogue newcomers
    "rotation_slots": 4,         # never-measured models pulled in per cycle
    "weights": {"quality": 0.55, "speed": 0.25, "reliability": 0.20},
}

# Structurally NOT chat-reasoning models: embeddings, safety guards, reward
# models, parsers, retrievers, translators, image/video models. This is a
# *purpose* filter, not a brand allowlist — any new chat model passes.
EXCLUDED_PURPOSE_MARKERS = (
    "embed", "nemoguard", "guard", "safety", "reward", "parse",
    "nvclip", "deplot", "video-detector", "ising", "riva-translate",
    "retriever", "rerank", "vlm-embed", "cosmos-reason", "clip",
)

# Capability priors — used ONLY to decide who gets probed first, never as a
# quality score. The measured probe decides who actually serves.
CAPABILITY_PRIOR = {
    "ultra": 5.0, "super": 4.0, "nemotron": 3.0, "reasoning": 3.0,
    "r1": 3.0, "flash": 2.0, "pro": 2.0, "max": 2.0, "large": 2.0,
    "instruct": 1.0, "omni": -2.0, "preview": -1.0,
}
HEAVY_MARKERS = ("ultra", "550b", "253b", "340b", "-super-", "1t-")

# Parameter-count bonus: a trillion-parameter newcomer ("mistral-large-4-1t")
# must outrank a 3B legacy model in the probe queue, otherwise strong releases
# wait behind incumbent names. Capped so it is a prior, not a verdict.
SIZE_BONUS_RE = re.compile(r"(\d+(?:\.\d+)?)(tb|t|b)(?=$|[-_a-z])")


def parsed_params_b(model_id: str) -> Optional[float]:
    """
    Total parameter count encoded in the model id, in billions.
    "mistral-large-4-1t-a52b" -> 1000 (the TOTAL, not the 52B active part),
    "nemotron-3-ultra-550b-a55b" -> 550. None when the id carries no size.
    """
    best: Optional[float] = None
    for m in SIZE_BONUS_RE.finditer((model_id or "").lower()):
        val = float(m.group(1))
        unit = m.group(2)
        params_b = val * 1000.0 if unit in ("tb", "t") else val
        best = params_b if best is None else max(best, params_b)
    return best


def _size_bonus(model_id: str) -> float:
    best = 0.0
    for m in SIZE_BONUS_RE.finditer((model_id or "").lower()):
        val = float(m.group(1))
        unit = m.group(2)
        # "1t"/"1tb" = 1T params (1000B); "550b" = 550B. Plain version tags
        # like "v1"/"v0.1" never match (a bare digit needs a unit after it).
        params_b = val * 1000.0 if unit in ("tb", "t") else val
        best = max(best, min(6.0, params_b / 100.0))
    return best

ROLE_LABELS = {
    "duty_primary": "Дежурный основной",
    "duty_fast": "Дежурный быстрый",
    "duty_code": "Дежурный кодер",
    "duty_flagship": "Дежурный флагман",
    "duty_fallback": "Дежурный резерв",
}


def vendor_of(model_id: str) -> str:
    """The party that can break all at once — used for fleet diversity."""
    return (model_id or "").split("/", 1)[0].lower()


def _heavy(model_id: str) -> bool:
    low = (model_id or "").lower()
    return any(m in low for m in HEAVY_MARKERS)


class FleetTuner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        quick: bool = False,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ):
        self.api_key = (api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.quick = quick
        self.policy = dict(DEFAULT_POLICY)
        if config and isinstance(config.get("fleet_tuner"), dict):
            for k, v in config["fleet_tuner"].items():
                if k in self.policy:
                    self.policy[k] = v
        if quick:
            self.policy["max_candidates"] = min(8, self.policy["max_candidates"])
            self.policy["quality_candidates"] = min(3, self.policy["quality_candidates"])
            self.policy["quality_tasks"] = ["reasoning"]
            self.policy["param_sweep"] = False
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.tracker = BenchmarkTracker()
        self.state_mgr = StateManager()
        self._log: List[str] = []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def say(self, msg: str) -> None:
        print(f"[FleetTuner] {msg}")
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Stage 1: candidate discovery (structural, no brand allowlist)
    # ------------------------------------------------------------------
    def candidate_prior(self, model_id: str, is_current_fleet: bool, is_seed: bool) -> float:
        low = (model_id or "").lower()
        prior = _size_bonus(low)
        for marker, weight in CAPABILITY_PRIOR.items():
            if marker in low:
                prior += weight
        if is_current_fleet:
            prior += 10.0   # re-measure whoever serves now — continuity
        if is_seed:
            prior += 8.0    # the config seed is the starting point to verify
        return prior

    def build_candidate_pool(
        self,
        live_models: List[Dict[str, Any]],
        previous_fleet_ids: Optional[List[str]] = None,
        must_probe: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns the models worth measuring: everything chat-like that is not
        structurally excluded, ranked by prior, capped to max_candidates.

        `must_probe` ids bypass the prior ranking entirely — catalogue
        newcomers and never-measured models get guaranteed slots, so a strong
        release (e.g. a new trillion-param flagship) can never wait behind
        incumbent names.
        """
        previous = {str(x) for x in (previous_fleet_ids or [])}
        seed_families = set(self.tracker.quality_baseline.keys())
        forced: List[str] = []
        for mid in (must_probe or []):
            if mid and mid not in forced:
                forced.append(mid)

        pool: List[Dict[str, Any]] = []
        seen: set = set()

        def add_candidate(mid: str) -> None:
            if not mid or mid in seen:
                return
            low = mid.lower()
            if mid in LOCAL_ONLY_MODEL_IDS or low in {x.lower() for x in LOCAL_ONLY_MODEL_IDS}:
                return
            if any(marker in low for marker in EXCLUDED_PURPOSE_MARKERS):
                return
            floor = float(self.policy.get("min_params_b") or 0.0)
            size = parsed_params_b(mid)
            if size is not None and size < floor:
                # Hard size floor (operator decision 2026-10-31): anything at
                # or under ~120B never enters the pool. Ids WITHOUT a size
                # token stay eligible — they cannot be judged by name, the
                # probe decides (e.g. minimax/glm/deepseek carry no size).
                self._skipped_small = getattr(self, "_skipped_small", 0) + 1
                return
            family = self.tracker.get_model_reputation(mid).get("matched_model")
            is_seed = family in seed_families and family != mid
            prior = self.candidate_prior(mid, mid in previous, is_seed)
            pool.append({
                "id": mid,
                "vendor": vendor_of(mid),
                "prior": prior,
                "heavy": _heavy(mid),
                "recommended_params": NIMScanner(self.api_key)._get_recommended_params(mid)
                if self.api_key else {"temperature": 0.6, "top_p": 0.9, "max_tokens": 2048},
            })
            seen.add(mid)

        for mid in forced:
            add_candidate(mid)

        rest: List[Dict[str, Any]] = []
        floor = float(self.policy.get("min_params_b") or 0.0)
        for m in live_models:
            mid = m.get("id", "") if isinstance(m, dict) else ""
            if mid and mid not in seen:
                low = mid.lower()
                if mid in LOCAL_ONLY_MODEL_IDS or low in {x.lower() for x in LOCAL_ONLY_MODEL_IDS}:
                    continue
                if any(marker in low for marker in EXCLUDED_PURPOSE_MARKERS):
                    continue
                size = parsed_params_b(mid)
                if size is not None and size < floor:
                    continue
                family = self.tracker.get_model_reputation(mid).get("matched_model")
                is_seed = family in seed_families and family != mid
                rest.append({
                    "id": mid, "vendor": vendor_of(mid),
                    "prior": self.candidate_prior(mid, mid in previous, is_seed),
                    "heavy": _heavy(mid),
                    "recommended_params": NIMScanner(self.api_key)._get_recommended_params(mid)
                    if self.api_key else {"temperature": 0.6, "top_p": 0.9, "max_tokens": 2048},
                })

        rest.sort(key=lambda c: c["prior"], reverse=True)
        cap = self.policy["max_candidates"]
        for c in rest[: max(0, cap - len(pool))]:
            add_candidate(c["id"])

        if len(rest) + len(forced) > cap:
            self.say(
                f"Candidate pool: {len(forced)} guaranteed slots (newcomers/rotation) "
                f"+ top {max(0, cap - len(pool))} of {len(rest)} by prior."
            )
        return pool[:cap]

    # ------------------------------------------------------------------
    # Stage 2: live measurement
    # ------------------------------------------------------------------
    def _post_chat(self, model_id: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        One streaming chat request. Streaming is mandatory for reasoning
        models (non-streaming POSTs hang until the socket times out on long
        CoT — a ground-truth rule already learned by the executor). Measures
        total latency and time-to-first-token separately.
        """
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            # stream must be in the BODY, not just the HTTP call: without it
            # NIM returns one plain JSON document, the SSE parser finds no
            # chunks, and every probe looks "EMPTY" (the second live run's
            # exact failure mode; the fail-safe kept the fleet meanwhile).
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        timeout = self.policy["heavy_timeout_sec"] if _heavy(model_id) else self.policy["probe_timeout_sec"]

        for attempt in (1, 2):
            nim_pacer.wait_for_slot()
            start = time.time()
            try:
                res = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=timeout)
            except requests.exceptions.Timeout:
                return {"ok": False, "status": "TIMEOUT", "latency_sec": float(timeout), "ttft_sec": float(timeout), "text": ""}
            except Exception as e:
                return {"ok": False, "status": f"NETWORK_ERROR:{type(e).__name__}", "latency_sec": round(time.time() - start, 2), "ttft_sec": 0.0, "text": ""}
            latency = round(time.time() - start, 2)

            if res.status_code == 429 and attempt == 1:
                time.sleep(6)
                continue
            if res.status_code == 401:
                return {"ok": False, "status": "AUTH_401", "latency_sec": latency, "ttft_sec": 0.0, "text": ""}
            if res.status_code != 200:
                return {"ok": False, "status": f"HTTP_{res.status_code}", "latency_sec": latency, "ttft_sec": 0.0, "text": ""}

            content, reasoning = [], []
            first_at = None
            try:
                for line in res.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data_body = line_str[6:]
                    if data_body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_body)
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {}) or {}
                        c = delta.get("content")
                        r = delta.get("reasoning_content")
                        if (c or r) and first_at is None:
                            first_at = time.time() - start
                        if c:
                            content.append(c)
                        if r:
                            reasoning.append(r)
                    except Exception:
                        continue
            except Exception as e:
                return {"ok": False, "status": f"STREAM_ERROR:{type(e).__name__}", "latency_sec": round(time.time() - start, 2), "ttft_sec": 0.0, "text": ""}

            total = round(time.time() - start, 2)
            text = ("".join(content)).strip() or ("".join(reasoning)).strip()
            return {
                "ok": bool(text),
                "status": "OK" if text else "EMPTY",
                "latency_sec": total,
                "ttft_sec": round(first_at, 2) if first_at else total,
                "text": text,
            }

        return {"ok": False, "status": "RATE_LIMITED", "latency_sec": timeout, "ttft_sec": 0.0, "text": ""}

    def ping_probe(self, model_id: str) -> Dict[str, Any]:
        """Cheap probe: reachable? latency? answers in Russian at all?"""
        res = self._post_chat(model_id, fleet_probes.PING_PROMPT, max_tokens=128, temperature=0.2)
        if not res["ok"] and res["status"] == "EMPTY":
            # Reasoning models may spend the whole ping budget on CoT. One
            # adaptive retry with a 4x budget prevents a false negative.
            retry = self._post_chat(model_id, fleet_probes.PING_PROMPT, max_tokens=512, temperature=0.2)
            if retry["ok"]:
                res = {**retry, "adaptive_retry": True}
        score, verdict = fleet_probes.score_probe("instruction", res.get("text", ""))
        res["instruction_score"] = score
        res["instruction_verdict"] = verdict
        return res

    def quality_probe(self, model_id: str, task_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        task = fleet_probes.PROBE_TASKS[task_name]
        max_tokens = int(task["max_tokens"])
        res = self._post_chat(
            model_id, task["prompt"],
            max_tokens=max_tokens,
            temperature=float(params.get("temperature", 0.2)),
        )
        if not res["ok"] and res["status"] == "EMPTY":
            # Deep-CoT models exhaust a small budget while thinking (MiniMax
            # ground rule: >= 2560 tokens). One 4x-budget retry keeps the
            # measurement fair instead of punishing the reasoning style.
            retry = self._post_chat(
                model_id, task["prompt"],
                max_tokens=min(max_tokens * 4, 8192),
                temperature=float(params.get("temperature", 0.2)),
            )
            if retry["ok"]:
                res = {**retry, "adaptive_retry": True}
        score, verdict = fleet_probes.score_probe(task_name, res.get("text", ""))
        res["task"] = task_name
        res["score"] = score
        res["verdict"] = verdict
        return res

    def sweep_params(self, model_id: str, task_name: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        The strict-JSON task is the parameter arena: short answers, objective
        parse. Tries the supplier-recommended temperature and a deterministic
        one; keeps the measured winner.
        """
        variants = [
            ("recommended", {"temperature": float(base_params.get("temperature", 0.6)), "top_p": float(base_params.get("top_p", 0.9))}),
            ("deterministic", {"temperature": 0.2, "top_p": 0.9}),
        ]
        best: Optional[Dict[str, Any]] = None
        for label, params in variants:
            res = self.quality_probe(model_id, task_name, params)
            res["variant"] = label
            if best is None or res["score"] > best["score"] or (
                res["score"] == best["score"] and res["latency_sec"] < best["latency_sec"]
            ):
                best = res
        return {
            "params": variants[best["variant"] == "deterministic"][1] if best else base_params,
            "score": best["score"] if best else 0.0,
            "latency_sec": best["latency_sec"] if best else 0.0,
            "variant": best["variant"] if best else "none",
        }

    # ------------------------------------------------------------------
    # Scoring and composition
    # ------------------------------------------------------------------
    def composite(self, quality: float, latency: float, ok_rate: float) -> Tuple[float, float, float]:
        p = self.policy["weights"]
        fast_cap = 30.0
        speed = max(0.0, min(1.0, 1.0 - (max(latency, 0.5) - 2.0) / fast_cap))
        total = p["quality"] * quality + p["speed"] * speed + p["reliability"] * ok_rate
        return round(100.0 * total, 2), round(speed, 3), round(quality, 3)

    def evaluate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Full measurement pass. Returns {model_id: measurement}.
        Candidates that fail the ping are recorded as failed — they are data,
        not silence: the decision log must show WHY a model was rejected.
        """
        measurements: Dict[str, Dict[str, Any]] = {}

        self.say(f"Stage A: pinging {len(candidates)} candidates...")
        for c in candidates:
            ping = self.ping_probe(c["id"])
            measurements[c["id"]] = {
                "id": c["id"], "vendor": c["vendor"], "prior": c["prior"],
                "ping": ping, "status": ping["status"],
                "latency_sec": ping["latency_sec"],
                "instruction_score": ping.get("instruction_score", 0.0),
            }
            mark = "OK" if ping["ok"] else ping["status"]
            self.say(f"  ping {c['id']}: {mark} ({ping['latency_sec']}s)")

        alive = [m for m in measurements.values() if m["ping"]["ok"]]
        # Fastest survivors first — quality probes cost real minutes.
        alive.sort(key=lambda m: (m["ping"]["latency_sec"], -m["prior"]))
        top = alive[: self.policy["quality_candidates"]]
        if len(alive) > len(top):
            self.say(f"Stage B: full objective probes on the {len(top)} fastest of {len(alive)} alive.")

        tasks = list(self.policy["quality_tasks"])
        weights = fleet_probes.probe_weights(tasks + ["instruction"])
        for m in top:
            mid = m["id"]
            task_scores: Dict[str, float] = {"instruction": m["instruction_score"]}
            task_verdicts: Dict[str, str] = {"instruction": m["ping"].get("instruction_verdict", "")}
            oks, total = 1, 1  # the ping already succeeded
            for task_name in tasks:
                res = self.quality_probe(mid, task_name, {"temperature": 0.2})
                task_scores[task_name] = res["score"]
                task_verdicts[task_name] = res["verdict"]
                oks += 1 if res["ok"] else 0
                total += 1
                self.say(f"  probe {mid}/{task_name}: {res['verdict']} (score={res['score']})")
            quality = sum(weights[t] * task_scores.get(t, 0.0) for t in weights)
            ok_rate = oks / total
            final, speed, _ = self.composite(quality, m["ping"]["latency_sec"], ok_rate)

            seed = self.tracker.get_model_reputation(mid)
            m.update({
                "task_scores": {k: round(v, 3) for k, v in task_scores.items()},
                "task_verdicts": task_verdicts,
                "quality": round(quality, 3),
                "speed_score": speed,
                "ok_rate": round(ok_rate, 3),
                "final_score": final,
                "seed_prior": {"tier": seed.get("tier"), "arena_elo": seed.get("arena_elo")} if seed.get("score_source") == "static_seed" else None,
                "status": "MEASURED",
            })

            # Parameter sweep: pick sampling parameters by measurement.
            if self.policy.get("param_sweep") and final >= self.policy["min_score"]:
                base_params = next((c["recommended_params"] for c in candidates if c["id"] == mid), {})
                sweep = self.sweep_params(mid, "json", base_params)
                m["recommended_params"] = {
                    "temperature": sweep["params"]["temperature"],
                    "top_p": sweep["params"]["top_p"],
                    "max_tokens": 4096,
                }
                m["param_sweep"] = {
                    "task": "json", "variant": sweep["variant"],
                    "score": sweep["score"], "latency_sec": sweep["latency_sec"],
                }
                self.say(f"  params {mid}: variant={sweep['variant']} (json score={sweep['score']})")

        for m in measurements.values():
            if m.get("status") != "MEASURED":
                m["final_score"] = None
                m["status"] = f"FAILED_{m['status']}"
        return measurements

    def compose_fleet(self, measurements: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Role assignment with vendor diversity. Roles:
          duty_primary  - best measured composite
          duty_fast     - best composite under the latency cap
          duty_code     - best code probe
          duty_flagship - best composite among heavy/flagship models
          duty_fallback - best composite from a vendor not yet in the fleet
        """
        measured = [m for m in measurements.values() if m.get("status") == "MEASURED"]
        eligible = [m for m in measured if (m.get("final_score") or 0) >= self.policy["min_score"]]
        notes: List[str] = []
        if not eligible:
            return [], notes

        def best(pool, key):
            return max(pool, key=key) if pool else None

        fleet: Dict[str, Dict[str, Any]] = {}
        used_vendors: List[str] = []

        def take(m, role):
            if not m or m["id"] in fleet:
                return
            m.setdefault("roles", []).append(role)
            fleet[m["id"]] = m
            used_vendors.append(m["vendor"])

        def best(pool, key):
            """Prefers models not already in the fleet, so distinct roles land
            on distinct models instead of collapsing onto one tied winner."""
            free = [m for m in pool if m["id"] not in fleet]
            return max(free or pool, key=key) if (free or pool) else None

        eligible.sort(key=lambda m: m["final_score"], reverse=True)
        take(best(eligible, lambda m: m["final_score"]), "duty_primary")

        fast_pool = [m for m in eligible if m["ping"]["latency_sec"] <= self.policy["fast_latency_cap_sec"]]
        take(best(fast_pool, lambda m: m["final_score"]), "duty_fast")

        # A specialised role requires measured competence in it: a model that
        # scored 0 on the code probe must not be appointed "duty_code" just
        # because everyone else is already in the fleet.
        code_pool = [m for m in eligible if m["task_scores"].get("code", 0.0) >= 0.5]
        take(best(code_pool, lambda m: m["task_scores"].get("code", 0.0)), "duty_code")

        flagship = [m for m in eligible if _heavy(m["id"]) or (m.get("seed_prior") or {}).get("tier", "").startswith("Ultra")]
        take(best(flagship, lambda m: m["final_score"]), "duty_flagship")

        rest = [m for m in eligible if m["vendor"] not in used_vendors]
        take(best(rest, lambda m: m["final_score"]), "duty_fallback")

        # Fill remaining slots by score, keeping vendor diversity.
        for m in eligible:
            if len(fleet) >= self.policy["max_fleet_size"]:
                break
            if m["id"] not in fleet and (not used_vendors or m["vendor"] not in used_vendors or len(fleet) < 3):
                m.setdefault("roles", []).append("duty_fallback" if m["vendor"] not in used_vendors else "reserve")
                fleet[m["id"]] = m
                used_vendors.append(m["vendor"])

        chosen = sorted(fleet.values(), key=lambda m: m["final_score"], reverse=True)
        for m in chosen:
            notes.append(
                f"{m['id']} -> {'/'.join(m['roles'])} | score={m['final_score']} "
                f"(задачи: {m['task_scores']}, задержка {m['ping']['latency_sec']}с, ok={m['ok_rate']})"
            )
        return chosen, notes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def build_fleet_entry(self, m: Dict[str, Any]) -> Dict[str, Any]:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        roles = m.get("roles") or ["reserve"]
        return {
            "id": m["id"],
            "owned_by": m["vendor"],
            "tier": ROLE_LABELS.get(roles[0], roles[0]),
            "roles": roles,
            "quality_score": m["final_score"],
            "score_source": "live_probe",
            "score_verified": True,
            "score_as_of": now_iso,
            "status": "APPROVED",
            "measured": {
                "latency_sec": m["ping"]["latency_sec"],
                "ok_rate": m["ok_rate"],
                "speed_score": m["speed_score"],
                "task_scores": m["task_scores"],
                "task_verdicts": m["task_verdicts"],
            },
            "recommended_params": m.get("recommended_params") or {"temperature": 0.6, "top_p": 0.9, "max_tokens": 4096},
            "param_sweep": m.get("param_sweep"),
            "seed_prior": m.get("seed_prior"),
        }

    def enable_offline_stub(self) -> None:
        """
        Deterministic stub probes: exercises the full measure->score->compose
        pipeline with no network and no key (CI, local logic tests). Answers
        are canned per task family; only the pipeline logic is under test.
        """
        def stub(model_id: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
            low = (prompt or "").lower()
            mid = (model_id or "").lower()
            if "готовность" in low:
                return {"ok": True, "status": "OK", "latency_sec": 3.0,
                        "text": "готовность — состояние полной готовности к действию"}
            if "шар" in low or "вероятность" in low:
                text = "Всего пар 66, разных цветов 47. Итог: 47/66"
                return {"ok": True, "status": "OK", "latency_sec": 9.0 if "550b" in mid else 14.0, "text": text}
            if "asyncio" in low:
                return {"ok": True, "status": "OK", "latency_sec": 8.0,
                        "text": ("```python\nimport asyncio\nasync def check_urls(urls):\n"
                                 "    sem = asyncio.Semaphore(5)\n    try:\n"
                                 "        async with sem:\n            pass\n"
                                 "    except Exception:\n        pass\n```")}
            if "json" in low or "извлеки" in low:
                return {"ok": True, "status": "OK", "latency_sec": 4.0,
                        "text": ('{"company": "NVIDIA Corporation", "model": "Nemotron-4 340B",'
                                 ' "release_date": "June 2024", "context_tokens": 4096}')}
            return {"ok": False, "status": "EMPTY", "latency_sec": 5.0, "text": ""}
        self._post_chat = stub

    def run(self, dry_run: bool = False, offline_catalog: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes one tuning cycle. offline_catalog: list of model ids to use
        instead of the live API (local testing / CI without a key).
        """
        started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.say(f"Tuning cycle started at {started} (quick={self.quick})")

        state = self.state_mgr.load_local_state()
        previous_fleet = [m.get("id") for m in state.get("active_fleet", []) if isinstance(m, dict) and m.get("id")]

        if not self.api_key and offline_catalog is None:
            self.say("No NVIDIA_API_KEY: keeping the previous fleet, marking degraded.")
            return self._finish(state, None, reason="NO_API_KEY", dry_run=dry_run)

        if offline_catalog is not None:
            # Accept both plain ids and {"id": ...} dicts — the CLI passes the
            # committed snapshot (strings), tests may pass richer fixtures.
            live_models = [
                {"id": (m if isinstance(m, str) else m.get("id"))}
                for m in offline_catalog if m
            ]
            self.say(f"Offline catalog supplied: {len(live_models)} ids (no network probes will pass without a key).")
        else:
            scanner = NIMScanner(api_key=self.api_key or "none", base_url=self.base_url)
            live_models = scanner.fetch_available_models()
            if not live_models:
                self.say("Live catalogue unavailable: keeping the previous fleet.")
                return self._finish(state, None, reason="CATALOG_UNAVAILABLE", dry_run=dry_run)

        # Guaranteed probe slots: (a) catalogue newcomers since the last
        # snapshot, (b) a rotation through models never measured before, so
        # the whole chat-like catalogue gets probed over successive cycles.
        # A strong release (e.g. a fresh trillion-param flagship) therefore
        # cannot wait behind incumbent names.
        live_ids = {m.get("id") for m in live_models if isinstance(m, dict) and m.get("id")}
        try:
            previous_snapshot = {str(x) for x in (state.get("catalog_snapshot") or [])}
            measured_before = {
                m.get("id") for m in (state.get("fleet_measurements") or [])
                if isinstance(m, dict) and m.get("id")
            }
        except Exception:
            previous_snapshot, measured_before = set(), set()

        def _chat_like(mid: str) -> bool:
            low = (mid or "").lower()
            return (mid not in LOCAL_ONLY_MODEL_IDS
                    and not any(marker in low for marker in EXCLUDED_PURPOSE_MARKERS))

        must_probe: List[str] = [
            mid for mid in sorted(live_ids - previous_snapshot) if _chat_like(mid)
        ][: int(self.policy["max_newcomers"])]
        unmeasured = [
            mid for mid in sorted(live_ids - measured_before - set(must_probe)) if _chat_like(mid)
        ]
        must_probe.extend(unmeasured[: max(0, int(self.policy["rotation_slots"]) - len(must_probe))])
        if must_probe:
            self.say(f"Guaranteed probe slots: {must_probe}")

        candidates = self.build_candidate_pool(live_models, previous_fleet_ids=previous_fleet, must_probe=must_probe)
        self.say(f"Candidate pool ({len(candidates)}): {[c['id'] for c in candidates]}")

        measurements = self.evaluate(candidates)
        chosen, notes = self.compose_fleet(measurements)

        if not chosen:
            self.say("No candidate passed the measured minimum: KEEPING the previous fleet.")
            return self._finish(state, {"measurements": measurements, "candidates": candidates}, reason="NO_QUALIFIED", dry_run=dry_run)

        fleet_entries = [self.build_fleet_entry(m) for m in chosen]
        for n in notes:
            self.say(f"  role: {n}")

        report = {
            "tuned_at": started,
            "method": "live_probe" if offline_catalog is None else "offline_dry_run",
            "quick": self.quick,
            "policy": self.policy,
            "candidates": [c["id"] for c in candidates],
            "measurements": [
                {
                    "id": m["id"], "vendor": m["vendor"], "prior": m["prior"], "status": m["status"],
                    "final_score": m.get("final_score"),
                    "latency_sec": m.get("ping", {}).get("latency_sec"),
                    "task_scores": m.get("task_scores"),
                    "roles": m.get("roles", []),
                }
                for m in measurements.values()
            ],
            "fleet": fleet_entries,
            "decision_log": self._log[-40:],
        }

        if dry_run or offline_catalog is not None:
            self.say("Dry-run/offline: state NOT modified.")
            return report

        state["active_fleet"] = fleet_entries
        state["fleet_source"] = "live_probe"
        state["fleet_decision"] = {
            "tuned_at": started,
            "method": "live_probe",
            "roles": {e["id"]: e["roles"] for e in fleet_entries},
            "log": self._log[-40:],
        }
        state["fleet_measurements"] = report["measurements"]
        self.state_mgr.save_local_state(state)

        summary = ", ".join(f"{e['id']} ({e['tier']}, {e['quality_score']})" for e in fleet_entries)
        self.state_mgr.append_history_log(
            f"ФЛОТ ПЕРЕОБРАН | метод=живой замер | кандидатов={len(candidates)} | "
            f"измерено={sum(1 for m in measurements.values() if m.get('status') == 'MEASURED')} | состав={summary}"
        )

        report_path = PROJECT_ROOT / "data" / "fleet_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.say(f"Report written to {report_path}")
        return report

    def _finish(self, state: Dict[str, Any], data: Optional[Dict[str, Any]], reason: str, dry_run: bool) -> Dict[str, Any]:
        """
        Degraded outcome: the previous fleet stays in place and the reason is
        recorded. A failed tuning cycle must never blank the duty team.
        """
        self.say(f"Cycle finished with status {reason}.")
        if not dry_run:
            self.state_mgr.append_history_log(f"ФЛОТ НЕ ПЕРЕОБРАН | причина={reason} | предыдущий состав сохранён")
            state["fleet_source"] = f"previous ({reason})"
        return {"status": reason, "fleet_kept": state.get("active_fleet", []), "log": self._log[-20:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Fleet Tuner")
    parser.add_argument("--quick", action="store_true", help="reduced probe budget (emergency mode)")
    parser.add_argument("--dry-run", action="store_true", help="measure but do not persist")
    parser.add_argument("--offline", action="store_true", help="use the committed catalog snapshot, no network")
    args = parser.parse_args()

    offline_ids: Optional[List[str]] = None
    if args.offline:
        try:
            with open(PROJECT_ROOT / "active_models.json", "r", encoding="utf-8") as f:
                offline_ids = json.load(f).get("catalog_snapshot") or []
        except Exception:
            offline_ids = []

    tuner = FleetTuner(quick=args.quick)
    if args.offline:
        tuner.enable_offline_stub()
    report = tuner.run(dry_run=args.dry_run, offline_catalog=offline_ids)
    fleet = report.get("fleet") or report.get("fleet_kept") or []
    print("\n=== RESULT ===")
    for e in fleet:
        print(f"  {e.get('id')} | {e.get('tier')} | score={e.get('quality_score')} | verified={e.get('score_verified')}")


if __name__ == "__main__":
    main()
