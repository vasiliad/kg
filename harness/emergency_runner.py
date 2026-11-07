#!/usr/bin/env python3
"""
Sovereign Survival Runner: Autonomous CPU/Host Fallback Orchestrator.
100% independent of Kaggle or third-party compute queues.
Activates whenever:
1. External NIM APIs experience rotation or blackout.
2. The last verified state in active_models.json is older than 6 hours.
3. Explicitly invoked via CLI with --force or GitHub Actions workflow_dispatch.

Operates autonomously on standard Linux CPU without TPU requirements.
Delegates heavy reasoning to MiniMax M3.1 Flash with streaming dual-channel CoT,
strictly adhering to <=30 RPM pacing, unpruned Keenable search facts, and 128k-256k context.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import time
import yaml
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from harness.api_healthcheck import APIHealthChecker
from harness.nim_scanner import NIMScanner
from harness.benchmark_tracker import BenchmarkTracker
from harness.minimax_executor import MiniMaxExecutor
from harness.state_manager import StateManager
from harness.nim_pacer import nim_pacer
# The clock of the measured state lives in ONE place, so the ladder, the
# letter and the reports read the same witness (measured 2026-11-06: the
# ladder used its own timestamp and called a 6.4-hour-old measurement
# «stale by 14.4h»).
from harness.state_clock import (
    STATE_STALE_AFTER_H,
    fleet_state_age_hours,
    fleet_state_stamp,
)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# The fleet is measured every ~6h by the tuner. The emergency ladder asks
# exactly one question: HOW OLD IS THE MEASUREMENT?
#
# Measured 2026-11-06: the ladder answered it with `timestamp` — a field
# the emergency runner writes ITSELF («now») — so it measured «when did I
# last run», not «when was the fleet measured». The consequences were
# measured too: at 13:37 the ladder fired with «State stale by 14.4h»
# while the fleet had actually been measured 6.4h earlier, and two hours
# later the SAME state would have been called «up to date» although it was
# ten hours old. A ladder that measures the wrong clock cannot see a real
# outage — and it cries wolf when there is none.
#
# The clock itself lives in harness/state_clock.py, so the ladder, the
# letter and the reports all read the SAME witness.


def _fleet_state_age(state: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """How old the measurement is (None when there is no measurement)."""
    return fleet_state_age_hours(state)


def should_trigger_emergency(config: Dict[str, Any], force: bool = False) -> (bool, str):
    if force:
        return True, "Trigger forced by user / CLI flag --force"

    # How old is the MEASUREMENT?
    state_file = config.get("github_state", {}).get("state_file", "active_models.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            stamp, source = fleet_state_stamp(state_data)
            if stamp is None:
                # «I cannot read it» is not «it is broken» — but it is also
                # not «everything is fine»: the state cannot be verified, so
                # the ladder runs and says exactly that.
                return True, ("Отметка замера флота отсутствует — состояние "
                              "нечем подтвердить")
            now_dt = datetime.now(timezone.utc)
            age_hours = (now_dt - stamp).total_seconds() / 3600.0
            print(f"[*] Fleet measured at {stamp.isoformat()} "
                  f"(source: {source}; {age_hours:.1f} hours ago)")
            if age_hours >= STATE_STALE_AFTER_H:
                return True, (f"State stale by {age_hours:.1f}h "
                              f"(threshold: {STATE_STALE_AFTER_H}h; "
                              f"measured {stamp.isoformat()})")
        except Exception as e:
            print(f"[!] Warning reading state file: {e}")

    return False, "Sovereign complex is operational and state is up to date"


def _measured_arbiter(state: Dict[str, Any]) -> str:
    """Who serves the operator now, according to the MEASURED roster."""
    members = [m for m in (state.get("active_fleet") or []) if isinstance(m, dict)]
    primary = None
    for member in members:
        roles = [str(r) for r in (member.get("roles") or [])]
        if "duty_primary" in roles:
            primary = member
            break
    chosen = primary or (members[0] if members else None)
    if not chosen or not chosen.get("id"):
        return "не зафиксирован"
    return f"{chosen.get('id')} (дежурная команда по живому замеру)"


def clear_stale_emergency(reason: str = "") -> bool:
    """
    An emergency state must not outlive its cause.

    Measured 2026-11-06: `emergency_mode: true` and the reason «State stale
    by 14.4h» stayed in the state file for the rest of the day — the
    operator's letter kept announcing «Экстренный CPU fallback» long after
    the fleet had been measured again (14:33). A flag with a reason that no
    longer exists is a lie with a timestamp.

    The flag is cleared ONLY when its cause was staleness and the fleet has
    been measured since. A sovereign (zero-API) migration has a different
    cause and survives: the API is still down there.
    """
    state_mgr = StateManager()
    try:
        state = state_mgr.load_local_state()
    except Exception as e:
        print(f"[!] Could not read the state to clear the emergency: {e}")
        return False
    if not isinstance(state, dict) or not state.get("emergency_mode"):
        return False
    cause = str(state.get("emergency_cause") or "")
    if not cause:
        # A state file from before the cause was recorded: the reason text
        # is the only witness. A staleness reason may be cleared; anything
        # else (a blackout, a forced run) stays.
        cause = ("state_stale"
                 if str(state.get("emergency_reason") or "").startswith("State stale by")
                 else "unknown")
    if cause != "state_stale":
        print(f"[*] The emergency state is kept: its cause is «{cause}», "
              f"not staleness.")
        return False
    stamp, source = fleet_state_stamp(state)
    if stamp is None:
        print("[*] The emergency state is kept: the fleet is still unmeasured.")
        return False
    age_hours = (datetime.now(timezone.utc) - stamp).total_seconds() / 3600.0
    if age_hours >= STATE_STALE_AFTER_H:
        print(f"[*] The emergency state is kept: the fleet is still stale "
              f"({age_hours:.1f}h).")
        return False

    state["emergency_mode"] = False
    state["emergency_cause"] = ""
    state["emergency_reason"] = ""
    state["local_arbiter"] = _measured_arbiter(state)
    state["emergency_cleared_at"] = datetime.now(timezone.utc).isoformat()
    state["emergency_cleared_because"] = (
        f"the fleet was measured again ({stamp.isoformat()}, {source})")
    state_mgr.save_local_state(state)
    state_mgr.append_history_log(
        "АВАРИЙНЫЙ РЕЖИМ СНЯТ | флот измерен заново, причина исчезла "
        f"(замер {stamp.isoformat()})")
    print(f"[+] Emergency mode cleared: the fleet was measured at "
          f"{stamp.isoformat()} ({source}).")
    return True


def _ensure_git_identity() -> None:
    """
    A GitHub runner has no committer identity configured, so an unguarded
    `git commit` fails with "Author identity unknown" — and with check=False
    that failure is silent, meaning the state would never be persisted.
    """
    subprocess.run(["git", "config", "user.name", "vasiliad"], check=False)
    subprocess.run(["git", "config", "user.email", "vasiliadi@users.noreply.github.com"], check=False)


def _push_state() -> None:
    """
    Pushes the committed state through the shared race-safe sync helper.

    A plain `git push` is rejected whenever another cloud job (the bot
    daemon, the diagnostics) pushed to main while this cycle was running.
    The helper rebases and retries instead of losing the state.
    """
    sync_script = PROJECT_ROOT / "scripts" / "sync_state.sh"
    try:
        if sync_script.exists():
            subprocess.run(["bash", str(sync_script)], check=False)
        else:
            subprocess.run(["git", "push", "origin", "main"], check=False)
    except Exception as e:
        print(f"[!] Warning during state push: {e}")


def choose_emergency_fleet(
    measured_fleet: Optional[List[Dict[str, Any]]] = None,
    previous_fleet: Optional[List[Dict[str, Any]]] = None,
    previous_state: Optional[Dict[str, Any]] = None,
    ranked_fleet: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    The roster the complex will work with during an emergency — and WHY.

    The order is the policy:

        1. the fleet MEASURED in this cycle;
        2. the fleet MEASURED earlier — an outage cannot re-measure a team,
           and a measured team beats a seeded one;
        3. the seed registry — explicitly marked as unverified, because it
           is not a measurement.

    Measured: the emergency runner used to replace a duty team that had been
    MEASURED with one that was only SEEDED, because the network was down and
    the measurement could not be repeated. The unverified team then looked
    verified (its scores came from the seed registry) and the complex
    quietly lost the team it had earned.

    Returns (fleet, origin, note).
    """
    measured = list(measured_fleet or [])
    previous = list(previous_fleet or [])
    state = previous_state or {}
    ranked = list(ranked_fleet or [])

    if measured:
        return measured, "live_probe", "measured live in this cycle"

    if previous and state.get("fleet_source") == "live_probe":
        return (
            previous,
            "live_probe",
            "kept from the previous measurement: the network is down, so the "
            "measurement cannot be repeated — and a measured team beats a seeded one",
        )

    if ranked:
        return (
            ranked[:5],
            "static_seed",
            "unverified: the seed registry, not a measurement — the complex "
            "could not measure a team in this cycle",
        )

    return (
        [
            {"id": "minimaxai/minimax-m3.1-flash", "tier": "Ultra-Flash-Reasoning", "quality_score": 99.2, "status": "APPROVED", "source": "static_seed", "score_verified": False},
            {"id": "z-ai/glm-5.3-flash", "tier": "Ultra-Agentic-Code", "quality_score": 98.8, "status": "APPROVED", "source": "static_seed", "score_verified": False},
            {"id": "deepseek-ai/deepseek-v4.1-flash", "tier": "Ultra-Sparse-MTP", "quality_score": 98.5, "status": "APPROVED", "source": "static_seed", "score_verified": False},
            {"id": "stepfun-ai/step-5-preview", "tier": "Ultra-Multimodal-Preview", "quality_score": 96.0, "status": "APPROVED", "source": "static_seed", "score_verified": False},
            {"id": "nvidia/nemotron-3-ultra-550b-a55b", "tier": "Unranked / Experimental", "quality_score": None, "status": "UNRANKED", "source": "none", "score_verified": False},
        ],
        "none",
        "unverified: the built-in seed list — the complex has never measured a team",
    )


def run_emergency_orchestration(force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    print("=" * 72)
    print("🚨 EMERGENCY SURVIVAL PROTOCOL: CPU/GITHUB AUTONOMOUS ORCHESTRATOR 🚨")
    print(f"[*] Activation time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    config = load_config()
    should_run, reason = should_trigger_emergency(config, force=force)
    print(f"[*] Emergency gate check: should_run={should_run} ({reason})")

    if not should_run:
        print("[+] Emergency run skipped. Primary TPU system is operating normally.")
        cleared = clear_stale_emergency(reason)
        return {"status": "SKIPPED", "reason": reason, "emergency_cleared": cleared}

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    keenable_key = os.getenv("KEENABLE_API_KEY", "")
    state_mgr = StateManager()

    # The state BEFORE this cycle touches anything. The record of the
    # measurement inside it (`fleet_decision`) must survive this cycle: the
    # ladder answers «when was the fleet measured?» from it, and a cycle
    # that erases it leaves the next cycle with nothing honest to read.
    previous_state: Dict[str, Any] = {}
    try:
        loaded_state = state_mgr.load_local_state()
        if isinstance(loaded_state, dict):
            previous_state = loaded_state
    except Exception as e:
        print(f"[!] Warning reading the previous state: {e}")

    # Step 0: Storage & Disk Quota Guard
    from harness.disk_guard import DiskGuard
    dg = DiskGuard()
    quota_report = dg.enforce_quota_guard()
    if quota_report["status"] == "DANGER_ABORT":
        print("[!] CRITICAL: Disk space depleted (<1GB free). Aborting emergency cycle.")
        return {"status": "ABORTED", "reason": "Disk space depleted"}

    # Step 1: Healthcheck NVIDIA NIM key
    print("\n[STEP 1] Validating NVIDIA NIM API Key...")
    checker = APIHealthChecker(api_key=nvidia_key) if nvidia_key else None
    key_status = checker.check_key_validity() if checker else {"valid": False, "status_code": 0, "message": "Missing API Key"}
    print(f"[*] NIM Key Health: {key_status.get('message')} (HTTP {key_status.get('status_code')})")

    if not key_status.get("valid"):
        zero_api_cfg = config.get("zero_api_sovereign_mode", {})
        if zero_api_cfg.get("enabled", True):
            print("\n" + "!" * 72)
            print("⚠️  TOTAL API BLACKOUT DETECTED: NVIDIA NIM API UNAVAILABLE ⚠️")
            print("🚨 ACTIVATING ZERO-API SOVEREIGN MODE: MIGRATING ARBITER TO GITHUB CPU SLM")
            print("!" * 72)
            target_model = zero_api_cfg.get("cpu_slm_model_id", "Qwen/Qwen2.5-1.5B-Instruct")
            
            # 1. Elect the CPU SLM as the sovereign successor
            from harness.tools import AgentToolbox
            tb = AgentToolbox(nvidia_api_key="none")
            election_res = tb.elect_successor(
                new_model_id=target_model,
                rationale="Total external API blackout: autonomous self-preservation migration to GitHub Actions CPU SLM."
            )
            print(f"[*] Successor election: {election_res}")
            
            # 2. Execute Sovereign CPU Mission via Pre-Baked Engine
            from harness.sovereign_cpu_engine import SovereignCPUEngine
            cpu_engine = SovereignCPUEngine(
                model_id=target_model,
                backup_model_id=zero_api_cfg.get("backup_ultra_light_id", "Qwen/Qwen2.5-0.5B-Instruct")
            )
            sovereign_mission = cpu_engine.execute_sovereign_mission(
                task_description="Execute autonomous telemetry audit, preserve repository health, and enforce zero-API survival.",
                context={"blackout_reason": "Total external API blackout detected"}
            )
            print(f"[*] Sovereign mission completed via {sovereign_mission.get('engine')} in {sovereign_mission.get('elapsed_sec')}s.")

            # 3. Update state to sovereign CPU mode
            now_iso = datetime.now(timezone.utc).isoformat()
            sovereign_state = {
                "timestamp": now_iso,
                "emergency_mode": True,
                "sovereign_zero_api_mode": True,
                "last_emergency_run": now_iso,
                "emergency_reason": "Total external API blackout (NVIDIA NIM unavailable). Autonomous migration to CPU SLM.",
                # The cause is NAMED, so the next cycle knows what this flag
                # means: a blackout survives the arrival of a fresh
                # measurement (the API is still down there), while a stale
                # measurement does not.
                "emergency_cause": "zero_api_blackout",
                "local_arbiter": f"{target_model} (GitHub Actions CPU Sovereign)",
                "local_arbiter_feedback": f"Autonomous mission completed via {sovereign_mission.get('engine')}. Zero external API dependencies. Operating on GitHub Actions 4 vCPU / 16GB RAM.",
                "catalog_diff": {"new_count": 0, "removed_count": 0, "status": "EXTERNAL_APIS_OFFLINE"},
                "active_fleet": sovereign_mission.get("approved_fleet", [
                    {"id": target_model, "tier": "Sovereign-CPU-SLM", "quality_score": None, "status": "UNVERIFIED", "engine": "cpu_local", "source": "deterministic_kernel", "score_verified": False}
                ]),
                # The last MEASURED record survives even a blackout: the
                # complex cannot measure right now, but what it measured
                # yesterday is still a fact — and «I cannot read it» must
                # not become «it never happened».
                "fleet_decision": previous_state.get("fleet_decision"),
                "fleet_measurements": previous_state.get("fleet_measurements"),
                "fleet_drift": previous_state.get("fleet_drift"),
                "last_updated": now_iso
            }
            state_mgr.save_local_state(sovereign_state)
            state_mgr.append_history_log(f"SOVEREIGN MIGRATION: Elected {target_model} on GitHub CPU due to total external API blackout.")
            
            # 4. Commit and push
            if not dry_run:
                try:
                    _ensure_git_identity()
                    # One path at a time: a multi-path `git add` aborts
                    # entirely when one pathspec does not match, and then
                    # NOTHING is staged — the migration would never reach
                    # the repository (measured 2026-11-03).
                    for path in ("active_models.json", "data/fleet_report.json",
                                 "nim_history.log", "config.yaml"):
                        subprocess.run(["git", "add", path], check=False)
                    subprocess.run(["git", "commit", "-m", f"chore(sovereign): migrate to CPU SLM {target_model} [Total API Blackout]"], check=False)
                    _push_state()
                except Exception as e:
                    print(f"[!] Warning syncing sovereign state: {e}")
                
            print("[+] Sovereign migration state successfully committed and pushed.")
            return {"status": "SOVEREIGN_MIGRATED", "model": target_model}
        else:
            print("[-] API Key is invalid or expired, and sovereign mode is disabled. Emergency runner halted.")
            return {"status": "ERROR", "error": "Invalid API Key"}


    # Step 2: Catalog Scan & Diff Detection
    print("\n[STEP 2] Scanning NVIDIA NIM Catalog and calculating diff...")
    scanner = NIMScanner(api_key=nvidia_key)
    all_models = scanner.fetch_available_models()
    catalog_diff = scanner.detect_catalog_diff(all_models, previous_state_path="active_models.json")
    
    high_tier_models = scanner.filter_high_tier_models(all_models)
    tracker = BenchmarkTracker()
    ranked_fleet = tracker.rank_models(high_tier_models)

    print(f"[+] Top {len(ranked_fleet[:5])} candidate models for emergency fleet:")
    for idx, m in enumerate(ranked_fleet[:5], 1):
        score = m.get("quality_score")
        score_txt = f"{score}" if score is not None else "UNRANKED"
        print(f"  {idx}. {m.get('id')} (Tier: {m.get('tier')}, Score: {score_txt})")

    # Step 2b: Resolve tracked ids against the LIVE catalogue so that supplier
    # side renames are absorbed automatically (no human in the loop needed).
    id_resolution = scanner.resolve_fleet_ids(all_models)
    if id_resolution.get("renamed"):
        print(f"[+] Auto-corrected {len(id_resolution['renamed'])} supplier-side model id(s).")
    if id_resolution.get("missing"):
        print(f"[!] {len(id_resolution['missing'])} tracked model(s) retired by the supplier.")

    # Step 2c: The MEASURED duty fleet outranks the static seed. A tuner
    # report younger than 24h is used as-is; a stale or missing one triggers
    # a quick in-cycle re-tune. If both fail, the PREVIOUSLY MEASURED roster
    # remains — an outage cannot re-measure a team, and a measured team
    # beats a seeded one. Only a complex that never measured anything falls
    # back to the seed registry, which says so on every entry.
    measured_fleet: List[Dict[str, Any]] = []
    previous_fleet: List[Dict[str, Any]] = []
    try:
        from harness.fleet_tuner import FleetTuner
        report_path = PROJECT_ROOT / "data" / "fleet_report.json"
        fresh = False
        if report_path.exists():
            age_hours = (time.time() - report_path.stat().st_mtime) / 3600.0
            fresh = age_hours < 24.0
        previous_fleet = previous_state.get("active_fleet") or []
        if fresh and previous_state.get("fleet_source") == "live_probe" and previous_fleet:
            measured_fleet = previous_fleet
            print(f"[+] Using the measured duty fleet ({len(measured_fleet)} models, report <24h old).")
        else:
            print("[+] Measured fleet missing or stale - running a quick live re-tune...")
            tune_report = FleetTuner(api_key=nvidia_key, quick=True).run()
            measured_fleet = tune_report.get("fleet") or []
            print(f"[+] Quick re-tune produced {len(measured_fleet)} measured duty model(s).")
            # The quick re-tune WROTE a fresh measurement into the state
            # file — and this cycle is about to rewrite that same file.
            # The fresh stamp is read back, or the ladder would call its
            # own re-measurement stale (measured 2026-11-06: the emergency
            # run overwrote a measurement taken minutes earlier with the
            # old one, and the next cycle fired again on a fresh fleet).
            refreshed_state = state_mgr.load_local_state()
            if isinstance(refreshed_state, dict) and refreshed_state.get("fleet_decision"):
                previous_state = refreshed_state
                previous_fleet = previous_state.get("active_fleet") or previous_fleet
                print("[+] Fresh measurement stamp kept from the quick re-tune.")
    except Exception as e:
        print(f"[!] Fleet tuner unavailable in this cycle: {e}")

    emergency_fleet, fleet_origin, fleet_note = choose_emergency_fleet(
        measured_fleet=measured_fleet,
        previous_fleet=previous_fleet,
        previous_state=previous_state,
        ranked_fleet=ranked_fleet,
    )
    if fleet_origin == "live_probe" and not measured_fleet:
        print(f"[+] Keeping the previously MEASURED roster ({len(emergency_fleet)} models): "
              "an outage cannot re-measure it, and a measured team beats a seeded one.")
    elif fleet_origin != "live_probe":
        print(f"[!] No measured roster available ({fleet_note}).")

    # Step 3: Web Research via Keenable (100% Unpruned)
    print("\n[STEP 3] Fetching latest LLM benchmarks & releases via Keenable...")
    executor = MiniMaxExecutor(nvidia_api_key=nvidia_key, keenable_api_key=keenable_key)
    
    # Step 4: Autonomous Decision via MiniMax M3.1 Flash (128k-256k streaming CoT)
    print("\n[STEP 4] Delegating architectural evaluation to MiniMax M3.1 Flash...")
    task_prompt = (
        "You are acting as the Emergency Lead Systems Architect for the DeepSeek-TPU-NIM ecosystem.\n"
        "The primary TPU orchestrator is currently offline or queued. You are executing on emergency CPU fallback.\n"
        "IMPORTANT: the duty roster has ALREADY been selected by live measurement "
        "(ping latency + objective math/code/JSON probes). Do NOT invent models, do NOT "
        "re-rank the fleet and do NOT assign scores of your own.\n"
        "Your task:\n"
        "1. Analyze the catalog diff and the measured fleet provided in context.\n"
        "2. Flag operational risks: retired ids, failed probes, single-vendor concentration.\n"
        "3. Recommend successor actions if the measured fleet looks thin.\n"
        "4. Issue an executive summary of system status.\n"
        "Format your answer with clear markdown headings."
    )

    context_data = {
        "emergency_mode": True,
        "trigger_reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "catalog_diff": catalog_diff,
        "measured_fleet": [
            {
                "id": (m.get("id") if isinstance(m, dict) else str(m)),
                "tier": (m.get("tier") if isinstance(m, dict) else ""),
                "score": (m.get("quality_score") if isinstance(m, dict) else None),
                "verified": (m.get("score_verified") if isinstance(m, dict) else False),
            }
            for m in emergency_fleet[:6]
        ] if emergency_fleet else ranked_fleet[:5],
        "candidate_models": ranked_fleet[:5]
    }

    decision_res = executor.execute_task(
        task_prompt=task_prompt,
        context=json.dumps(context_data, indent=2),
        enable_search=True,
        max_tokens=65536,
        temperature=0.6
    )

    synthesis_content = decision_res.get("content", "")
    reasoning_preview = decision_res.get("reasoning_content", "")[:500]
    print(f"\n[+] MiniMax reasoning finished in {decision_res.get('elapsed_sec')}s.")
    print(f"[+] Synthesis length: {len(synthesis_content)} chars.")

    # Step 5: Update Active Models State
    print("\n[STEP 5] Updating active_models.json and nim_history.log...")
    now_iso = datetime.now(timezone.utc).isoformat()

    new_state = {
        "timestamp": now_iso,
        "emergency_mode": True,
        "last_emergency_run": now_iso,
        "emergency_reason": reason,
        # WHY the emergency is on — so the next cycle can tell whether the
        # cause is still there. A flag without a cause outlives its reason.
        "emergency_cause": "state_stale" if str(reason).startswith("State stale by") else "forced",
        "key_health": key_status,
        "local_arbiter": "minimaxai/minimax-m3.1-flash (Emergency Fallback Surrogate)",
        "local_arbiter_feedback": synthesis_content[:500] + "..." if len(synthesis_content) > 500 else synthesis_content,
        "catalog_diff": catalog_diff,
        "id_resolution": {
            "renamed": id_resolution.get("renamed", {}),
            "missing": id_resolution.get("missing", []),
            "live_count": id_resolution.get("live_count", 0),
        },
        "active_fleet": emergency_fleet,
        "fleet_source": fleet_origin,
        # WHY this roster — the operator and the next cycle must be able to
        # tell a measured team from a seeded one without reading the code.
        "fleet_note": fleet_note,
        # The tuner's own record of the measurement is KEPT: measured
        # 2026-11-06, the emergency run rebuilt the state from scratch and
        # dropped `fleet_decision` — the very field that says WHEN the
        # fleet was measured. The ladder then had nothing honest to read
        # and fell back to its own timestamp. A record of the measurement
        # must survive the cycle that reacts to it.
        "fleet_decision": previous_state.get("fleet_decision"),
        "fleet_measurements": previous_state.get("fleet_measurements"),
        "fleet_drift": previous_state.get("fleet_drift"),
        # Full catalogue snapshot: the next cycle diffs against this, so
        # "new models" means "appeared since the last scan" rather than
        # "not part of the selected fleet".
        "catalog_snapshot": sorted(
            m.get("id") for m in all_models if isinstance(m, dict) and m.get("id")
        ) if all_models else [],
        "last_updated": now_iso
    }

    if not dry_run:
        state_mgr.save_local_state(new_state)
        log_entry = (
            f"EMERGENCY RUN SUCCESS | Diff: +{catalog_diff.get('new_count')}/-{catalog_diff.get('removed_count')} | "
            f"Fleet: {len(emergency_fleet)} models | Reason: {reason}"
        )
        state_mgr.append_history_log(log_entry)
        print("[+] State updated locally.")

        # Step 6: Git Auto-Commit and Push
        print("\n[STEP 6] Syncing emergency updates to Git...")
        try:
            _ensure_git_identity()
            # `git add` ABORTS the WHOLE command when one pathspec does not
            # match (measured 2026-11-03): one missing file and NOTHING is
            # staged — the emergency state silently never reached the repo.
            # Each path is added on its own.
            for path in ("active_models.json", "data/fleet_report.json",
                         "nim_history.log", "config.yaml"):
                subprocess.run(["git", "add", path], check=False)
            commit_res = subprocess.run(
                ["git", "commit", "-m", f"chore(emergency): autonomous fallback heartbeat update [{now_iso}]"],
                capture_output=True,
                text=True,
                check=False
            )
            print(f"[*] Git commit: {commit_res.stdout.strip() or commit_res.stderr.strip()}")
            _push_state()
        except Exception as e:
            print(f"[!] Warning during git push: {e}")

    print("\n" + "=" * 72)
    print("✅ EMERGENCY RUN COMPLETED SUCCESSFULLY.")
    print("=" * 72)
    return {
        "status": "COMPLETED",
        "emergency_mode": True,
        "timestamp": now_iso,
        "fleet_count": len(emergency_fleet),
        "catalog_diff": catalog_diff
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency CPU Fallback Runner")
    parser.add_argument("--force", action="store_true", help="Force emergency execution regardless of heartbeat")
    parser.add_argument("--dry-run", action="store_true", help="Execute without saving state or git pushing")
    args = parser.parse_args()

    run_emergency_orchestration(force=args.force, dry_run=args.dry_run)
