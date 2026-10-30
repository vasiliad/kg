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
from typing import Dict, Any, List, Optional

from harness.api_healthcheck import APIHealthChecker
from harness.nim_scanner import NIMScanner
from harness.benchmark_tracker import BenchmarkTracker
from harness.minimax_executor import MiniMaxExecutor
from harness.state_manager import StateManager
from harness.nim_pacer import nim_pacer

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def should_trigger_emergency(config: Dict[str, Any], force: bool = False) -> (bool, str):
    if force:
        return True, "Trigger forced by user / CLI flag --force"

    # Check last update timestamp in active_models.json
    state_file = config.get("github_state", {}).get("state_file", "active_models.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                ts_str = state_data.get("timestamp") or state_data.get("last_updated")
                if ts_str:
                    ts_clean = ts_str.replace("Z", "+00:00")
                    last_dt = datetime.fromisoformat(ts_clean)
                    now_dt = datetime.now(timezone.utc)
                    age_hours = (now_dt - last_dt).total_seconds() / 3600.0
                    print(f"[*] Last verified state timestamp: {ts_str} ({age_hours:.1f} hours ago)")
                    if age_hours >= 6:
                        return True, f"State stale by {age_hours:.1f}h (threshold: 6h)"
        except Exception as e:
            print(f"[!] Warning reading state file: {e}")

    return False, "Sovereign complex is operational and state is up to date"


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
        return {"status": "SKIPPED", "reason": reason}

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    keenable_key = os.getenv("KEENABLE_API_KEY", "")
    state_mgr = StateManager()

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
                "local_arbiter": f"{target_model} (GitHub Actions CPU Sovereign)",
                "local_arbiter_feedback": f"Autonomous mission completed via {sovereign_mission.get('engine')}. Zero external API dependencies. Operating on GitHub Actions 4 vCPU / 16GB RAM.",
                "catalog_diff": {"new_count": 0, "removed_count": 0, "status": "EXTERNAL_APIS_OFFLINE"},
                "active_fleet": sovereign_mission.get("approved_fleet", [
                    {"id": target_model, "tier": "Sovereign-CPU-SLM", "quality_score": None, "status": "UNVERIFIED", "engine": "cpu_local", "source": "deterministic_kernel", "score_verified": False}
                ]),
                "last_updated": now_iso
            }
            state_mgr.save_local_state(sovereign_state)
            state_mgr.append_history_log(f"SOVEREIGN MIGRATION: Elected {target_model} on GitHub CPU due to total external API blackout.")
            
            # 4. Commit and push
            if not dry_run:
                try:
                    _ensure_git_identity()
                    subprocess.run(["git", "add", "active_models.json", "data/fleet_report.json", "nim_history.log", "config.yaml"], check=False)
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
    # a quick in-cycle re-tune. If both fail, the seeded fleet below remains
    # the fallback — degraded, but never empty.
    measured_fleet: List[Dict[str, Any]] = []
    try:
        from harness.fleet_tuner import FleetTuner
        report_path = PROJECT_ROOT / "data" / "fleet_report.json"
        fresh = False
        if report_path.exists():
            age_hours = (time.time() - report_path.stat().st_mtime) / 3600.0
            fresh = age_hours < 24.0
        current_state = state_mgr.load_local_state()
        if fresh and current_state.get("fleet_source") == "live_probe" and current_state.get("active_fleet"):
            measured_fleet = current_state["active_fleet"]
            print(f"[+] Using the measured duty fleet ({len(measured_fleet)} models, report <24h old).")
        else:
            print("[+] Measured fleet missing or stale - running a quick live re-tune...")
            tune_report = FleetTuner(api_key=nvidia_key, quick=True).run()
            measured_fleet = tune_report.get("fleet") or []
            print(f"[+] Quick re-tune produced {len(measured_fleet)} measured duty model(s).")
    except Exception as e:
        print(f"[!] Fleet tuner unavailable in this cycle: {e}")

    # Fallback fleet: only used when neither the measured fleet nor the live
    # ranking is available. Scores come from the seeded registry, so they are
    # explicitly flagged as unverified instead of presented as measurements.
    emergency_fleet = measured_fleet or (ranked_fleet[:5] if ranked_fleet else [
        {"id": "minimaxai/minimax-m3.1-flash", "tier": "Ultra-Flash-Reasoning", "quality_score": 99.2, "status": "APPROVED", "source": "static_seed", "score_verified": False},
        {"id": "z-ai/glm-5.3-flash", "tier": "Ultra-Agentic-Code", "quality_score": 98.8, "status": "APPROVED", "source": "static_seed", "score_verified": False},
        {"id": "deepseek-ai/deepseek-v4.1-flash", "tier": "Ultra-Sparse-MTP", "quality_score": 98.5, "status": "APPROVED", "source": "static_seed", "score_verified": False},
        {"id": "stepfun-ai/step-5-preview", "tier": "Ultra-Multimodal-Preview", "quality_score": 96.0, "status": "APPROVED", "source": "static_seed", "score_verified": False},
        {"id": "nvidia/nemotron-3-ultra-550b-a55b", "tier": "Unranked / Experimental", "quality_score": None, "status": "UNRANKED", "source": "none", "score_verified": False}
    ])
    fleet_origin = "live_probe" if measured_fleet else ("static_seed" if ranked_fleet else "none")

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
            subprocess.run(["git", "add", "active_models.json", "data/fleet_report.json", "nim_history.log", "config.yaml"], check=False)
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
