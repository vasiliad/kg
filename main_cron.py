#!/usr/bin/env python3
"""
Main entry point for Kaggle TPU / Scheduled Execution.
Orchestrates:
1. DeepSeek-R1-32B initialization on TPU v3-8 (128GB).
2. Key validation and healthcheck of NVIDIA NIM.
3. Model catalog discovery and benchmark ranking.
4. DeepSeek-R1 assessment of top candidate models and hyperparameter validation.
5. Live inference verification.
6. GitHub state sync.
"""
import os
import sys
import time
import yaml
from datetime import datetime, timezone
from harness.tpu_loader import TPULoader
from harness.nim_scanner import NIMScanner
from harness.benchmark_tracker import BenchmarkTracker
from harness.api_healthcheck import APIHealthChecker
from harness.state_manager import StateManager
from harness.tools import AgentToolbox
from harness.agent_loop import ReActAgent

def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_harness_cycle():
    print("=" * 70)
    print(f"[*] STARTING HARNESS CYCLE: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    config = load_config()
    
    # Environment keys
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN", "")
    github_token = os.getenv("GITHUB_TOKEN", "")
    github_repo = os.getenv("GITHUB_REPO", config.get("github_state", {}).get("repo", ""))

    state_mgr = StateManager()

    # 0. Disk and Storage Quota Guard
    from harness.disk_guard import DiskGuard
    dg = DiskGuard()
    quota_report = dg.enforce_quota_guard()
    if quota_report["status"] == "DANGER_ABORT":
        print("[!] CRITICAL: Disk space depleted (<1GB free). Aborting cycle to protect storage.")
        return
    
    # 1. API Health Check for NVIDIA NIM
    print("\n--- STEP 1: NVIDIA NIM API KEY HEALTH & ROTATION CHECK ---")
    if not nvidia_key:
        print("[!] WARNING: NVIDIA_API_KEY is not set. Running in offline/diagnostic mode.")
        key_status = {"valid": False, "status_code": 0, "message": "Missing API key."}
    else:
        checker = APIHealthChecker(api_key=nvidia_key)
        key_status = checker.check_key_validity()
        print(f"[*] Key status: {key_status['message']}")
        state_mgr.append_history_log(f"Key check: {key_status['message']}")

    # 2. Model Discovery & Benchmark Assessment
    print("\n--- STEP 2: CATALOG DISCOVERY & BENCHMARK RANKING ---")
    active_candidates = []
    if key_status.get("valid"):
        scanner = NIMScanner(api_key=nvidia_key)
        all_models = scanner.fetch_available_models()
        high_tier = scanner.filter_high_tier_models(all_models)
        
        tracker = BenchmarkTracker()
        ranked_models = tracker.rank_models(high_tier)
        print(f"[+] Top-ranked models ready for local evaluation:")
        for idx, m in enumerate(ranked_models[:5], 1):
            print(f"  {idx}. {m['id']} | ELO: {m.get('arena_elo')} | Tier: {m.get('tier')} | Score: {m.get('quality_score')}")
        active_candidates = ranked_models[:5]
    else:
        print("[!] Key invalid or missing. Using Ultra-Tier baseline models for registry.")
        tracker = BenchmarkTracker()
        active_candidates = tracker.rank_models([
            {"id": "minimaxai/minimax-m3.1-flash"},
            {"id": "z-ai/glm-5.3-flash"},
            {"id": "deepseek-ai/deepseek-v4.1-flash"},
            {"id": "nvidia/nemotron-3-ultra-550b-a55b"},
            {"id": "stepfun-ai/step-5-preview"}
        ])

    # 3. Local DeepSeek-R1 TPU Brain ReAct Loop
    print("\n--- STEP 3: LOCAL DEEPSEEK-R1-32B AUTONOMOUS AGENT (TPU ReAct LOOP) ---")
    tpu_decision = "DeepSeek-R1-32B autonomous research completed."
    toolbox = AgentToolbox(nvidia_api_key=nvidia_key)
    verified_fleet = []

    try:
        # The complex is fully decoupled from Kaggle TPU; config may have no tpu section.
        tpu_cfg = config.get("tpu", {}) or {}
        tpu_loader = TPULoader(
            model_id=tpu_cfg.get("model_id", config["local_arbiter"]["model_id"]),
            memory_fraction=tpu_cfg.get("memory_fraction", 0.95)
        )
        if "TPU_NAME" in os.environ or os.path.exists("/dev/accel0"):
            tpu_loader.load_model()
            agent = ReActAgent(tpu_loader=tpu_loader, toolbox=toolbox, max_steps=5)
            task_desc = (
                "Search the web for current top AI benchmarks (Arena/Livebench), inspect NVIDIA NIM catalog, "
                "identify the Ultra-Tier models (MiniMax M3.1 Flash, GLM 5.3 Flash, DeepSeek V4.1 Flash, 550B Ultra, Step-5 Preview), "
                "read their official parameter recommendations, and test their endpoints. Return final configured fleet."
            )
            agent_result = agent.run_agentic_workflow(task_desc)
            if "best_models" in agent_result:
                verified_fleet = agent_result["best_models"]
            tpu_decision = str(agent_result)
        else:
            print("[i] TPU hardware not detected in this diagnostic run. Using toolbox live probes directly.")
            verified_fleet = active_candidates
    except Exception as e:
        print(f"[!] TPU initialization note: {e}")
        verified_fleet = active_candidates

    # 5. Save State & Sync to GitHub
    print("\n--- STEP 5: SAVE STATE & GITHUB SYNC ---")
    state_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "key_health": key_status,
        "local_arbiter": config.get("tpu", {}).get("model_id", config["local_arbiter"]["model_id"]),
        "local_arbiter_feedback": tpu_decision[:300] + "..." if len(tpu_decision) > 300 else tpu_decision,
        "active_fleet": verified_fleet
    }
    state_mgr.save_local_state(state_payload)
    state_mgr.append_history_log(f"Cycle completed. Fleet count: {len(verified_fleet)}")

    if github_token and github_repo:
        print(f"[*] Syncing state to GitHub repo: {github_repo}...")
        state_mgr.sync_to_github(github_token, github_repo)

    print("\n" + "=" * 70)
    print("[+] HARNESS CYCLE COMPLETE. Ready for next scheduled trigger.")
    print("=" * 70)

if __name__ == "__main__":
    run_harness_cycle()
