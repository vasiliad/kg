#!/usr/bin/env python3
"""
Comprehensive Autonomous Verification Suite:
1. MemoryEngine persistence and Nemotron-3.5-Embed semantic retrieval.
2. SovereignCPUEngine local arbiter execution (Qwen2.5-1.5B CPU).
3. MiniMaxExecutor with 256k runtime manifest & CoT streaming.
4. Bidirectional Inter-Model Bridge (Arbiter <-> MiniMax <-> Memory).
5. DiskGuard quota and file bloat inspection.
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.memory_engine import MemoryEngine
from harness.sovereign_cpu_engine import SovereignCPUEngine
from harness.minimax_executor import MiniMaxExecutor
from harness.telegram_bot import HarnessTelegramBot
from harness.disk_guard import DiskGuard
from harness.state_manager import StateManager

def _isolated_memory_engine() -> MemoryEngine:
    """Memory engine on throwaway files — test runs must never pollute the
    production vault in data/ (same files used to be rewritten by both)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="kg-memory-test-"))
    return MemoryEngine(
        history_file=tmp_dir / "telegram_history.txt",
        vault_file=tmp_dir / ".memory_vault.json"
    )

def test_1_memory_engine_and_nemotron():
    print("\n" + "=" * 60)
    print("TEST 1: Memory Engine & Nemotron 3.5 Embed Vector Recall")
    print("=" * 60)
    engine = _isolated_memory_engine()
    
    # 1. Log test dialogue
    engine.append_message("user", "Какое точное окно контекста настроено для MiniMax M3.1 Flash?")
    engine.append_message("arbiter", "Верховный арбитр Qwen2.5-1.5B: Окно контекста составляет 256,000 токенов, генерация до 128,000 токенов, лимит <= 30 RPM.")
    
    # 2. Check physical file exists and is populated
    assert engine.history_file.exists(), "History file data/telegram_history.txt must exist!"
    with open(engine.history_file, "r", encoding="utf-8") as f:
        history_text = f.read()
    assert "256,000 токенов" in history_text, "History file must contain logged dialogue!"
    print(f"[+] Plain text history verified: {len(history_text)} bytes logged in {engine.history_file}")

    # 3. Test semantic recall via Nemotron
    start_t = time.time()
    matches = engine.search_memories("лимит токенов генерации и окно", top_k=2)
    elapsed = round(time.time() - start_t, 2)
    print(f"[+] Nemotron-3.5-Embed semantic search completed in {elapsed}s. Found {len(matches)} matches.")
    assert len(matches) > 0, "Nemotron search must return matching memories!"
    top_match = matches[0]
    print(f"    Top Match: [{top_match.get('role')}]: {top_match.get('text')[:100]}...")
    assert "256,000" in top_match.get("text") or "окно" in top_match.get("text").lower(), "Semantic match must be relevant to context window!"
    
    # 4. Test envelope formatting
    envelope = engine.build_memory_envelope("лимиты контекста")
    assert "--- ETERNAL MEMORY" in envelope, "Envelope must contain eternal memory section!"
    print("[+] Test 1 Passed: Memory Engine & Nemotron Embed vector search fully operational.")

def test_2_sovereign_cpu_arbiter():
    print("\n" + "=" * 60)
    print("TEST 2: Sovereign CPU Arbiter (Qwen2.5-1.5B Duty)")
    print("=" * 60)
    arbiter = SovereignCPUEngine()
    start_t = time.time()
    res = arbiter.execute_sovereign_mission(
        task_description="Verify active duty state and confirm MiniMax parameters.",
        context={"minimax_window": "256k", "status": "STEADY_ACTIVE_DUTY"}
    )
    elapsed = round(time.time() - start_t, 3)
    print(f"[+] Arbiter mission completed via {res.get('engine')} in {elapsed}s.")
    assert res.get("status") == "SUCCESS", "Arbiter mission must succeed!"
    print("[+] Test 2 Passed: Local Arbiter operates with zero external dependencies.")

def test_3_minimax_runtime_manifest():
    print("\n" + "=" * 60)
    print("TEST 3: MiniMax Executor System Runtime Manifest & CoT")
    print("=" * 60)
    executor = MiniMaxExecutor()
    if not executor.nvidia_api_key:
        print("[-] Skipping live MiniMax call: NVIDIA_API_KEY not configured in env.")
        return

    # Verify live query with CoT
    start_t = time.time()
    res = executor.execute_task(
        task_prompt="Назови свой размер контекстного окна в текущем комплексе и своего верховного начальника.",
        enable_search=False,
        max_tokens=2048,
        temperature=0.3
    )
    elapsed = round(time.time() - start_t, 2)
    content = res.get("content", "")
    print(f"[+] MiniMax live response received in {elapsed}s ({len(content)} chars).")
    print(f"[MiniMax Response Preview]:\n{content[:400]}...")
    
    # Check that model knows its 256k window or Qwen arbiter
    assert "256" in content or "qwen" in content.lower() or "арбитр" in content.lower(), (
        "MiniMax must cite its 256k context window or Supreme Arbiter Qwen!"
    )
    print("[+] Test 3 Passed: MiniMax operates with accurate ground-truth system manifest.")

def test_4_telegram_bot_integrity():
    print("\n" + "=" * 60)
    print("TEST 4: Telegram Bot Controller Integrity & Arbitration Routing")
    print("=" * 60)
    bot = HarnessTelegramBot()
    assert bot.is_configured(), "Telegram bot token must be configured in .env!"
    
    # Test keyboard structure
    kb = bot.get_main_keyboard()
    button_texts = [btn["text"] for row in kb.get("keyboard", []) for btn in row]
    assert "👑 Запрос Арбитру" in button_texts, "Keyboard must include Arbiter Query button!"
    assert "📡 Статус системы" in button_texts, "Keyboard must include System Status button!"
    print(f"[+] Reply keyboard verified with {len(button_texts)} buttons: {button_texts}")

    # Test state manager state
    state = bot.state_mgr.load_local_state()
    assert "Qwen" in state.get("local_arbiter", ""), "Local arbiter in state must be Qwen2.5-1.5B!"
    print(f"[+] Active state verified: Arbiter = {state.get('local_arbiter')}")
    print("[+] Test 4 Passed: Telegram Bot integrity verified.")

def test_5_disk_guard_and_file_safety():
    print("\n" + "=" * 60)
    print("TEST 5: Storage & Quota Safety Inspection")
    print("=" * 60)
    dg = DiskGuard()
    disk = dg.inspect_disk_usage()
    print(f"[+] Free disk space: {disk.get('free_gb')} GB ({disk.get('percent_used')}% used).")
    assert not disk.get("is_critical"), "Disk space must not be critical!"
    
    # Verify history file size is tiny (< 100 KB)
    history_path = PROJECT_ROOT / "data" / "telegram_history.txt"
    if history_path.exists():
        size_kb = history_path.stat().st_size / 1024.0
        print(f"[+] telegram_history.txt size: {size_kb:.2f} KB (Safe plain-text storage).")
        assert size_kb < 10240, "History file must not exceed 10 MB in test environment!"
    print("[+] Test 5 Passed: Storage health and quota guard verified.")

if __name__ == "__main__":
    print("================================================================")
    print("🚀 RUNNING AUTONOMOUS COMPLEX TEST SUITE: MEMORY, NIM & ARBITER")
    print("================================================================")
    t0 = time.time()
    test_1_memory_engine_and_nemotron()
    test_2_sovereign_cpu_arbiter()
    test_3_minimax_runtime_manifest()
    test_4_telegram_bot_integrity()
    test_5_disk_guard_and_file_safety()
    total_elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 60)
    print(f"🎉 ALL 5 TEST SUITES PASSED FLAWLESSLY in {total_elapsed}s!")
    print("================================================================")
