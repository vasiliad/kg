#!/usr/bin/env python3
"""
Resilience & Honesty Stress Suite — runs OFFLINE (no network, no key needed).

It attacks the complex the way reality does: corrupted caches, missing
secrets, dead providers, broken state files, and the temptation to fabricate
a score instead of measuring. Every test either passes offline or is skipped
with an explicit reason.

Suites:
  1. Fleet tuner: candidate discovery is structural, not a brand allowlist.
  2. Fleet tuner: every probe failing must KEEP the previous fleet.
  3. Fleet tuner: full offline cycle produces verified measured scores.
  4. Duty team: dispatch on seed fleet, corrupt state, missing state.
  5. Memory: corrupted vault rebuilds from the durable log.
  6. State: corrupt active_models.json does not crash the harness.
  7. Email: no recipient / no password => skipped, never half-sent.
  8. Runtime settings: sender signature round-trip, header-safe.
  9. Bot: no password rejects everyone; correct password authorizes (isolated).
 10. Honesty: reports never print a fabricated default score.
 11. Math: the urn probe's expected fraction is actually correct.
 12. Secrets: no real passphrase embedded in the public source.
 13. CI: every workflow that pushes has contents: write.
 14. Sync: the race-safe push helper actually rebases and retries.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SOVEREIGN_FORCE_DETERMINISTIC", "1")

from harness import fleet_probes
from harness.fleet_tuner import FleetTuner, DEFAULT_POLICY
from harness.duty_team import DutyTeam, KEEPER_ID
from harness.state_manager import StateManager
from harness.memory_engine import MemoryEngine
from harness.email_sender import EmailSender
from harness import runtime_settings


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="kg-resilience-"))


def test_1_candidate_discovery_is_structural():
    print("\n" + "=" * 60)
    print("TEST 1: Tuner candidate discovery — structural, not an allowlist")
    print("=" * 60)
    tuner = FleetTuner(api_key="test-key")
    catalog = [
        {"id": "nvidia/nemotron-3.5-embed-8b"},          # embedding -> out
        {"id": "nvidia/llama-3.1-nemoguard-8b-content-safety"},  # guard -> out
        {"id": "nvidia/nemotron-4-340b-reward"},          # reward -> out
        {"id": "nvidia/nemotron-parse"},                  # parser -> out
        {"id": "nvidia/riva-translate-4b-instruct"},      # translator -> out
        {"id": "nvidia/nemotron-3-ultra-550b-a55b"},      # flagship chat -> in
        {"id": "brand-new-vendor/hypothetical-mega-900b-instruct"},  # unknown -> in
        {"id": "Qwen/Qwen2.5-1.5B-Instruct"},             # local-only -> out
    ]
    pool = tuner.build_candidate_pool(catalog, previous_fleet_ids=[])
    ids = [c["id"] for c in pool]
    assert "nvidia/nemotron-3.5-embed-8b" not in ids, "embeddings must not be probed as chat"
    assert "nvidia/nemotron-4-340b-reward" not in ids, "reward models must be excluded"
    assert "nvidia/nemotron-parse" not in ids, "parsers must be excluded"
    assert "Qwen/Qwen2.5-1.5B-Instruct" not in ids, "local-only models are not NIM candidates"
    assert "nvidia/nemotron-3-ultra-550b-a55b" in ids, "flagship chat must be probed"
    assert "brand-new-vendor/hypothetical-mega-900b-instruct" in ids, \
        "an unknown new model must compete on equal terms — no brand allowlist"
    print(f"[+] Pool of {len(ids)}: {ids}")
    print("[+] Test 1 Passed: discovery is purpose-based, new vendors can enter the fleet.")


def test_2_all_probes_failing_keeps_previous_fleet():
    print("\n" + "=" * 60)
    print("TEST 2: Total probe failure must keep the previous fleet")
    print("=" * 60)
    workdir = _tmp_dir()
    prev_state = workdir / "active_models.json"
    prev_fleet = [{"id": "minimaxai/minimax-m3.1-flash", "tier": "seed", "quality_score": None, "status": "APPROVED", "score_verified": False}]
    prev_state.write_text(json.dumps({"active_fleet": prev_fleet}), encoding="utf-8")
    os.chdir(workdir)

    tuner = FleetTuner(api_key="test-key")
    tuner.enable_offline_stub()
    # Everything dies: every request returns a network error.
    tuner._post_chat = lambda *a, **k: {"ok": False, "status": "NETWORK_ERROR:Stub", "latency_sec": 1.0, "text": ""}
    report = tuner.run(offline_catalog=["x/model-a", "x/model-b"])

    kept = report.get("fleet_kept") or []
    assert [m.get("id") for m in kept] == ["minimaxai/minimax-m3.1-flash"], \
        f"the previous fleet must survive a dead cycle, got {kept}"
    assert report.get("status") in ("NO_QUALIFIED", "CATALOG_UNAVAILABLE", "NO_API_KEY")
    print(f"[+] Cycle status: {report.get('status')}; previous fleet kept: {[m['id'] for m in kept]}")
    print("[+] Test 2 Passed: a dead cycle cannot blank the duty team.")


def test_3_offline_full_cycle_produces_verified_scores():
    print("\n" + "=" * 60)
    print("TEST 3: Full measure->score->compose cycle (deterministic stub)")
    print("=" * 60)
    workdir = _tmp_dir()
    os.chdir(workdir)
    tuner = FleetTuner(api_key="test-key")
    tuner.enable_offline_stub()
    catalog = [
        "vendor-a/alpha-70b-instruct",
        "vendor-b/beta-ultra-550b",
        "vendor-c/gamma-flash-30b",
        "nvidia/nemotron-3.5-embed-8b",  # must be filtered out
    ]
    report = tuner.run(offline_catalog=catalog)
    fleet = report.get("fleet") or []
    assert fleet, "the stub cycle must produce a fleet"
    for entry in fleet:
        assert entry["score_source"] == "live_probe", "measured entries must be tagged live_probe"
        assert entry["score_verified"] is True, "measured entries must be verified"
        assert entry["measured"]["latency_sec"] > 0
    roles = {r for e in fleet for r in e["roles"]}
    assert "duty_primary" in roles, "a primary must be appointed"
    assert len({e["id"] for e in fleet}) == len(fleet), "no duplicate fleet members"
    n_embed = [e for e in fleet if "embed" in e["id"]]
    assert not n_embed, "embeddings must never serve as duty chat models"
    print(f"[+] Fleet: {[(e['id'], e['roles'], e['quality_score']) for e in fleet]}")
    print("[+] Test 3 Passed: verified measured scores, role spread, no dupes.")


def test_4_duty_team_degrades_gracefully():
    print("\n" + "=" * 60)
    print("TEST 4: Duty team dispatch on seed fleet / corrupt / missing state")
    print("=" * 60)
    # (a) real seed fleet (no roles yet)
    team = DutyTeam()
    plan = team.dispatch("напиши python функцию для проверки API")
    assert len(plan["calls"]) >= 1, "even a seed fleet must yield calls"
    assert plan["rationale"], "the plan must carry a plain-Russian rationale"
    # (b) corrupt state
    workdir = _tmp_dir()
    (workdir / "active_models.json").write_text("{broken json!!", encoding="utf-8")
    team2 = DutyTeam(state_path=workdir / "active_models.json")
    plan2 = team2.dispatch("что угодно")
    assert plan2["calls"] == [], "corrupt state must yield an empty plan, not a crash"
    roster_txt = team2.format_roster()
    assert "не измерены" in roster_txt, "the roster must say honestly that nobody is measured"
    # (c) missing state
    team3 = DutyTeam(state_path=workdir / "nope.json")
    assert team3.dispatch("задача")["calls"] == []
    # (d) keeper is never listed as a network duty model
    for m in team.load_roster()["duty"]:
        assert m["id"] != KEEPER_ID, "the local keeper is the chief, not a duty model"
    print(f"[+] Seed-fleet plan: {[c['id'] for c in plan['calls']]}")
    print("[+] Test 4 Passed: corrupt/missing state degrade without exceptions.")


def test_5_memory_vault_corruption_rebuilds():
    print("\n" + "=" * 60)
    print("TEST 5: Corrupted semantic vault rebuilds from the durable log")
    print("=" * 60)
    workdir = _tmp_dir()
    engine = MemoryEngine(
        history_file=workdir / "telegram_history.txt",
        vault_file=workdir / ".memory_vault.json",
    )
    engine.append_message("user", "Точное окно контекста MiniMax M3.1 Flash составляет 256,000 токенов.")
    # Corrupt the derived cache — the durable log is the source of truth.
    engine.vault_file.write_text("{{{not json", encoding="utf-8")
    matches = engine.search_memories("окно контекста", top_k=2)
    assert isinstance(matches, list), "search must never raise on a corrupt vault"
    # append must also survive and rewrite the vault cleanly
    engine.append_message("arbiter", "Окно контекста подтверждено: 256,000 токенов.")
    with open(engine.vault_file, "r", encoding="utf-8") as f:
        json.load(f)
    print("[+] Test 5 Passed: corrupted vault is rebuilt; log remains the source of truth.")


def test_6_corrupt_state_file_does_not_crash():
    print("\n" + "=" * 60)
    print("TEST 6: Corrupt active_models.json — harness keeps running")
    print("=" * 60)
    workdir = _tmp_dir()
    (workdir / "active_models.json").write_text("definitely not json", encoding="utf-8")
    mgr = StateManager(state_file=str(workdir / "active_models.json"),
                       history_file=str(workdir / "nim_history.log"))
    state = mgr.load_local_state()
    assert isinstance(state, dict), "load must return a dict default"
    mgr.append_history_log("ПРОВЕРКА УСТОЙЧИВОСТИ: запись после битого state")
    log_text = (workdir / "nim_history.log").read_text(encoding="utf-8")
    assert "ПРОВЕРКА УСТОЙЧИВОСТИ" in log_text
    print("[+] Test 6 Passed: corrupt state is contained, logging continues.")


def test_7_email_never_half_sends():
    print("\n" + "=" * 60)
    print("TEST 7: Email skips cleanly without recipient / credentials")
    print("=" * 60)
    sender_no_creds = EmailSender(sender_email="ivanvasiliadi@gmail.com", app_password=None)
    res = sender_no_creds.send_email("someone@example.com", "Тест", "<p>ok</p>")
    assert res["status"] == "SKIPPED_NOT_CONFIGURED", res
    sender_full = EmailSender(sender_email="ivanvasiliadi@gmail.com", app_password="x" * 16)
    res2 = sender_full.send_email("", "Тест", "<p>ok</p>")
    assert res2["status"] == "SKIPPED_NO_RECIPIENT", res2
    res3 = sender_full.send_email("not-an-email", "Тест", "<p>ok</p>")
    assert res3["status"] == "SKIPPED_NO_RECIPIENT", res3
    print("[+] Test 7 Passed: email paths fail closed with explicit statuses.")


def test_8_sender_signature_roundtrip():
    print("\n" + "=" * 60)
    print("TEST 8: Sender signature is runtime-settable and header-safe")
    print("=" * 60)
    workdir = _tmp_dir()
    original = runtime_settings.SETTINGS_FILE
    runtime_settings.SETTINGS_FILE = workdir / "settings.json"
    try:
        applied = runtime_settings.set_sender_name("Комплекс <KG> тест")
        assert "<" not in applied and ">" not in applied, "header-breaking chars must be stripped"
        assert runtime_settings.get_sender_name() == applied, "the value must persist"
        try:
            runtime_settings.set_sender_name("   ")
            assert False, "empty signature must be rejected"
        except ValueError:
            pass
    finally:
        runtime_settings.SETTINGS_FILE = original
    print(f"[+] Applied signature: {applied!r}")
    print("[+] Test 8 Passed: signature change survives a restart file round-trip.")


def test_9_bot_access_guard():
    print("\n" + "=" * 60)
    print("TEST 9: Bot access guard — no password rejects everyone")
    print("=" * 60)
    from harness.telegram_bot import HarnessTelegramBot
    bot = HarnessTelegramBot()
    workdir = _tmp_dir()
    bot.sessions_file = workdir / "sessions.json"  # isolate the session store
    sent = []
    bot.send_message = lambda chat_id, text, **kw: sent.append((chat_id, text)) or True

    bot.bot_password = ""  # misconfigured secret
    bot.process_message({"chat": {"id": 424242}, "from": {"id": 1}, "text": "пароль?", "message_id": None})
    assert bot.authorized_chats == set() or 424242 not in bot.authorized_chats, \
        "an empty passphrase must authorize nobody"
    assert any("ДОСТУП ЗАКРЫТ" in t for _, t in sent), "the lock message must be shown"

    # correct passphrase authorizes exactly one chat
    bot.bot_password = "unit-test-pass-1"
    bot.process_message({"chat": {"id": 424242}, "from": {"id": 1}, "text": "unit-test-pass-1", "message_id": None})
    assert 424242 in bot.authorized_chats, "the correct passphrase must authorize"
    saved = json.loads((workdir / "sessions.json").read_text(encoding="utf-8"))
    assert 424242 in saved.get("authorized_chats", []), "sessions must persist to the isolated file"
    print("[+] Test 9 Passed: password gate fails closed and authorizes precisely.")


def test_10_reports_never_fabricate_scores():
    print("\n" + "=" * 60)
    print("TEST 10: The daily report never prints an invented score")
    print("=" * 60)
    from harness.daily_dispatcher import build_health_report_email
    workdir = _tmp_dir()
    state = {
        "active_fleet": [
            {"id": "a/measured-70b", "tier": "primary", "quality_score": 87.4, "score_verified": True, "roles": ["duty_primary"]},
            {"id": "b/seeded-70b", "tier": "seed", "quality_score": 99.2, "score_verified": False},
            {"id": "c/unranked-70b", "tier": "new", "quality_score": None, "score_verified": False},
        ],
        "local_arbiter": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    (workdir / "active_models.json").write_text(json.dumps(state), encoding="utf-8")
    os.chdir(workdir)
    mail = build_health_report_email()
    html_text = mail["html"]
    assert "(живой замер)" in html_text, "measured scores must be labelled"
    assert "(статический сид, не проверено)" in html_text, "seed scores must be marked unverified"
    assert "UNRANKED" in html_text, "missing scores must be UNRANKED, not invented"
    assert "Оценка: 95.0" not in html_text, "the fabricated default must be gone"
    print("[+] Test 10 Passed: score provenance is visible in the report.")


def test_11_urn_math_is_actually_correct():
    print("\n" + "=" * 60)
    print("TEST 11: The probe's expected fraction is verified by computation")
    print("=" * 60)
    from itertools import combinations
    balls = ["R"] * 3 + ["G"] * 4 + ["B"] * 5
    pairs = list(combinations(range(12), 2))
    diff = sum(1 for i, j in pairs if balls[i] != balls[j])
    total = len(pairs)
    from math import gcd
    g = gcd(diff, total)
    assert f"{diff // g}/{total // g}" == fleet_probes.REASONING_ANSWER, \
        f"expected fraction must equal {diff}/{total} reduced"
    score, verdict = fleet_probes.check_reasoning("Вероятность равна 19/22")
    assert score == 0.0, "the old WRONG seed answer (19/22) must score zero"
    print(f"[+] Enumerated: {diff}/{total} -> {fleet_probes.REASONING_ANSWER}; wrong answer scores {score}")
    print("[+] Test 11 Passed: the probe rewards only the mathematically correct answer.")


def test_12_no_secrets_in_public_source():
    print("\n" + "=" * 60)
    print("TEST 12: No real passphrase or key embedded in the public source")
    print("=" * 60)
    # The check must not contain the secret itself, so it is assembled here
    # from parts — otherwise this very file would be the one leak it hunts.
    forbidden = ["".join(["arabyan", "3G"]), "".join(["sk", "-tinyfish-"])]
    extensions = {".py", ".sh", ".yml", ".yaml", ".md", ".json", ".txt"}
    scan_roots = ["harness", "scripts", "tests", ".github", "config.yaml"]
    hits = []
    for root in scan_roots:
        p = PROJECT_ROOT / root
        files = [p] if p.is_file() else list(p.rglob("*")) if p.exists() else []
        for f in files:
            if f.is_file() and f.suffix.lower() in extensions:
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for secret in forbidden:
                    if secret in content:
                        hits.append(f"{f} contains {secret!r}")
    assert not hits, f"REAL SECRETS IN PUBLIC SOURCE: {hits}"
    print("[+] Test 12 Passed: scanned harness/scripts/tests/.github — no embedded secrets.")


def test_13_workflows_that_push_have_contents_write():
    print("\n" + "=" * 60)
    print("TEST 13: Every pushing workflow has contents: write")
    print("=" * 60)
    import yaml
    wf_dir = PROJECT_ROOT / ".github" / "workflows"
    problems = []
    for wf in sorted(wf_dir.glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        text = json.dumps(doc)
        pushes = "sync_state.sh" in text or "git push" in text or "git commit" in text
        perms = (doc.get("permissions") or {})
        contents = perms.get("contents") if isinstance(perms, dict) else None
        if pushes and contents != "write":
            problems.append(f"{wf.name}: contents={contents!r} but the workflow commits/pushes")
    assert not problems, f"PERMISSION HOLES: {problems}"
    print("[+] Test 13 Passed: all pushing workflows hold contents: write.")


def test_14b_size_floor_keeps_only_big_models():
    print("\n" + "=" * 60)
    print("TEST 14b: Hard parameter floor — 120B-and-under never enter the pool")
    print("=" * 60)
    from harness.fleet_tuner import parsed_params_b
    # id parsing: total params, not the active MoE part
    assert parsed_params_b("mistralai/mistral-large-4-1t-a52b") == 1000.0
    assert parsed_params_b("nvidia/nemotron-3-ultra-550b-a55b") == 550.0
    assert parsed_params_b("minimaxai/minimax-m3.1-flash") is None  # no size token
    tuner = FleetTuner(api_key="test-key")
    # Hermetic resolver: no network in unit tests (the real one may hit the
    # docs search, which would make this test flaky).
    from harness.model_size import SizeResolver as _SR
    tuner.size_resolver = _SR(overrides={}, searcher=lambda q: [],
                              cache_path=_tmp_dir() / "sizes.json")
    catalog = [
        {"id": "nvidia/nemotron-3-ultra-550b-a55b"},   # 550B -> in
        {"id": "mistralai/mistral-large-4-1t-a52b"},   # 1T   -> in
        {"id": "nvidia/nemotron-3.5-super-vl-120b-a12b"},  # 120B -> OUT (operator: too small)
        {"id": "google/gemma-2b"},                     # 2B   -> OUT
        # no size token in the id, but the catalogue metadata resolves it:
        {"id": "minimaxai/minimax-m3.1-flash", "description": "A 230 billion parameter MoE model."},
        {"id": "vendor/undocumented-instruct"},        # unresolvable -> OUT (fail-closed)
    ]
    pool_ids = [c["id"] for c in tuner.build_candidate_pool(catalog, previous_fleet_ids=[])]
    assert "nvidia/nemotron-3-ultra-550b-a55b" in pool_ids
    assert "mistralai/mistral-large-4-1t-a52b" in pool_ids
    assert "nvidia/nemotron-3.5-super-vl-120b-a12b" not in pool_ids, "120B must be below the floor"
    assert "google/gemma-2b" not in pool_ids, "2B must be below the floor"
    assert "minimaxai/minimax-m3.1-flash" in pool_ids, "size resolved from metadata -> eligible"
    assert "vendor/undocumented-instruct" not in pool_ids, "unresolvable size -> excluded"
    print(f"[+] Pool under a 150B floor: {pool_ids}")
    print("[+] Test 14b Passed: only big models (or resolvable ones) enter the pool.")


def test_14c_unknown_size_is_resolved_or_excluded():
    print("\n" + "=" * 60)
    print("TEST 14c: No size in the id -> metadata/docs resolve it, else OUT")
    print("=" * 60)
    from harness.model_size import (
        parse_size_from_text, parse_size_from_id, SizeResolver,
    )
    # text parsing: prose and "NB parameters" forms; bare "4B tokens" must NOT match
    assert parse_size_from_text("a 550 billion parameter flagship") == 550.0
    assert parse_size_from_text("1.2 trillion parameters") == 1200.0
    assert parse_size_from_text("70B params, 8B active") == 70.0
    assert parse_size_from_text("supports 4B token context") is None
    assert parse_size_from_id("mistralai/mistral-large-4-1t-a52b") == 1000.0
    # resolver precedence: id -> catalog metadata -> cache -> docs; unresolved = None
    overrides = {}
    cache = Path(tempfile.mkdtemp(prefix="kg-sizes-")) / "model_sizes.json"
    r1 = SizeResolver(overrides=overrides, cache_path=cache)
    assert r1.resolve("x/model-550b-instruct") == 550.0 and r1.sources["x/model-550b-instruct"] == "id"
    entry = {"id": "vendor/mystery-instruct", "description": "A dense 230B parameter model."}
    r2 = SizeResolver(overrides=overrides, cache_path=cache)
    assert r2.resolve("vendor/mystery-instruct", entry) == 230.0
    r3 = SizeResolver(overrides=None, cache_path=cache)  # a fresh cycle loads the persisted cache
    assert r3.resolve("vendor/mystery-instruct", {"id": "vendor/mystery-instruct"}) == 230.0
    # docs search resolves once and caches
    def fake_search(query):
        return ["Mystery-Instruct is a 310 billion parameter dense model"]
    r4 = SizeResolver(overrides=overrides, searcher=fake_search, cache_path=cache)
    assert r4.resolve("vendor/mystery2") == 310.0
    # fail-closed: nothing anywhere -> None
    r5 = SizeResolver(overrides=overrides, searcher=lambda q: [], cache_path=cache)
    assert r5.resolve("vendor/completely-unknown") is None
    # the pool must reject an unresolvable model even in guaranteed slots
    tuner = FleetTuner(api_key="test-key")
    tuner.size_resolver = r5  # everything unresolved
    pool_ids = [c["id"] for c in tuner.build_candidate_pool(
        [{"id": "vendor/completely-unknown"}], previous_fleet_ids=[], must_probe=["vendor/completely-unknown"],
    )]
    assert pool_ids == [], "an unresolvable model must NOT enter the pool, even as a newcomer"
    print("[+] id/metadata/docs/cache precedence verified; unresolvable => excluded")
    print("[+] Test 14c Passed: the pool cannot accumulate garbage.")


def test_14d_tinyfish_pool_rotates_and_fails_over():
    print("\n" + "=" * 60)
    print("TEST 14d: TinyFish key pool — rotation, failover, no keys in source")
    print("=" * 60)
    from harness.tinyfish import TinyFishSearchPool, SEARCH_ENDPOINT
    pool = TinyFishSearchPool(keys=["".join(["sk-tinyfish", "-", "A"]), "B", "C"])
    assert len(pool) == 3 and pool.is_configured
    # round-robin order
    seq = [pool._next_key(), pool._next_key(), pool._next_key(), pool._next_key()]
    assert seq[0] != seq[1] != seq[2] and seq[3] == seq[0], f"rotation broken: {seq}"
    # a rejected key is skipped on subsequent calls
    pool.mark_bad(seq[0])
    nxt = pool._next_key()
    assert nxt != seq[0], "a marked-bad key must not be handed out"
    # empty env -> unusable but never raises
    empty = TinyFishSearchPool(keys=[])
    assert empty.search("anything") == [] and not empty.is_configured
    # response parsing (hermetic): standard shape
    class FakeRes:
        status_code = 200
        def json(self):
            return {"results": [
                {"title": "T1", "url": "https://a", "snippet": "s1"},
                {"title": "T2", "url": "https://b", "snippet": "s2"},
            ]}
    parsed = TinyFishSearchPool._parse(FakeRes(), 5)
    assert parsed[0] == {"title": "T1", "url": "https://a", "snippet": "s1"}
    # a real network call is NOT attempted here — the search() path with a
    # live key is exercised in the cloud; this suite stays offline.
    assert SEARCH_ENDPOINT.startswith("https://")
    print(f"[+] Pool of {len(pool)} keys rotates; bad keys are skipped; parsing is standard-shaped")
    print("[+] Test 14d Passed: search survives a dead key, and no key can leak into source.")


def test_14_sync_helper_rebases_and_retries():
    print("\n" + "=" * 60)
    print("TEST 14: Race-safe push helper exists and actually retries")
    print("=" * 60)
    script = (PROJECT_ROOT / "scripts" / "sync_state.sh").read_text(encoding="utf-8")
    assert "git rebase" in script, "the helper must rebase onto origin/main"
    assert "MAX_ATTEMPTS" in script, "the helper must retry up to a bounded attempt count"
    assert "checkout --theirs" in script, "on conflict the newest run's state files win"
    print("[+] Test 14 Passed: concurrent-job push race is handled by design.")


if __name__ == "__main__":
    print("================================================================")
    print("🛡️  RESILIENCE & HONESTY STRESS SUITE (offline)")
    print("================================================================")
    t0 = time.time()
    tests = [
        test_1_candidate_discovery_is_structural,
        test_2_all_probes_failing_keeps_previous_fleet,
        test_3_offline_full_cycle_produces_verified_scores,
        test_4_duty_team_degrades_gracefully,
        test_5_memory_vault_corruption_rebuilds,
        test_6_corrupt_state_file_does_not_crash,
        test_7_email_never_half_sends,
        test_8_sender_signature_roundtrip,
        test_9_bot_access_guard,
        test_10_reports_never_fabricate_scores,
        test_11_urn_math_is_actually_correct,
        test_12_no_secrets_in_public_source,
        test_13_workflows_that_push_have_contents_write,
        test_14_sync_helper_rebases_and_retries,
        test_14b_size_floor_keeps_only_big_models,
        test_14c_unknown_size_is_resolved_or_excluded,
        test_14d_tinyfish_pool_rotates_and_fails_over,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"[-] {t.__name__} FAILED: {e}")
    total = round(time.time() - t0, 2)
    if failed:
        print("\n" + "=" * 60)
        print(f"❌ {len(failed)} of {len(tests)} suites FAILED in {total}s:")
        for name, err in failed:
            print(f"    - {name}: {err}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print(f"🎉 ALL {len(tests)} RESILIENCE SUITES PASSED in {total}s!")
    print("================================================================")
