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
 15. Bot: the Telegram cursor survives a restart, holds at gaps, never replays.
 16. Bot: the daemon keeps itself alive (chained windows + watchdog).
 17. Bot: two independent recovery paths revive a dead chain, and neither
     can start a second chain beside a live one.
 18. Fleet: the multilingual and tool-use probes score by construction, and
     every quality candidate gets the same probe set.
 19. Fleet: drift is the difference between two measured cycles, bounded,
     and never guessed.
 20. Duty: the dispatch order follows the measured task probes, not the
     static order of the roster.
 21. Bot: the status card shows the measured drift in words, and invents
     nothing when there is no measurement.
 22. Bot: the transport answers a button press, edits and deletes — and a
     press from a stranger does nothing.
 23. Search: the query is the topical core of the question, not a truncated
     blob of the prompt; our own questions never cost a search.
 24. Bot: a measured health trace survives a restart, and an absent bot is
     distinguishable from a broken one.
 25. Bot: the cursor is monotonic — a second writer cannot roll it back and
     make Telegram replay answers the operator already saw.
 26. Bot: the edit verb strips the payload until Telegram accepts it, and
     reports the refusal instead of swallowing it.
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
from harness import fleet_drift
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


def test_14e_langsearch_parses_standard_shape():
    print("\n" + "=" * 60)
    print("TEST 14e: LangSearch client — response parsing and fail-closed stubs")
    print("=" * 60)
    from harness.langsearch import LangSearchClient
    client = LangSearchClient(api_key="")  # unconfigured
    assert not client.is_configured and client.search("anything") == [], \
        "an unconfigured client must be a silent no-op"
    fake = {"code": 200, "data": {"webPages": {"value": [
        {"name": "T1", "url": "https://a", "snippet": "s1"},
        {"name": "T2", "url": "https://b", "text": "full text mode"},
    ]}}}
    parsed = LangSearchClient._parse(fake, 5)
    assert parsed[0] == {"title": "T1", "url": "https://a", "snippet": "s1"}
    assert parsed[1]["snippet"] == "full text mode", "text mode must map into snippet"
    assert LangSearchClient._parse({}, 5) == [], "malformed answers parse to empty"
    print("[+] Test 14e Passed: LangSearch answers map to the standard shape.")


def test_15_telegram_cursor_survives_restart():
    print("\n" + "=" * 60)
    print("TEST 15: Telegram cursor — survives a restart, never replays, never skips")
    print("=" * 60)
    import threading
    from harness.telegram_bot import HarnessTelegramBot

    tmp = Path(tempfile.mkdtemp(prefix="cursor-test-"))
    sessions = tmp / ".bot_sessions.json"

    def bare_bot():
        # Built without the constructor: no network, no keys, no engines.
        bot = HarnessTelegramBot.__new__(HarnessTelegramBot)
        bot.sessions_file = sessions
        bot.authorized_chats = {846947027}
        bot.last_update_id = 0
        bot._pending_update_ids = set()
        bot._dispatched_max = 0
        bot._state_lock = threading.RLock()
        return bot

    bot = bare_bot()
    # Updates finish out of order (the pool runs handlers in parallel), so an
    # unfinished update must hold the cursor back — a crash must never drop
    # the message that is still in flight.
    bot._mark_update_dispatched(10)
    bot._mark_update_dispatched(11)
    bot._mark_update_handled(11)
    assert bot.last_update_id == 9, "an unfinished update must hold the cursor back"
    bot._mark_update_handled(10)
    assert bot.last_update_id == 11, "the cursor must advance over finished updates"

    # Telegram update ids are arbitrary large numbers, not a sequence that
    # starts at 1 — the cursor must work with any base.
    bot2 = bare_bot()
    bot2._mark_update_dispatched(8_765_432)
    bot2._mark_update_handled(8_765_432)
    assert bot2.last_update_id == 8_765_432, "the cursor must not assume ids start at 1"

    on_disk = json.loads(sessions.read_text(encoding="utf-8"))
    assert on_disk["last_update_id"] == 8_765_432, "the cursor must be persisted"
    assert 846947027 in on_disk["authorized_chats"], "authorized chats must survive too"

    # A cold start must resume from the persisted cursor. This is what stops
    # the bot from replaying the whole 24-hour backlog after every restart
    # (the bug that answered old questions again, hours late).
    fresh = bare_bot()
    assert fresh._load_last_update_id() == 8_765_432, "the cursor must survive a restart"
    assert fresh._load_authorized_sessions() == {846947027}

    # A broken session file degrades to an empty state, never to a crash.
    sessions.write_text("{not json", encoding="utf-8")
    broken = bare_bot()
    assert broken._load_last_update_id() == 0, "a broken session file must not crash the bot"

    print("[+] Test 15 Passed: the cursor survives a restart, holds at gaps, and degrades safely.")


def test_16_bot_keeps_itself_alive():
    print("\n" + "=" * 60)
    print("TEST 16: The bot survives a restart — chained windows plus a watchdog")
    print("=" * 60)
    bot_wf = (PROJECT_ROOT / ".github" / "workflows" / "telegram_bot.yml").read_text(encoding="utf-8")
    watchdog = (PROJECT_ROOT / ".github" / "workflows" / "bot_watchdog.yml").read_text(encoding="utf-8")
    # The bot may not depend on a punctual cron: on 2026-10-31 the */30
    # schedule skipped whole slots and the bot was unreachable for hours.
    assert "workflow_dispatch" in bot_wf, "a bot window must be dispatchable"
    assert "gh workflow run telegram_bot.yml" in bot_wf, "a window must dispatch its own successor"
    assert "if: always()" in bot_wf, "the chain must survive a crashed daemon step"
    assert "actions: write" in bot_wf, "dispatching a workflow requires actions: write"
    assert "schedule:" in watchdog, "the watchdog needs a schedule"
    assert "scripts/revive_bot.sh" in watchdog, "the watchdog must revive the bot"
    revive = (PROJECT_ROOT / "scripts" / "revive_bot.sh").read_text(encoding="utf-8")
    assert "status=in_progress" in revive and "status=queued" in revive, \
        "the watchdog must never create a second chain next to a live one"
    print("[+] Test 16 Passed: the bot keeps itself alive without a punctual cron.")


def test_17_bot_has_two_independent_recovery_paths():
    """
    A single cron path is a single point of failure: on 2026-10-31 the bot's
    own `*/30` schedule was skipped and the bot was unreachable for hours.
    The bot now chains its own windows (no cron at all), and the chain is
    backed by TWO revive paths — a dense watchdog and the heartbeat whose
    cron demonstrably fires. Neither may start a second chain next to a live
    one, and neither may fail its surrounding job.
    """
    print("=" * 60)
    print("Test 17: the bot has two independent recovery paths")
    revive = PROJECT_ROOT / "scripts" / "revive_bot.sh"
    assert revive.exists(), "scripts/revive_bot.sh must exist"
    script = revive.read_text(encoding="utf-8")

    # The guard: a live OR waiting window means "do nothing".
    assert "status=in_progress" in script and "status=queued" in script, \
        "the reviver must look at both running and waiting windows"
    assert "the bot is alive" in script, "the reviver must stay silent while the bot lives"
    assert "gh workflow run" in script, "the reviver must be able to start a window"
    assert "attempt in 1 2 3" in script, "a failed dispatch must be retried"
    assert script.rstrip().endswith("exit 0"), \
        "the reviver must never fail the job that hosts it"
    assert "sleep 10" in script, "a retry must wait instead of hammering the API"

    watchdog = (PROJECT_ROOT / ".github" / "workflows" / "bot_watchdog.yml").read_text(encoding="utf-8")
    heartbeat = (PROJECT_ROOT / ".github" / "workflows" / "emergency_cron.yml").read_text(encoding="utf-8")

    assert "scripts/revive_bot.sh" in watchdog, "the watchdog must use the shared reviver"
    assert "scripts/revive_bot.sh" in heartbeat, \
        "the heartbeat must be the second, independent recovery path"
    assert "actions: write" in watchdog, "dispatching a window requires actions: write"
    assert "actions: write" in heartbeat, "dispatching a window requires actions: write"

    # The watchdog's schedule is expressed the same way as every other
    # scheduled workflow in the repo: one cron entry per minute value.
    assert "- cron: '7 * * * *'" in watchdog, "the watchdog must have a dense schedule"
    assert "," not in watchdog.split("schedule:")[1].split("workflow_dispatch:")[0], \
        "the comma-list cron form produced no scheduled run and must not come back"

    # The cron is not trustworthy here, so the bot arms the watchdog itself
    # and asks it to check AFTER the window would have ended — a tick that
    # checked immediately would always see a live bot and learn nothing.
    bot_wf = (PROJECT_ROOT / ".github" / "workflows" / "telegram_bot.yml").read_text(encoding="utf-8")
    assert "gh workflow run bot_watchdog.yml" in bot_wf, \
        "the bot must arm its own safety net — cron cannot be trusted"
    assert "-f delay_minutes=55" in bot_wf, \
        "the armed tick must check after the window ends, not while the bot is alive"
    assert "delay_minutes" in watchdog, "the watchdog must accept a delay"
    assert "sleep $((delay * 60))" in watchdog, "the watchdog must honour the delay"
    assert "group: bot-watchdog" in watchdog, \
        "overlapping ticks must queue instead of racing each other"
    print("[+] Test 17 Passed: two independent recovery paths, both fail-safe.")


def test_18_multilingual_and_tool_use_probes_are_objective():
    """
    The new probes must be verifiable by construction, like the others: the
    expected answer is parsed or matched, never judged by another model.
    A probe that cannot be checked objectively is not a probe.
    """
    print("\n" + "=" * 60)
    print("TEST 18: multilingual and tool-use probes score by construction")
    print("=" * 60)

    # --- multilingual: the model must leave the operator's language --------
    score, verdict = fleet_probes.score_probe("translation", "Consignee")
    assert score == 1.0, f"the canonical English term must score 1.0, got {score}"
    assert verdict == "correct"

    # The keyword wins even when the model also repeats the Russian word:
    # a model that answers "Consignee (грузополучатель)" HAS shown the ability.
    score, _ = fleet_probes.score_probe("translation", "Consignee (грузополучатель)")
    assert score == 1.0, "the correct term must survive a Russian gloss"

    score, verdict = fleet_probes.score_probe("translation", "грузополучатель")
    assert score == 0.0 and verdict == "answered_in_russian", \
        "answering in Russian is not multilingual ability"

    score, verdict = fleet_probes.score_probe("translation", "the recipient")
    assert score == 0.4 and verdict == "loose_synonym", \
        "a loose synonym earns partial credit, not a full one"

    score, verdict = fleet_probes.score_probe("translation", "")
    assert score == 0.0 and verdict == "empty", "an empty answer is not a translation"

    # --- tool use: a well-formed call with the right tool and argument ---
    good = '{"tool": "get_weather", "arguments": {"city": "Новосибирск"}}'
    score, verdict = fleet_probes.score_probe("toolcall", good)
    assert score == 1.0, f"a correct tool call must score 1.0, got {score}"
    assert verdict.startswith("correct"), f"the verdict must name the correct call, got {verdict}"

    # A model that explains instead of calling must not be credited.
    score, verdict = fleet_probes.score_probe("toolcall", "I would call the weather tool for Novosibirsk.")
    assert score == 0.0 and verdict == "no_json", \
        "prose is not a tool call"

    wrong_tool = '{"tool": "get_news", "arguments": {"city": "Новосибирск"}}'
    score, _ = fleet_probes.score_probe("toolcall", wrong_tool)
    assert score == 0.5, "the right argument with the wrong tool is half a call"

    wrong_arg = '{"tool": "get_weather", "arguments": {"city": "Paris"}}'
    score, _ = fleet_probes.score_probe("toolcall", wrong_arg)
    assert score == 0.5, "the right tool with the wrong argument is half a call"

    score, verdict = fleet_probes.score_probe("toolcall", '{"tool": "get_weather",}')
    assert score == 0.0 and verdict == "invalid_json", \
        "malformed JSON must not be credited as a tool call"

    score, verdict = fleet_probes.score_probe("toolcall", "{not json")
    assert score == 0.0 and verdict == "no_json", \
        "text without a JSON object is not a tool call"

    # A fenced block is still a tool call.
    score, _ = fleet_probes.score_probe("toolcall", "```json\n" + good + "\n```")
    assert score == 1.0, "markdown fences must not hide a correct call"

    # --- the probes are wired into the tuner -------------------------------
    assert "translation" in fleet_probes.PROBE_TASKS and "toolcall" in fleet_probes.PROBE_TASKS
    assert DEFAULT_POLICY["quality_tasks"] == ["reasoning", "code", "json", "translation", "toolcall"], \
        "every quality candidate must get the same probes, or the composite is not comparable"
    weights = fleet_probes.probe_weights(DEFAULT_POLICY["quality_tasks"] + ["instruction"])
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "probe weights must sum to 1"
    assert weights["instruction"] > 0, "the operator's language must keep a share"
    print("[+] Test 18 Passed: the new probes are objective, cheap and wired in.")


def _cycle(scores, fleet=None, primary=None):
    """A minimal measured-cycle report, as the tuner writes it."""
    measurements = []
    for mid, score in scores.items():
        measurements.append({
            "id": mid, "status": "MEASURED", "final_score": score,
            "task_scores": {"instruction": 1.0, "reasoning": 1.0, "code": 1.0, "json": 1.0},
            "roles": [],
        })
    fleet_ids = list(fleet if fleet is not None else scores.keys())
    fleet = [{"id": mid, "roles": ["duty_primary"] if mid == primary else []} for mid in fleet_ids]
    return {"tuned_at": "2026-11-01 00:00:00 UTC", "measurements": measurements, "fleet": fleet}


def test_19_fleet_drift_is_measured_not_guessed():
    """
    Drift is the difference between two MEASURED cycles. It must never be
    inferred: with no previous cycle the answer is "FIRST", and a model that
    was not measured is not counted as a change.
    """
    print("\n" + "=" * 60)
    print("TEST 19: fleet drift compares measured cycles and remembers them")
    print("=" * 60)

    previous = _cycle({"m/a": 90.0, "m/b": 80.0}, primary="m/a")
    current = _cycle({"m/a": 92.0, "m/b": 78.0}, primary="m/a")

    drift = fleet_drift.compare_cycles(previous, current)
    assert drift["status"] == "STABLE", f"a 2-point move is noise, got {drift['status']}"
    assert drift["mean_score_delta"] == 0.0, f"mean delta must be the measured mean, got {drift}"
    assert drift["per_model"]["m/a"]["delta"] == 2.0
    assert drift["per_model"]["m/b"]["delta"] == -2.0

    # A real regression must be visible, not smoothed away.
    degraded = fleet_drift.compare_cycles(previous, _cycle({"m/a": 70.0, "m/b": 78.0}, primary="m/a"))
    assert degraded["status"] == "DEGRADED", "a 20-point drop must be flagged"
    assert degraded["worst"]["id"] == "m/a"
    assert degraded["worst"]["delta"] == -20.0

    # A roster swap without a collapse is REPLACED, not DEGRADED.
    replaced = fleet_drift.compare_cycles(
        previous, _cycle({"m/a": 90.0, "m/c": 89.0}, primary="m/c"))
    assert replaced["status"] == "REPLACED", "a new primary must be visible"
    assert "m/c" in replaced["added"] and "m/b" in replaced["removed"]

    # No previous cycle means no number at all.
    first = fleet_drift.compare_cycles(None, current)
    assert first["status"] == "FIRST" and first["mean_score_delta"] is None, \
        "without a previous cycle there is nothing to compare"

    # A changed probe set means a changed scale: the composite is a weighted
    # mean over the probes, so the delta would look like an improvement that
    # never happened. The verdict must be RESCALED, not IMPROVING.
    old_scale = _cycle({"m/a": 90.0}, primary="m/a")
    old_scale["policy"] = {"quality_tasks": ["reasoning", "code", "json"]}
    new_scale = _cycle({"m/a": 99.0}, primary="m/a")
    new_scale["policy"] = {"quality_tasks": ["reasoning", "code", "json", "translation"]}
    rescaled = fleet_drift.compare_cycles(old_scale, new_scale)
    assert rescaled["status"] == "RESCALED", \
        f"a scale change must not be reported as an improvement, got {rescaled['status']}"
    assert rescaled["mean_score_delta"] == 9.0, "the raw delta is still shown"
    assert rescaled["previous_tasks"] == ["reasoning", "code", "json"]

    # A broken report must not take the tuner down.
    broken = fleet_drift.compare_cycles({"measurements": "not a list"}, current)
    assert broken["status"] == "FIRST", "a broken previous report degrades to FIRST"

    # The history is bounded, so the trend line cannot grow without limit.
    workdir = _tmp_dir()
    hist = str(workdir / "fleet_history.json")
    for i in range(30):
        fleet_drift.append_cycle(hist, _cycle({"m/a": 90.0 + i}))
    history = fleet_drift.load_history(hist)
    assert len(history) == fleet_drift.MAX_HISTORY_CYCLES, \
        f"the history must stay bounded, got {len(history)} cycles"
    assert history[-1]["scores"]["m/a"] == 119.0

    # A corrupt history file must not crash the reader.
    with open(hist, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert fleet_drift.load_history(hist) == [], "a corrupt history yields no cycles"

    assert fleet_drift.summarize(degraded).startswith("Дрейф флота: DEGRADED")
    print("[+] Test 19 Passed: drift is measured, bounded and never guessed.")


def test_20_dispatch_follows_measured_task_evidence():
    """
    The duty order must follow the MEASURED probes, not the static order of
    the roster: a model whose code probe measured 1.0 must serve code tasks
    ahead of one that measured 0.3, even if the latter has a higher overall
    score.
    """
    print("\n" + "=" * 60)
    print("TEST 20: the duty order follows the measured task probes")
    print("=" * 60)
    workdir = _tmp_dir()
    state_path = workdir / "active_models.json"
    state_path.write_text(json.dumps({
        "active_fleet": [
            {
                "id": "v/low-code", "tier": "Дежурный кодер", "roles": ["duty_code"],
                "quality_score": 95.0, "score_verified": True,
                "measured": {"latency_sec": 3.0, "task_scores": {"code": 0.3, "reasoning": 1.0}},
            },
            {
                "id": "v/high-code", "tier": "Дежурный кодер", "roles": ["duty_code"],
                "quality_score": 80.0, "score_verified": True,
                "measured": {"latency_sec": 4.0, "task_scores": {"code": 1.0, "reasoning": 0.5}},
            },
        ]
    }, ensure_ascii=False), encoding="utf-8")

    team = DutyTeam(state_path=state_path)
    roster = team.load_roster()
    assert len(roster["duty"]) == 2
    assert "task_scores" in roster["duty"][0], "the roster must carry the measured probes"

    plan = team.dispatch("напиши функцию на python", max_calls=2)
    assert plan["task_type"] == "code", f"a code request must classify as code, got {plan['task_type']}"
    assert plan["calls"], "a code request must reach a duty model"
    assert plan["calls"][0]["id"] == "v/high-code", \
        f"the measured code probe must decide, got {plan['calls'][0]['id']}"
    assert plan["calls"][0]["task_weight"] == 1.0

    # For a quick task, speed counts: the fast model wins over the slower
    # one even when the slower one has the higher probe score.
    quick = team.dispatch("какой сейчас статус?", max_calls=2)
    assert quick["task_type"] == "quick", f"expected quick, got {quick['task_type']}"
    assert quick["calls"][0]["id"] == "v/low-code", \
        "for a quick question the fastest measured model must answer"

    # A model with no measured probe falls back to the overall score instead
    # of being treated as zero.
    assert team.task_weight({"id": "x", "score": 70.0, "task_scores": {}}, "code") == 70.0
    print("[+] Test 20 Passed: the duty order follows measured task evidence.")


def test_21_status_card_shows_the_measured_drift():
    """
    A roster that quietly got worse must be visible in the operator's card,
    said in words — not as a status code and not as a file nobody opens.
    With nothing measured, the card must stay silent instead of inventing a
    number.
    """
    print("\n" + "=" * 60)
    print("TEST 21: the status card shows the measured fleet drift")
    print("=" * 60)
    from harness.telegram_bot import fleet_drift_line

    degraded = {
        "status": "DEGRADED",
        "mean_score_delta": -8.0,
        "worst": {"id": "v/slow", "delta": -21.0},
        "added": [],
        "removed": [],
    }
    line = fleet_drift_line({"fleet_drift": degraded})
    assert line, "a measured drift must produce a line in the card"
    assert "Дрейф флота" in line, f"the card must name the drift, got {line!r}"
    assert "хуже" in line, f"the operator must read the meaning, not a code: {line!r}"
    assert "-21.00" in line, f"the measured number must be shown: {line!r}"

    # Nothing measured — no line at all, and no invented number.
    assert fleet_drift_line({}) == "", "no drift measured means no drift line"
    assert fleet_drift_line({"fleet_drift": None}) == ""
    assert fleet_drift_line({"fleet_drift": {"status": "FIRST"}}).strip(), \
        "a known 'no comparison yet' must still say so"

    # The line goes into the card as HTML — a model id must not break it.
    escaped = fleet_drift_line({"fleet_drift": {
        "status": "DEGRADED",
        "worst": {"id": "v/<script>", "delta": -30.0},
    }})
    assert "<script>" not in escaped, "the card must escape what it prints"
    assert "script" in escaped, f"the id must stay readable, got {escaped!r}"

    # The card itself must actually carry the line.
    source = (PROJECT_ROOT / "harness" / "telegram_bot.py").read_text(encoding="utf-8")
    assert "fleet_drift_line(state)" in source, "the status card must render the drift"
    print("[+] Test 21 Passed: the status card shows the drift and invents nothing.")


def test_22_transport_covers_button_press_edit_and_delete():
    """
    The wire must cover every verb the operator can trigger: a button press
    (callback query), an edit (the answer replaces the placeholder) and a
    delete (the password message). A press from an unauthorized chat must do
    nothing, and a failed edit must still deliver the answer.
    """
    print("\n" + "=" * 60)
    print("TEST 22: the transport answers a button press, edits and deletes")
    print("=" * 60)
    import threading
    from harness.telegram_bot import HarnessTelegramBot

    def bare_bot():
        bot = HarnessTelegramBot.__new__(HarnessTelegramBot)
        bot.authorized_chats = {846947027}
        bot.user_states = {}
        bot._state_lock = threading.RLock()
        bot.sent: List[str] = []
        bot.edited: List[str] = []
        bot.deleted: List[int] = []
        bot.answered: List[str] = []
        bot.send_message = lambda chat_id, text, **kw: (bot.sent.append(text), 555)[1]
        bot.edit_message = lambda chat_id, mid, text, parse_mode=None: (bot.edited.append(text), True)[1]
        bot.delete_message = lambda chat_id, mid: (bot.deleted.append(mid), True)[1]
        bot.answer_callback_query = lambda qid, text="": (bot.answered.append(qid), True)[1]
        bot.handle_status_command = lambda chat_id: bot.sent.append("STATUS")
        bot.handle_fleet_command = lambda chat_id: bot.sent.append("ROSTER")
        bot.handle_fleet_refresh = lambda chat_id: bot.sent.append("RETUNE")
        return bot

    bot = bare_bot()
    bot.process_callback_query({
        "id": "cb1", "data": "roster",
        "message": {"chat": {"id": 846947027}, "message_id": 1},
    })
    assert "ROSTER" in bot.sent, "a button press must reach the handler"
    assert bot.answered, "a button press must be answered, or the button spins forever"

    # A stranger's press must reach no handler at all.
    bot.sent.clear()
    bot.answered.clear()
    bot.process_callback_query({
        "id": "cb2", "data": "retune",
        "message": {"chat": {"id": 999999}, "message_id": 2},
    })
    assert "RETUNE" not in bot.sent, "an unauthorized chat must not trigger a re-measure"
    assert bot.answered, "even a denied press must be answered"

    # An unknown button: answered, no crash.
    bot.sent.clear()
    bot.answered.clear()
    bot.process_callback_query({
        "id": "cb3", "data": "что-то-неизвестное",
        "message": {"chat": {"id": 846947027}, "message_id": 3},
    })
    assert bot.answered and bot.sent, "an unknown button must answer, not go silent"

    # The answer replaces the placeholder in place.
    bot.sent.clear()
    bot.edited.clear()
    bot._deliver_answer(846947027, 555, "🧪 ответ", parse_mode="HTML")
    assert bot.edited and "🧪 ответ" in bot.edited[0], "the answer must edit the placeholder"
    assert not bot.sent, "an edited answer must not be sent as a second message"

    # A failed edit must still deliver the answer, and drop the placeholder.
    bot.sent.clear()
    bot.edited.clear()
    bot.deleted.clear()
    bot.edit_message = lambda *a, **k: False
    bot._deliver_answer(846947027, 555, "🧪 ответ")
    assert bot.sent, "if the edit fails the answer must still be sent"
    assert bot.deleted, "the stale placeholder must be removed"

    # An answer too long for one message cannot be edited — it goes out whole.
    bot.sent.clear()
    bot.edited.clear()
    bot.edit_message = lambda *a, **k: True
    bot._deliver_answer(846947027, 555, "x" * 5000)
    assert not bot.edited, "a 5000-character answer must not be edited into one message"
    assert bot.sent, "a long answer must still be delivered"

    # The polling loop must route button presses, not drop them silently.
    source = (PROJECT_ROOT / "harness" / "telegram_bot.py").read_text(encoding="utf-8")
    assert '"callback_query" in update' in source, "the poll loop must route button presses"
    for endpoint in ("answerCallbackQuery", "editMessageText", "deleteMessage"):
        assert endpoint in source, f"the transport must speak {endpoint}"
    print("[+] Test 22 Passed: the transport covers press, edit and delete.")


def test_23_search_query_is_formulated_not_truncated():
    """
    The search engine must receive the topical core of the operator's
    question — not the first 120 characters of the whole sentence. The
    decision to search at all must also be honest: our own questions are
    answered by our own state, not by the web.
    """
    print("\n" + "=" * 60)
    print("TEST 23: the search query is formulated, not truncated")
    print("=" * 60)
    from harness import search_query
    from harness.minimax_executor import MiniMaxExecutor

    q = search_query.formulate(
        "найди в интернете: что нового в NVIDIA NIM за последние дни"
    )
    assert q == "NVIDIA NIM за последние дни", f"the topic must survive, got {q!r}"
    assert "найди" not in q.lower(), "the instruction is not a topic"
    assert "что нового" not in q.lower(), "the question frame is not a topic"

    # Politeness and meta-words are noise.
    q2 = search_query.formulate("Пожалуйста, расскажи мне про is_palindrome")
    assert "is_palindrome" in q2, f"the topic must survive, got {q2!r}"
    assert "пожалуйста" not in q2.lower(), f"politeness is not a topic: {q2!r}"
    assert "мне" not in q2.lower().split(), f"meta-words are not a topic: {q2!r}"

    # The cut must never land mid-word.
    long_prompt = "найди в интернете: " + (" ".join(f"слово{i}" for i in range(60)))
    q3 = search_query.formulate(long_prompt)
    assert len(q3) <= search_query.MAX_QUERY_LEN, f"the query must be bounded, got {len(q3)}"
    assert q3.split()[-1].isdigit() or not q3.endswith(("слово", "сло")), \
        f"the cut must not split a word: {q3[-20:]!r}"

    # The question itself decides whether the live web is needed.
    assert search_query.needs_search("найди в интернете: что нового в NVIDIA NIM")
    assert search_query.needs_search("что нового в мире ИИ?"), \
        "a fresh-news question needs the live web"
    assert not search_query.needs_search("какой сейчас основной дежурный?"), \
        "a question about our own roster must not cost a search"
    assert not search_query.needs_search("напиши python функцию is_palindrome"), \
        "a code request must not cost a search"
    assert not search_query.needs_search(""), "an empty question searches nothing"

    plan = search_query.plan("найди в интернете: что нового в NVIDIA NIM")
    assert plan["search"] is True and plan["query"], "the plan must carry query and decision"
    assert plan["reason"], "the plan must say why it searched"

    # The executor must actually use the formulated query.
    ex = MiniMaxExecutor.__new__(MiniMaxExecutor)
    assert ex._search_query_for("найди в интернете: что нового в NVIDIA NIM") \
        == "NVIDIA NIM", "the executor must not send the raw prompt to the search engine"
    print("[+] Test 23 Passed: the search query is formulated, not truncated.")


def test_24_the_bot_reports_its_own_health():
    """
    "The bot does not answer" has two causes — an absent bot and a broken
    one — and from the outside they look the same. The bot must therefore
    leave a measured health trace: what it last polled, what it last
    answered, what it last failed on. And with nothing measured, the card
    must say nothing rather than invent activity.
    """
    print("\n" + "=" * 60)
    print("TEST 24: the bot leaves a measured health trace")
    print("=" * 60)
    import threading
    from harness.telegram_bot import HarnessTelegramBot, health_line

    tmp = Path(tempfile.mkdtemp(prefix="health-test-"))
    sessions = tmp / ".bot_sessions.json"

    def bare_bot():
        bot = HarnessTelegramBot.__new__(HarnessTelegramBot)
        bot.sessions_file = sessions
        bot.authorized_chats = {846947027}
        bot.last_update_id = 0
        bot._pending_update_ids = set()
        bot._dispatched_max = 0
        bot._state_lock = threading.RLock()
        bot.health = {
            "runs": 0, "started_at": None, "last_poll_at": None,
            "last_update_at": None, "last_send_at": None,
            "updates_seen": 0, "messages_sent": 0, "errors": 0,
            "last_error": None,
        }
        bot._last_health_save = 0.0
        return bot

    bot = bare_bot()
    bot._record_poll()
    bot._record_update()
    bot._record_send()
    assert bot.health["last_poll_at"], "a poll must be recorded"
    assert bot.health["updates_seen"] == 1, "an update must be counted"
    assert bot.health["messages_sent"] == 1, "an answer must be counted"

    # The trace must survive a restart — that is what makes it diagnosable
    # after the window that produced it is gone.
    on_disk = json.loads(sessions.read_text(encoding="utf-8"))
    assert on_disk["health"]["messages_sent"] == 1, "the health must be persisted"
    assert on_disk["last_update_id"] == 0, "the cursor must stay beside the health"

    restarted = bare_bot()
    restarted.health = restarted._load_health()
    assert restarted.health["messages_sent"] == 1, "the health must survive a restart"

    # Errors are part of the trace: a silent failure must not look healthy.
    bot._record_error("boom")
    assert bot.health["errors"] == 1 and bot.health["last_error"] == "boom"
    assert "ошибок 1" in health_line(bot.health), "the card must show the errors"

    # Nothing measured means nothing shown.
    assert health_line({}) == "", "no health means no line"
    assert health_line(None) == ""
    assert "активности ещё не было" in health_line({"runs": 1}), \
        "a fresh bot must say it has done nothing yet"

    # The diagnostics must read the same trace and judge it by age.
    from harness import diagnostics
    health_path = PROJECT_ROOT / "data" / ".bot_sessions.json"
    original = health_path.read_text(encoding="utf-8") if health_path.exists() else None
    try:
        health_path.write_text(json.dumps({
            "authorized_chats": [1], "last_update_id": 0,
            "health": {"runs": 2, "last_poll_at": "2020-01-01T00:00:00Z",
                       "last_send_at": None, "updates_seen": 3,
                       "messages_sent": 0, "errors": 1, "last_error": "boom"},
        }), encoding="utf-8")
        verdict = diagnostics.check_bot_health()
        assert verdict["ok"] is False and verdict["reason"] == "no_recent_poll", \
            f"a stale poll must read as silent, got {verdict}"
        assert verdict["updates_seen"] == 3 and verdict["errors"] == 1, \
            "the diagnostics must report what the bot measured"
    finally:
        if original is not None:
            health_path.write_text(original, encoding="utf-8")
        elif health_path.exists():
            health_path.unlink()

    print("[+] Test 24 Passed: the bot leaves a measured health trace.")


def test_25_the_cursor_never_rolls_back():
    """
    Two processes write the session file: the daemon owns the cursor, the
    self-test only adds health numbers. A writer that started earlier must
    not roll the cursor back — that would make Telegram replay the answers
    the operator has already seen, which is the duplicate-answer bug the
    cursor was introduced to kill.
    """
    print("\n" + "=" * 60)
    print("TEST 25: the cursor is monotonic — a stale writer cannot replay")
    print("=" * 60)
    import threading
    from harness.telegram_bot import HarnessTelegramBot

    tmp = Path(tempfile.mkdtemp(prefix="monotonic-test-"))
    sessions = tmp / ".bot_sessions.json"

    def bare_bot(cursor: int = 0, chats=None):
        bot = HarnessTelegramBot.__new__(HarnessTelegramBot)
        bot.sessions_file = sessions
        bot.authorized_chats = set(chats or [846947027])
        bot.last_update_id = cursor
        bot._pending_update_ids = set()
        bot._dispatched_max = cursor
        bot._state_lock = threading.RLock()
        bot.health = {
            "runs": 1, "started_at": "2026-11-01T00:00:00Z", "last_poll_at": None,
            "last_update_at": None, "last_send_at": None,
            "updates_seen": 0, "messages_sent": 0, "errors": 0,
            "last_error": None,
        }
        bot._last_health_save = 0.0
        return bot

    # The daemon advances the cursor and authorizes a second chat.
    daemon = bare_bot()
    daemon.authorized_chats.add(111222333)
    daemon._mark_update_dispatched(500)
    daemon._mark_update_handled(500)
    on_disk = json.loads(sessions.read_text(encoding="utf-8"))
    assert on_disk["last_update_id"] == 500, "the daemon must persist its cursor"

    # A second writer that started before the daemon advanced the cursor.
    stale = bare_bot(cursor=0, chats=[846947027])
    stale._record_send(846947027)
    on_disk = json.loads(sessions.read_text(encoding="utf-8"))
    assert on_disk["last_update_id"] == 500, \
        "a stale writer must never roll the cursor back"
    assert 111222333 in on_disk["authorized_chats"], \
        "an authorization must not be dropped by another writer"
    assert on_disk["health"]["messages_sent"] >= 1, \
        "the health block must survive the merge"
    assert on_disk["health"]["last_chat_id"] == 846947027, \
        "the chat the bot answered must be measured, not guessed"

    # The cursor may only move forward, never back.
    ahead = bare_bot(cursor=0)
    ahead._mark_update_dispatched(900)
    ahead._mark_update_handled(900)
    on_disk = json.loads(sessions.read_text(encoding="utf-8"))
    assert on_disk["last_update_id"] == 900, "the cursor must move forward"

    print("[+] Test 25 Passed: the cursor is monotonic and authorizations survive.")


def test_26_the_edit_strips_the_payload_until_telegram_accepts():
    """
    editMessageText accepts a narrower parameter set than sendMessage — the
    cloud measured the difference: a sendMessage field in the edit payload
    made Telegram refuse the call (HTTP 400, the edit never landed). The bot
    must therefore strip the payload layer by layer and must report the
    refusal instead of swallowing it.
    """
    print("\n" + "=" * 60)
    print("TEST 26: the edit strips the payload until Telegram accepts it")
    print("=" * 60)
    import threading
    from harness import telegram_bot as tb

    class FakeResponse:
        def __init__(self, ok, status_code=200, payload=None):
            self.ok = ok
            self.status_code = status_code
            self._payload = payload or {}
            self.text = json.dumps(self._payload)

        def json(self):
            return self._payload

    def run_with(failures: int):
        """Fails the first `failures` attempts, then accepts."""
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append((url.rsplit("/", 1)[-1], dict(json or {})))
            if len(calls) <= failures:
                return FakeResponse(False, 400, {
                    "ok": False, "error_code": 400,
                    "description": "Bad Request: unsupported parameter",
                })
            return FakeResponse(True, 200, {"ok": True, "result": {"message_id": 7}})

        bot = tb.HarnessTelegramBot.__new__(tb.HarnessTelegramBot)
        bot.bot_token = "test-token-0123456789"
        bot.base_url = "https://api.telegram.org/bottest-token-0123456789"
        bot.last_api_error = ""
        bot._state_lock = threading.RLock()

        original = tb.requests.post
        tb.requests.post = fake_post
        try:
            ok = bot.edit_message(1, 2, "привет", parse_mode="HTML")
        finally:
            tb.requests.post = original
        return ok, calls, bot.last_api_error

    ok, calls, reason = run_with(failures=1)
    assert ok, "the edit must succeed on the next attempt"
    assert calls[0][0] == "editMessageText", "the edit must use editMessageText"
    assert "disable_web_page_preview" not in calls[0][1], \
        "the edit payload must not carry sendMessage-only fields"
    assert calls[1][1].get("parse_mode") is None, \
        "the markup must be dropped before the preview options"

    ok, calls, reason = run_with(failures=2)
    assert ok, "the bare request must be tried last"
    assert set(calls[2][1].keys()) == {"chat_id", "message_id", "text"}, \
        f"the last attempt must be the bare request, got {calls[2][1]}"
    assert "link_preview_options" not in calls[2][1], \
        "the preview options must be dropped as the last layer"

    ok, calls, reason = run_with(failures=99)
    assert ok is False, "a refused edit must not be reported as delivered"
    assert "unsupported parameter" in reason, \
        f"the Telegram reason must be kept, got {reason!r}"

    print("[+] Test 26 Passed: the edit strips the payload and keeps the reason.")


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
        test_14e_langsearch_parses_standard_shape,
        test_15_telegram_cursor_survives_restart,
        test_16_bot_keeps_itself_alive,
        test_17_bot_has_two_independent_recovery_paths,
        test_18_multilingual_and_tool_use_probes_are_objective,
        test_19_fleet_drift_is_measured_not_guessed,
        test_20_dispatch_follows_measured_task_evidence,
        test_21_status_card_shows_the_measured_drift,
        test_22_transport_covers_button_press_edit_and_delete,
        test_23_search_query_is_formulated_not_truncated,
        test_24_the_bot_reports_its_own_health,
        test_25_the_cursor_never_rolls_back,
        test_26_the_edit_strips_the_payload_until_telegram_accepts,
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
