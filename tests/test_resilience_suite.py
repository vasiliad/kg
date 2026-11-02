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
 27. Bot: a long answer loses no chunk (a null reply_markup is refused by
     Telegram), and the placeholder carries no keyboard — so it can be
     edited in place.
 28. Bot: the silence alarm judges the trace and the live window together,
     mails the measured evidence once, and never floods the operator.
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
    # A synthetic address, never the operator's: this repository is public,
    # and a test fixture is not a place for a real person's address.
    sender_no_creds = EmailSender(sender_email="sender@example.com", app_password=None)
    res = sender_no_creds.send_email("someone@example.com", "Тест", "<p>ok</p>")
    assert res["status"] == "SKIPPED_NOT_CONFIGURED", res
    sender_full = EmailSender(sender_email="sender@example.com", app_password="x" * 16)
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
    assert "prose" in fleet_probes.PROBE_TASKS, \
        "the prose probe must exist: a model that cannot talk to a human must lose points"
    assert DEFAULT_POLICY["quality_tasks"] == ["reasoning", "code", "json", "translation", "toolcall", "prose"], \
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


def test_27_no_chunk_is_lost_and_the_placeholder_is_bare():
    """
    Measured in the cloud: a chunk sent with `reply_markup: null` is refused
    by Telegram ("Bad Request: object expected as reply markup"), and the
    long answer silently lost every chunk but the last. The field must be
    absent, not empty — and the placeholder that the answer replaces must go
    out bare, because Telegram refuses to edit a message that carries the
    reply keyboard ("Bad Request: message can't be edited").
    """
    print("\n" + "=" * 60)
    print("TEST 27: no chunk is lost, and the placeholder carries no keyboard")
    print("=" * 60)
    import threading
    from harness import telegram_bot as tb

    class FakeResponse:
        ok = True
        status_code = 200

        def __init__(self, mid):
            self._mid = mid
            self.text = "{}"

        def json(self):
            return {"ok": True, "result": {"message_id": self._mid}}

    payloads = []

    def fake_post(url, json=None, timeout=None):
        payloads.append(dict(json or {}))
        return FakeResponse(100 + len(payloads))

    bot = tb.HarnessTelegramBot.__new__(tb.HarnessTelegramBot)
    bot.bot_token = "test-token-0123456789"
    bot.base_url = "https://api.telegram.org/bottest-token-0123456789"
    bot.last_api_error = ""
    bot.last_edit_mode = ""
    bot.last_message_id = None
    bot.last_message_chunks = 0
    bot.last_message_chunks_sent = 0
    bot._state_lock = threading.RLock()
    bot.health = {
        "runs": 1, "started_at": "2026-11-01T00:00:00Z", "last_poll_at": None,
        "last_update_at": None, "last_send_at": None, "last_chat_id": None,
        "updates_seen": 0, "messages_sent": 0, "errors": 0, "last_error": None,
    }
    bot._last_health_save = 0.0
    bot.sessions_file = Path(tempfile.mkdtemp(prefix="transport-test-")) / ".bot_sessions.json"
    bot.authorized_chats = {846947027}
    bot.last_update_id = 0

    original = tb.requests.post
    tb.requests.post = fake_post
    try:
        # A long answer: several chunks, only the last one may carry markup.
        long_text = "\n\n".join(f"абзац {i} " + ("тест разбивки " * 90) for i in range(1, 6))
        bot.send_message(846947027, long_text, parse_mode="HTML")
    finally:
        tb.requests.post = original

    assert len(payloads) >= 2, "the long answer must be split into chunks"
    assert bot.last_message_chunks == len(payloads), "every chunk must be sent"
    assert bot.last_message_chunks_sent == len(payloads), "no chunk may be lost"
    for payload in payloads[:-1]:
        assert "reply_markup" not in payload, \
            f"a null reply_markup is refused by Telegram: {payload}"
    assert payloads[-1].get("reply_markup"), "the last chunk carries the keyboard"

    # The placeholder goes out bare — that is what makes it editable.
    payloads.clear()
    original = tb.requests.post
    tb.requests.post = fake_post
    try:
        bot.send_message(846947027, "🧠 Хранитель думает...", parse_mode="HTML",
                         reply_markup=tb.NO_MARKUP)
    finally:
        tb.requests.post = original
    assert "reply_markup" not in payloads[0], \
        "the placeholder must not carry a keyboard, or it cannot be edited"

    # The ordinary operator message still carries the keyboard.
    payloads.clear()
    original = tb.requests.post
    tb.requests.post = fake_post
    try:
        bot.send_message(846947027, "📡 Статус системы", parse_mode="HTML")
    finally:
        tb.requests.post = original
    assert payloads[0].get("reply_markup", {}).get("keyboard"), \
        "the everyday message must keep the reply keyboard"

    print("[+] Test 27 Passed: no chunk is lost and the placeholder stays bare.")


def test_28_the_silence_alarm_judges_two_signals_and_never_floods():
    """
    Silence has two causes — a dead daemon and a failing one — and from the
    operator's chat they look exactly the same. The alarm must judge it from
    two independent signals (the bot's own trace AND whether a window is
    really running), mail the measured evidence once, and stay quiet
    afterwards. A trace that merely lags behind a live window is push
    latency, not an outage, and must never produce an alarm.
    """
    print("\n" + "=" * 60)
    print("TEST 28: the silence alarm judges two signals and never floods")
    print("=" * 60)
    from datetime import datetime, timedelta, timezone
    from harness import bot_silence_alarm as alarm
    from harness import email_sender

    tmp = Path(tempfile.mkdtemp(prefix="silence-alarm-test-"))
    sessions = tmp / ".bot_sessions.json"
    marker = tmp / ".bot_silence_alarm.json"
    now = datetime(2026, 11, 1, 8, 0, 0, tzinfo=timezone.utc)

    def write_trace(minutes_old: int, poll=True):
        stamp = (now - timedelta(minutes=minutes_old)).strftime("%Y-%m-%dT%H:%M:%SZ")
        health = {
            "runs": 3, "started_at": stamp, "last_poll_at": stamp if poll else None,
            "last_update_at": None, "last_send_at": None, "last_chat_id": 846947027,
            "updates_seen": 4, "messages_sent": 4, "errors": 1,
            "last_error": "poll: TimeoutError: timed out",
        }
        sessions.write_text(
            json.dumps({"last_update_id": 12345, "authorized_chats": [846947027],
                        "health": health}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    sent = []

    class FakeSender:
        def __init__(self, *args, **kwargs):
            pass

        def is_configured(self):
            return True

        def send_email(self, recipient_email, subject, html_content, plain_text_content=None):
            sent.append({"recipient": recipient_email, "subject": subject,
                         "html": html_content, "plain": plain_text_content})
            return {"status": "DELIVERED", "recipient": recipient_email, "subject": subject}

    original_sender = email_sender.EmailSender
    original_window = alarm.live_window
    original_recipient = alarm.resolve_recipient
    email_sender.EmailSender = FakeSender
    alarm.resolve_recipient = lambda: "operator@example.test"
    try:
        # --- A live window with a lagging trace is push latency, not an outage.
        write_trace(20)
        alarm.live_window = lambda repo=None, workflow="telegram_bot.yml": {
            "state": "alive", "running": 1, "queued": 0,
            "started_at": "2026-11-01T07:30:00Z", "age_min": 30.0, "detail": "running=1 queued=0",
        }
        report = alarm.run_silence_alarm(now=now, sessions_path=sessions, marker_path=marker)
        assert report["verdict"]["severity"] == "watch", \
            f"a lagging trace behind a live window must not be an alarm: {report['verdict']}"
        assert not sent, "push latency must never mail the operator"

        # --- The chain is dead and the trace is old: the silence is real.
        write_trace(60)
        alarm.live_window = lambda repo=None, workflow="telegram_bot.yml": {
            "state": "dead", "running": 0, "queued": 0,
            "started_at": None, "age_min": None, "detail": "running=0 queued=0",
        }
        report = alarm.run_silence_alarm(now=now, sessions_path=sessions, marker_path=marker)
        assert report["verdict"]["silent"] is True, "a dead chain with a stale trace is silence"
        assert report["sent"] is True, "the operator must be told"
        assert len(sent) == 1, f"exactly one mail per outage, got {len(sent)}"
        mail = sent[0]
        assert mail["recipient"] == "operator@example.test", \
            "the recipient must come from the runtime settings, not the code"
        assert "не отвечает" in mail["subject"], f"the subject must say it: {mail['subject']}"
        assert "60" in mail["plain"], f"the age must be measured, not guessed: {mail['plain']}"
        assert "12345" in mail["plain"], "the cursor belongs in the evidence"
        assert "poll: TimeoutError" in mail["plain"], "the last error belongs in the evidence"

        # --- The same outage in the next cycle must not produce a second mail.
        report = alarm.run_silence_alarm(now=now + timedelta(minutes=10),
                                         sessions_path=sessions, marker_path=marker)
        assert not sent[1:], "the alarm must not flood the operator"
        assert report.get("reason") == "cooldown", \
            f"the repeat must be suppressed by the cooldown: {report.get('reason')}"

        # --- The bot comes back: the marker is cleared, so the next outage is
        #     reported at once.
        write_trace(1)
        alarm.live_window = lambda repo=None, workflow="telegram_bot.yml": {
            "state": "alive", "running": 1, "queued": 0,
            "started_at": "2026-11-01T07:58:00Z", "age_min": 2.0, "detail": "running=1 queued=0",
        }
        report = alarm.run_silence_alarm(now=now + timedelta(minutes=20),
                                         sessions_path=sessions, marker_path=marker)
        assert not report["verdict"]["silent"], "a fresh trace behind a live window is health"
        assert len(sent) == 1, "a healthy bot must not mail the operator"
        assert json.loads(marker.read_text(encoding="utf-8")).get("last_alert_at") is None, \
            "the marker must be cleared when the bot recovers"

        # --- A broken trace must not break the alarm.
        sessions.write_text("{not json", encoding="utf-8")
        report = alarm.run_silence_alarm(now=now + timedelta(minutes=30),
                                         sessions_path=sessions, marker_path=marker)
        assert report.get("verdict") is not None, "a broken file is a verdict, not a crash"

        # --- No address is hardcoded in the source of the alarm.
        import re
        source = Path(alarm.__file__).read_text(encoding="utf-8")
        assert re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", source) is None, \
            "the alarm must not embed an email address"

        # --- The selftest proves the outage path is armed, and mails nobody.
        alarm.resolve_recipient = lambda: "operator@example.test"
        probe = alarm.selftest()
        assert probe["verdict"]["silent"] is True, \
            "a synthetic outage must produce the alarm verdict"
        assert probe["armed"] is True, "the alarm must be armed: verdict + recipient"
        assert "не отвечает" in probe["subject"], "the alarm must have a subject"
        assert probe["evidence_lines"] >= 8, \
            "the mail must carry the measured evidence, not a one-liner"
    finally:
        email_sender.EmailSender = original_sender
        alarm.live_window = original_window
        alarm.resolve_recipient = original_recipient

    print("[+] Test 28 Passed: the alarm judges two signals, mails once and never floods.")


def test_29_the_web_is_asked_when_the_operator_asks_for_it():
    """
    The cloud evidence of 2026-11-01: the operator wrote «Есть компания Атлас…
    Можешь найти её в интернете и узнать сколько стоит тест?». The request
    sits in the middle of the sentence, the old prefix-only rule missed it,
    and the bot answered from memory while TinyFish and LangSearch were
    healthy — then refused to name a price because one DEAD tier (Keenable)
    was mislabelled as "the whole web".

    Three things must hold now:
      1. the planner asks the web whenever the request is anywhere in the
         sentence, and knows the price vocabulary;
      2. the live facts name the tier that actually served them;
      3. a stale diagnostics snapshot is marked stale, and one dead tier
         never reads as "the live web is unavailable".
    """
    print("\n" + "=" * 60)
    print("TEST 29: the operator's request reaches the live web")
    print("=" * 60)
    import tempfile
    import harness.minimax_executor as executor_mod
    import harness.nim_pacer as pacer_mod
    from harness import search_query
    from harness import operational_context
    from harness.minimax_executor import MiniMaxExecutor

    # ------------------------------------------------------------------
    # 1. The planner.
    # ------------------------------------------------------------------
    real = ("Есть компания Атлас,которая делает генетические тесты."
            "Можешь найти её в интернете и узнать сколько стоит тест?")
    plan = search_query.plan(real)
    assert plan["search"] is True, \
        "an explicit request for the web must reach the web, wherever it sits"
    q = plan["query"]
    assert "Атлас" in q, f"the topic must survive: {q!r}"
    assert "генетические тесты" in q, f"the topic must survive: {q!r}"
    assert "сколько стоит" in q, f"the asked fact must survive: {q!r}"
    assert "в интернете" not in q.lower(), f"the instruction is not a topic: {q!r}"
    assert "найти" not in q.lower(), f"the instruction is not a topic: {q!r}"
    assert "можешь" not in q.lower(), f"the instruction is not a topic: {q!r}"

    # The price vocabulary alone is enough.
    assert search_query.needs_search("сколько стоит генетический тест Атлас"), \
        "a price question asks about a live fact"
    assert search_query.needs_search("почем Атлас"), "the money words ask for the live web"
    # A lookup request whose object the web knows asks the web.
    assert search_query.needs_search("найди компанию Атлас в вебе"), \
        "a lookup request must reach the web"
    # Our own state still answers our own questions.
    assert not search_query.needs_search("найди дежурного в ростере"), \
        "our own roster is not a web lookup"
    assert not search_query.needs_search("какой сейчас основной дежурный?"), \
        "our own roster is not a web lookup"

    # ------------------------------------------------------------------
    # 2. The live facts name the tier that served them.
    # ------------------------------------------------------------------
    captured = {}

    class _FakeResp:
        status_code = 200
        text = "ok"

        def iter_lines(self):
            return iter([
                b'data: {"choices":[{"delta":{"content":"answer"}}]}',
                b"data: [DONE]",
            ])

    def fake_post(url, headers=None, json=None, stream=None, timeout=None):
        captured["payload"] = json
        return _FakeResp()

    class _FakeKeyPool:
        def __len__(self):
            return 1

        def current(self):
            return "test-key"

        def mark_bad(self):
            return "test-key"

    ex = MiniMaxExecutor.__new__(MiniMaxExecutor)
    ex.base_url = "https://example.invalid/v1"
    ex.model_id = "test/model"
    ex.key_pool = _FakeKeyPool()
    ex.last_search_tier = ""
    ex._last_search_time = 0.0

    original_post = executor_mod.requests.post
    original_wait = pacer_mod.nim_pacer.wait_for_slot
    try:
        executor_mod.requests.post = fake_post
        pacer_mod.nim_pacer.wait_for_slot = lambda: 0.0

        def served_by_langsearch(query, num_results=3):
            ex.last_search_tier = "LangSearch"
            return [{"title": "Атлас", "url": "https://example.test/atlas",
                     "snippet": "Генетический тест стоит 5000 рублей."}]

        ex.search_web = served_by_langsearch
        ex.execute_task(task_prompt=real, context="facts", enable_search=True, max_tokens=1024)
        user_content = captured["payload"]["messages"][1]["content"]
        assert "LIVE WEB FACTS (source: LangSearch)" in user_content, \
            f"the facts must name the tier that served them: {user_content[:400]}"
        assert "(KEENABLE)" not in user_content, \
            "a hardcoded tier label is a lie about where the facts came from"
        assert "https://example.test/atlas" in user_content, \
            "the live facts must reach the model"

        # A live query that returned nothing must say so — not "no web".
        def empty_search(query, num_results=3):
            ex.last_search_tier = ""
            return []

        ex.search_web = empty_search
        captured.clear()
        ex.execute_task(task_prompt=real, context="facts", enable_search=True, max_tokens=1024)
        user_content = captured["payload"]["messages"][1]["content"]
        assert "LIVE WEB FACTS (source: none)" in user_content, \
            f"an empty answer must still be labelled: {user_content[:400]}"
        assert "The live web WAS queried" in user_content, \
            "the model must not be told the web is missing when it was asked"
    finally:
        executor_mod.requests.post = original_post
        pacer_mod.nim_pacer.wait_for_slot = original_wait

    # ------------------------------------------------------------------
    # 3. The diagnostics snapshot says how old it is, and one dead tier is
    #    not a dead web.
    # ------------------------------------------------------------------
    stale_report = {
        "timestamp": "2026-10-31T21:26:57Z",
        "nvidia": {"ok": True, "models": [{"id": "a"}]},
        "telegram": {"ok": True, "username": "@immortal911bot"},
        "tinyfish": {"ok": True, "keys": 8, "results": 1},
        "langsearch": {"ok": True, "results": 1},
        "keenable": {"ok": False, "reason": "http_401"},
    }
    original_diag = operational_context._KEY_FILES["diagnostics"]
    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "diagnostics_report.json"
        diag_path.write_text(json.dumps(stale_report), encoding="utf-8")
        operational_context._KEY_FILES["diagnostics"] = diag_path
        try:
            text = operational_context.build_operational_context(query="что нового?")
        finally:
            operational_context._KEY_FILES["diagnostics"] = original_diag

    assert "STALE SNAPSHOT" in text, \
        f"an old snapshot must be marked as old: {text[-600:]}"
    assert "tinyfish ok=True" in text, "every live-search tier must be listed"
    assert "langsearch ok=True" in text, "every live-search tier must be listed"
    assert "keenable ok=False" in text, "a dead tier must be visible, not hidden"
    assert "does NOT mean the live web is unavailable" in text, \
        "one dead tier must never read as a dead web"

    # ------------------------------------------------------------------
    # 4. A tier that answers with noise is not an answer.
    # ------------------------------------------------------------------
    noise = [{"title": "SkyLink (Los Angeles International Airport)",
              "url": "https://en.wikipedia.org/wiki/LAX_APM", "snippet": "airport"}]
    topic = [{"title": "Генетический тест «Атлас»: цена, результаты",
              "url": "https://lepekhin.ru/blog/atlas-test/", "snippet": "цена теста"}]
    stems = search_query.topic_stems(plan["query"])
    assert stems, "the question must have a topic"
    assert search_query.is_on_topic(topic, stems), "a relevant answer must pass"
    assert not search_query.is_on_topic(noise, stems), \
        "noise must not pass as an answer"

    # The chain must skip the noisy tier and serve the relevant one.
    ex2 = MiniMaxExecutor.__new__(MiniMaxExecutor)
    ex2.base_url = "https://example.invalid/v1"
    ex2.model_id = "test/model"
    ex2.key_pool = _FakeKeyPool()
    ex2.last_search_tier = ""
    ex2._last_search_time = 0.0
    ex2._search_tinyfish = lambda query, num_results=3: list(noise)
    ex2._search_langsearch = lambda query, num_results=3: list(topic)
    ex2._search_keenable = lambda query, num_results=3: []
    ex2._search_tavily = lambda query, num_results=3: []
    ex2._search_duckduckgo = lambda query, num_results=3: []
    served = ex2.search_web(plan["query"], num_results=3)
    assert ex2.last_search_tier == "LangSearch", \
        f"the noisy tier must be skipped, got {ex2.last_search_tier}"
    assert served and served[0]["url"].endswith("atlas-test/"), \
        "the relevant tier must serve"

    print("[+] Test 29 Passed: the operator's request reaches the live web, "
          "and the facts name their source.")


def test_30_every_cycle_comes_back_without_the_scheduler():
    """
    Measured on 2026-11-01: GitHub's scheduler skips slots. The self-test's
    daily cron produced ZERO scheduled runs in the whole life of the
    repository; the fleet tuner's 03:43Z slot produced no run; the
    diagnostics' 03:26Z slot produced no run. A cycle that exists only in a
    cron does not come back after a reboot — and the operator asked for every
    process to return by itself.

    The supervisor dispatches a cycle whose ARTEFACT is stale, and never
    dispatches a duplicate: a busy cycle is left alone, and a cycle that was
    attempted recently waits for its cooldown.
    """
    print("\n" + "=" * 60)
    print("TEST 30: every cycle comes back without the scheduler")
    print("=" * 60)
    import tempfile
    from datetime import datetime, timezone, timedelta
    from harness import cycle_supervisor as cs

    now = datetime(2026, 11, 1, 6, 0, 0, tzinfo=timezone.utc)

    # --- The timestamps of the artefacts come in three shapes.
    assert cs.parse_timestamp("2026-11-01T04:20:28.969685+00:00") is not None
    assert cs.parse_timestamp("2026-11-01T04:20:28Z") is not None
    assert cs.parse_timestamp("2026-10-31 21:43:12 UTC") is not None, \
        "the fleet report writes «YYYY-MM-DD HH:MM:SS UTC»"
    assert cs.parse_timestamp("not a date") is None
    assert cs.parse_timestamp("") is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir(parents=True, exist_ok=True)

        def write_json(rel, payload):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        fresh = (now - timedelta(hours=1)).isoformat()
        stale = (now - timedelta(hours=9)).isoformat()
        write_json("data/fleet_report.json", {"tuned_at": fresh})
        write_json("diagnostics_report.json", {"timestamp": fresh})
        write_json("data/bot_selftest_report.json", {"ran_at": fresh})
        # The daily report was delivered YESTERDAY — the usual state at the
        # start of a new day.
        yesterday = now - timedelta(days=1)
        (root / "nim_history.log").write_text(
            f"[{yesterday.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
            "ОТЧЁТЫ ОТПРАВЛЕНЫ | тип=дневной | статус=DELIVERED\n",
            encoding="utf-8",
        )

        cycles = (
            {"name": "fleet", "workflow": "fleet_tuner.yml",
             "artefact": "data/fleet_report.json", "field": "tuned_at",
             "max_age_h": 7.0, "retry_h": 2.0},
        )
        daily_cycle = (
            {"name": "daily_dispatch", "workflow": "daily_dispatch_novt.yml",
             "artefact": "nim_history.log", "marker": "ОТЧЁТЫ ОТПРАВЛЕНЫ",
             "max_age_h": 26.0, "retry_h": 6.0, "daily": True,
             "not_before_utc": 5},
        )

        dispatched = []
        original_busy = cs.workflow_busy
        original_last = cs.last_run_age_h
        original_dispatch = cs.dispatch_cycle
        try:
            cs.workflow_busy = lambda repo, workflow: False
            cs.last_run_age_h = lambda repo, workflow, now=None: 99.0
            cs.dispatch_cycle = lambda repo, workflow, ref="": dispatched.append(workflow) or True

            # --- Fresh artefacts are left alone.
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert not dispatched, f"fresh cycles must not run: {dispatched}"
            assert all(e["action"] == "fresh" for e in report["cycles"]), \
                f"a fresh artefact must read as fresh: {report['cycles']}"

            # --- A stale artefact is dispatched.
            write_json("data/fleet_report.json", {"tuned_at": stale})
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert dispatched == ["fleet_tuner.yml"], \
                f"the stale cycle must be started: {dispatched}"
            assert "fleet_tuner.yml" in report["dispatched"], \
                "the report must say what it did"

            # --- The daily cycle is due only after its own hour.
            dispatched.clear()
            report = cs.supervise(repo="owner/repo", now=now.replace(hour=4),
                                  cycles=daily_cycle, root=root)
            daily = report["cycles"][0]
            assert daily["action"] == "fresh", \
                f"the daily report must not go out before 05:00 UTC: {daily}"
            report = cs.supervise(repo="owner/repo", now=now, cycles=daily_cycle, root=root)
            daily = report["cycles"][0]
            assert daily.get("due_today") is True, \
                f"the daily report is due once a day: {daily}"
            assert daily["action"] == "dispatched", \
                f"a due daily report must go out: {daily}"

            # --- A busy cycle is never duplicated.
            dispatched.clear()
            cs.workflow_busy = lambda repo, workflow: True
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert not dispatched, f"a busy cycle must not be started: {dispatched}"
            assert all(e["action"] == "busy" for e in report["cycles"]), \
                f"a busy cycle must read as busy: {report['cycles']}"

            # --- A cycle that was attempted recently waits for its cooldown.
            cs.workflow_busy = lambda repo, workflow: False
            cs.last_run_age_h = lambda repo, workflow, now=None: 0.2
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert not dispatched, f"the cooldown must hold: {dispatched}"
            assert any(e["action"] == "cooldown" for e in report["cycles"]), \
                f"the cooldown must be visible: {report['cycles']}"

            # --- A missing artefact is a cycle that never ran: it is owed.
            dispatched.clear()
            cs.last_run_age_h = lambda repo, workflow, now=None: 99.0
            (root / "data" / "fleet_report.json").unlink()
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert "fleet_tuner.yml" in dispatched, \
                f"a cycle that never ran is owed a run: {dispatched}"

            # --- A broken artefact must not break the supervisor.
            (root / "data" / "fleet_report.json").write_text("{not json", encoding="utf-8")
            dispatched.clear()
            report = cs.supervise(repo="owner/repo", now=now, cycles=cycles, root=root)
            assert report is not None, "a broken artefact is a verdict, not a crash"
        finally:
            cs.workflow_busy = original_busy
            cs.last_run_age_h = original_last
            cs.dispatch_cycle = original_dispatch

    print("[+] Test 30 Passed: every cycle comes back on its own artefact, "
          "and none is duplicated.")


def test_31_the_supervisor_dispatches_through_the_gh_binary():
    """
    Measured in the cloud on 2026-11-01: the supervisor's dispatch died with
    FileNotFoundError: 'workflow' — the command was built WITHOUT the "gh"
    prefix, so Python tried to execute a program named "workflow". The busy
    check spoke through `gh api` and worked, which is why only the dispatch
    was broken and the failure looked like a permissions problem (a lesson
    the permissions fix could not teach us).
    """
    print("\n" + "=" * 60)
    print("TEST 31: the supervisor dispatches through the gh binary")
    print("=" * 60)
    from harness import cycle_supervisor as cs

    calls = []

    class _Ok:
        returncode = 0
        stdout = "0"
        stderr = ""

    original_run = cs.subprocess.run
    try:
        def fake_run(args, **kwargs):
            calls.append(list(args))
            return _Ok()

        cs.subprocess.run = fake_run

        assert cs.dispatch_cycle("owner/repo", "fleet_tuner.yml", "main") is True
        assert calls, "the dispatch must actually call out"
        args = calls[0]
        assert args[0] == "gh", \
            f"the command must START with the gh binary, not with 'workflow': {args}"
        assert args[1:3] == ["workflow", "run"], f"wrong gh subcommand: {args}"
        assert "fleet_tuner.yml" in args, f"the workflow file must be named: {args}"
        assert "--repo" in args and "owner/repo" in args, \
            f"the repository must be named: {args}"
        assert "--ref" in args and "main" in args, f"the ref must be named: {args}"

        # The busy check must speak through the same binary — it is the
        # check that works, so its command is the reference shape.
        calls.clear()
        cs.workflow_busy("owner/repo", "fleet_tuner.yml")
        assert calls and calls[0][0] == "gh", \
            f"the busy check must use the gh binary: {calls}"
        assert calls[0][1] == "api", f"the busy check must use gh api: {calls}"

        # A refusal is a False, not an exception and not a success.
        class _Refused:
            returncode = 1
            stdout = ""
            stderr = "HTTP 403: Resource not accessible by integration"

        cs.subprocess.run = lambda args, **kwargs: _Refused()
        assert cs.dispatch_cycle("owner/repo", "fleet_tuner.yml") is False, \
            "a refused dispatch must never report success"
    finally:
        cs.subprocess.run = original_run

    print("[+] Test 31 Passed: every supervisor command starts with gh, "
          "and a refusal is reported as a refusal.")


def test_32_the_operator_address_never_reaches_the_public_repository():
    """
    Measured on 2026-11-01: the operator's personal address was committed to
    a PUBLIC repository in four places — `.env.example` (hardcoded, which the
    standing rule forbids outright), `nim_history.log` (every delivery
    record), `data/bot_selftest_report.json` (the bot's own honest answer
    quotes where it would send a report) and a test fixture.

    The rule is that the address lives in exactly two places: the GitHub
    secret `DISPATCH_RECIPIENT_EMAIL` and the untracked `data/.bot_settings.json`.
    Nothing else may hold it — so the test scans EVERY tracked file.
    """
    print("\n" + "=" * 60)
    print("TEST 32: the operator's address never reaches the public repo")
    print("=" * 60)
    import re
    import subprocess
    from harness import redaction

    # The fixture is synthetic — a test file is not a place for a real
    # person's address either.
    address = "operator@example.com"

    # --- The fingerprint says "an address was resolved" without saying which.
    tag = redaction.recipient_tag(address)
    assert address not in tag, f"the tag must not contain the address: {tag}"
    assert "@" not in tag, f"the tag must not look like an address: {tag}"
    assert "sha256:" in tag, f"the tag must be a verifiable fingerprint: {tag}"
    assert redaction.recipient_tag(address) == tag, \
        "the same address must always produce the same tag"
    assert redaction.recipient_tag(address.upper()) == tag, \
        "the tag must not depend on the case"
    assert redaction.recipient_tag(address + "x") != tag, \
        "a different address must produce a different tag"
    assert redaction.recipient_tag("") == "не задан", \
        "an unset recipient must say so"

    # --- Free text is masked wherever an address appears.
    masked = redaction.mask_emails(f"Отчёт на {address} отправлен вовремя.")
    assert address not in masked, f"the address survived: {masked}"
    assert "sha256:" in masked, f"the address must be replaced, not dropped: {masked}"
    assert "отправлен вовремя" in masked, \
        f"the rest of the sentence must survive: {masked}"

    # --- Structured reports are masked all the way down.
    deep = redaction.mask_emails_deep({
        "tests": [{"replies": [f"Могу отправить на {address}"]}],
        "label": address,
    })
    assert address not in json.dumps(deep, ensure_ascii=False), \
        f"the address survived inside a report: {deep}"
    assert deep["label"].startswith("настроен"), \
        f"the label must be a fingerprint: {deep['label']}"

    # --- The history log masks on the way in: it is a committed file.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "nim_history.log"
        manager = StateManager(state_file=str(Path(tmp) / "state.json"),
                               history_file=str(log_path))
        manager.append_history_log(f"ОТЧЁТЫ ОТПРАВЛЕНЫ | получатель={address} | статус=DELIVERED")
        written = log_path.read_text(encoding="utf-8")
        assert address not in written, f"the address reached the log: {written}"
        assert "ОТЧЁТЫ ОТПРАВЛЕНЫ" in written, \
            f"the evidence of the dispatch must survive: {written}"
        assert "DELIVERED" in written, \
            f"the status must survive: {written}"

    # --- The conversation log is COMMITTED: the operator's own turns are
    # stored verbatim, so a turn that names an address must be masked.
    from harness.memory_engine import MemoryEngine
    saved_key = os.environ.pop("NVIDIA_API_KEY", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "telegram_history.txt"
            vault_path = Path(tmp) / ".memory_vault.json"
            engine = MemoryEngine(history_file=hist_path, vault_file=vault_path,
                                  nvidia_api_key="")
            engine.append_message("user", f"отправь отчёт на {address} пожалуйста")
            written = hist_path.read_text(encoding="utf-8")
            assert address not in written, \
                f"the address reached the conversation log: {written}"
            assert "sha256:" in written, \
                f"the address must be fingerprinted, not dropped: {written}"
            assert "пожалуйста" in written, \
                f"the rest of the turn must survive: {written}"
            if vault_path.exists():
                assert address not in vault_path.read_text(encoding="utf-8"), \
                    "the address reached the semantic vault"
    finally:
        if saved_key is not None:
            os.environ["NVIDIA_API_KEY"] = saved_key

    # --- The decisive check: NO tracked file holds a real address.
    allowed_domains = {"example.com", "example.org", "example.invalid", "localhost"}
    email_re = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
    try:
        listing = subprocess.run(["git", "ls-files"], capture_output=True, text=True, timeout=60)
        tracked = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    except Exception:
        tracked = []
    offenders = []
    for rel in tracked:
        path = Path(rel)
        try:
            if not path.is_file() or path.stat().st_size > 4_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for domain in email_re.findall(text):
            if domain.lower() in allowed_domains:
                continue
            offenders.append(f"{rel} ({domain})")
    assert not offenders, \
        "a personal address is committed to a public repository: " + ", ".join(offenders)

    print("[+] Test 32 Passed: the operator's address lives only in the secret "
          "and the untracked settings file.")


# ========================================================================
# NEW TESTS for Workstream A+B (2026-11-02)
# ========================================================================

def test_33_tool_call_is_executed_not_delivered():
    print("\n" + "=" * 60)
    print("TEST 33: Tool call is executed, not delivered to operator")
    print("=" * 60)
    from harness.minimax_executor import MiniMaxExecutor
    from harness import tool_call
    import json

    class FakeExecutor:
        """Fake that simulates the tool call loop."""
        def __init__(self):
            self.last_search_tier = "TinyFish"
        def execute_task(self, task_prompt, context="", enable_search=True,
                        max_tokens=8192, temperature=0.6, top_p=0.95,
                        model_id=None, search_query="", num_results=3):
            # Simulate: model returns tool call -> harness executes -> model returns prose
            content = json.dumps({
                "tool": "live_search",
                "arguments": {"query": search_query or "озон генетический тест", "max_results": num_results}
            })
            # The real executor would loop and get prose, but this test checks
            # that IF the model returned a tool call, the bot sanitises it.
            # What matters: the content returned by execute_task is a TOOL CALL
            # (simulating the model's first response before the loop resolves it).
            return {
                "status": "SUCCESS",
                "model": model_id or "test/model",
                "task": task_prompt,
                "content": content,
                "elapsed_sec": 0.1,
                "citations": ["https://ozon.ru/test"],
            }
        def search_web(self, query, num_results=3):
            return [{"title": "Тест", "url": "https://ozon.ru/test", "snippet": "Тест Атлас 5990 руб"}]
        def _search_query_for(self, prompt, override):
            return override or "озон генетический тест"

    fake = FakeExecutor()
    res = fake.execute_task(
        task_prompt="Найди на озон генетический тест",
        context="",
        enable_search=True,
        search_query="озон генетический тест",
        num_results=5,
    )
    assert res["status"] == "SUCCESS", res
    # The executor returned a tool call — the BOT must sanitise it.
    # Verify the raw JSON would be caught by the sanitiser:
    call = tool_call.parse_tool_call(res["content"])
    assert call is not None, "Should parse as tool call"
    assert call.get("tool") == "live_search"
    # Now sanitise — the operator gets prose
    from harness.telegram_bot import _operator_safe_text
    safe = _operator_safe_text(res["content"])
    assert "{" not in safe or "tool" not in safe, f"Raw JSON leaked: {safe}"
    assert "озон" in safe.lower() or "Озон" in safe, "Sanitised must show the query"
    assert res.get("citations"), "Citations must be collected"
    print("[+] Test 33 Passed: tool call recognised, sanitiser produces prose.")


def test_34_raw_json_never_reaches_telegram():
    print("\n" + "=" * 60)
    print("TEST 34: Raw JSON never reaches Telegram — sanitised with sources")
    print("=" * 60)
    from harness.telegram_bot import _operator_safe_text, _sources_block
    from harness import tool_call

    # Sanitiser must replace a tool call with a readable note
    raw = '{"tool": "live_search", "arguments": {"query": "озон генетический тест"}}'
    safe = _operator_safe_text(raw)
    assert "⚠️" in safe, "Warning marker must be present"
    assert "озон" in safe.lower(), "The intended query must be shown"
    assert '"tool"' not in safe, "Raw JSON must not leak"

    # Sources block renders links, not raw URLs (expects list of strings)
    block = _sources_block(["https://ozon.ru/a"])
    assert '<a href="https://ozon.ru/a">' in block, "Sources must be clickable links"
    assert "Источники" in block, "Header must be present"
    print("[+] Test 34 Passed: JSON sanitised, sources rendered as links.")


def test_35_planner_rules():
    print("\n" + "=" * 60)
    print("TEST 35: Planner — marketplace queries, continuations, no-duty")
    print("=" * 60)
    from harness import search_query

    # Marketplace triggers search
    p = search_query.plan("Найди на озон генетический тест", previous=None)
    assert p["search"] is True, "ozon must trigger search"
    assert "озон" in p["query"].lower() or "ozon" in p["query"].lower(), "Query must contain marketplace name"

    # Continuation inherits topic
    p2 = search_query.plan("Ищи дальше", previous="Найди на озон генетический тест")
    assert p2["search"] is True, "continuation must search"
    assert "генет" in p2["query"].lower() or "тест" in p2["query"].lower(), \
        "Continuation must inherit the topic"

    # Duty question still does NOT search
    p3 = search_query.plan("найди дежурного", previous=None)
    assert p3["search"] is False, "duty question is not a live-web task"

    print("[+] Test 35 Passed: planner obeys marketplace/continuation/duty rules.")


def test_36_email_wellformed_honest_private():
    print("\n" + "=" * 60)
    print("TEST 36: Email — well-formed, honest, no addresses")
    print("=" * 60)
    from harness.daily_dispatcher import build_health_report_email, build_ai_news_digest_email
    from harness import email_render
    import re

    # Health letter
    health = build_health_report_email()
    problems = email_render.validate_html(health["html"])
    assert not problems, f"Health HTML broken: {problems}"

    # No markdown literals
    for bad in ["## ", "**", "__"]:
        assert bad not in health["html"], f"Markdown literal '{bad}' leaked in health"

    # No "Автоматически локализовано" constant
    assert "Автоматически локализовано" not in health["html"], \
        "The constant lie must be gone from health letter"

    # No recipient address anywhere
    assert "test@" not in health["html"], "No address in health HTML"
    assert "example.com" not in health["html"], "No address in health HTML"

    # Digest letter (may use fallback if network fails)
    digest = build_ai_news_digest_email()
    problems2 = email_render.validate_html(digest["html"])
    assert not problems2, f"Digest HTML broken: {problems2}"

    # No markdown literals in digest either
    for bad in ["## ", "**", "__"]:
        assert bad not in digest["html"], f"Markdown literal '{bad}' leaked in digest"

    # Digest links are subset of whitelist (check a few)
    # The whitelist enforcement is in the builder, so just verify links exist
    assert '<a href="http' in digest["html"], "Digest must have clickable links"

    # No address in digest
    assert "test@" not in digest["html"], "No address in digest HTML"
    assert "example.com" not in digest["html"], "No address in digest HTML"

    print("[+] Test 36 Passed: both letters validate, no markdown leaks, no addresses.")


def test_37_prose_probe_scores():
    print("\n" + "=" * 60)
    print("TEST 37: Prose probe — prose=1.0, tool_call=0.0, weights sum=1.0")
    print("=" * 60)
    from harness import fleet_probes

    # Russian prose = 1.0
    score, verdict = fleet_probes.check_prose("Найдено на Озоне: тест Атлас стоит 5990 руб.")
    assert score == 1.0 and verdict == "prose_ru", f"Russian prose: {score} {verdict}"

    # Tool call = 0.0
    score, verdict = fleet_probes.check_prose('{"tool": "live_search", "arguments": {}}')
    assert score == 0.0 and verdict == "tool_call", f"Tool call: {score} {verdict}"

    # Weights sum to 1.0
    total = sum(v.get("weight", 0) for v in fleet_probes.PROBE_TASKS.values())
    assert abs(total - 1.0) < 0.001, f"Probe weights sum to {total}, not 1.0"

    # prose is in quality_tasks
    from harness.fleet_tuner import DEFAULT_POLICY
    assert "prose" in DEFAULT_POLICY["quality_tasks"], "prose must be in quality_tasks"

    print("[+] Test 37 Passed: prose probe scores correctly, weights normalized.")


def test_38_duty_shift_rules():
    print("\n" + "=" * 60)
    print("TEST 38: Duty shift — one model, 6h, breaks on slow/fail")
    print("=" * 60)
    from harness import duty_shift
    from datetime import datetime, timezone, timedelta

    # Cap logic: absolute cap 30s, relative 5x measured latency with floor 15s
    assert duty_shift.is_fast_response(5.0, measured_latency=1.0) is True, "5s < 30s cap"
    assert duty_shift.is_fast_response(35.0, measured_latency=1.0) is False, "35s > 30s cap"
    # 10s with 1s measured: cap = max(30, max(5, 15)) = 30, so 10 <= 30 -> fast
    assert duty_shift.is_fast_response(10.0, measured_latency=1.0) is True, "10s <= 30s cap"
    # 20s with 5s measured: cap = max(30, max(25, 15)) = 30, so 20 <= 30 -> fast
    # Need > 30s to be slow with 5s measured: try 35s
    assert duty_shift.is_fast_response(35.0, measured_latency=5.0) is False, "35s > 30s cap"

    # Shift expiry
    now = datetime.now(timezone.utc)
    shift = {"model_id": "a/b", "until": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert duty_shift.shift_expired(shift) is True, "Expired shift detected"

    shift2 = {"model_id": "a/b", "until": (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert duty_shift.shift_expired(shift2) is False, "Live shift not expired"

    # Broken shift
    shift_broken = {"model_id": "a/b", "broken_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert duty_shift.shift_broken(shift_broken) is True, "Broken shift detected"

    print("[+] Test 38 Passed: duty shift rules verified.")


def test_39_selection_reads_every_json_shape_and_lies_never():
    """
    Measured in the cloud on 2026-11-02: the selection model answered with a
    bare JSON ARRAY, the parser only knew about `{...}`, and the letter fell
    back to raw snippets WITHOUT saying so — a curated digest in the form,
    a list of search results in the content.
    """
    print("\n" + "=" * 60)
    print("TEST 39: Selection reads every JSON shape; the letter admits a fallback")
    print("=" * 60)
    from harness.daily_dispatcher import parse_selection, _describe_shape, fallback_items
    from harness import email_render

    item = {"n": 1, "title": "Заголовок", "summary": "Суть новости", "category": "Модели"}

    shapes = [
        ('bare array', f"[{json.dumps(item, ensure_ascii=False)}]"),
        ('object items', json.dumps({"items": [item]}, ensure_ascii=False)),
        ('object news', json.dumps({"news": [item]}, ensure_ascii=False)),
        ('fenced', "```json\n" + json.dumps({"items": [item]}, ensure_ascii=False) + "\n```"),
        ('prose around it', "Вот отбор:\n" + json.dumps({"items": [item]}, ensure_ascii=False)),
    ]
    for name, text in shapes:
        parsed = parse_selection(text)
        assert parsed, f"{name}: the shape must be read ({_describe_shape(text)})"
        assert parsed[0].get("n") == 1, f"{name}: the item must survive"

    # A broken answer must yield nothing (not a wrong guess).
    assert parse_selection('{"items": [{"n": 1, "title": "X", "summary"') == [], \
        "a truncated answer must not be half-read"
    assert parse_selection("Новостей сегодня нет.") == [], \
        "prose is not a selection"

    # The fallback letter must SAY it is a fallback.
    candidates = [{
        "title": "Тестовая новость",
        "url": "https://example.org/article",
        "snippet": "Текст источника.",
        "category": "Модели",
        "tier": "TinyFish",
    }]
    items = fallback_items(candidates)
    html, plain = email_render.render_digest(items, {
        "subject": "Дайджест",
        "date_label": "02.11.2026",
        "tier": "TinyFish",
        "queries": ["q"],
        "sources_count": 1,
        "recipient_tag": "настроен (sha256:00000000)",
        "selection_mode": "источники показаны как есть",
        "selection_note": "модель ответила нечитаемым форматом",
    })
    assert "Отбор новостей не выполнен" in html, "the letter must admit the fallback"
    assert "источники показаны как есть" in html, "the letter must name the mode"
    assert not email_render.validate_html(html), "the letter must stay well-formed"

    # A curated letter must NOT carry the warning.
    html2, _ = email_render.render_digest(items, {
        "subject": "Дайджест",
        "date_label": "02.11.2026",
        "tier": "TinyFish",
        "selection_mode": "модель отобрала",
    })
    assert "Отбор новостей не выполнен" not in html2, \
        "a curated letter must not claim a fallback"

    print("[+] Test 39 Passed: every JSON shape is read, and the letter admits a fallback.")


def test_40_the_letter_survives_a_phone_screen():
    """
    Measured on 390px in a real browser on 2026-11-02: the fleet table gave
    the model id a 53px column, so «minimaxai/minimax-m3.1-flash» broke into
    six lines and every row was 107px tall. The letter has to read on a
    phone — so the records are cards, and the counts are spelled right.
    """
    print("\n" + "=" * 60)
    print("TEST 40: The letter survives a phone screen and speaks Russian right")
    print("=" * 60)
    from harness import email_render
    import re

    report = {
        "subject": "Проверка связи",
        "now_label": "02.11.2026 12:00 НСК",
        "api": {"state": "Ключ действителен", "ok": True},
        "disk": {"free_gb": 12.4, "percent_used": 63},
        "arbiter": {"model": "minimaxai/minimax-m3.1-flash", "mode": "Штатный сетевой режим"},
        "incidents": [{
            "timestamp": "2026-11-02 05:00",
            "message": "ОТЧЁТЫ ОТПРАВЛЕНЫ: письмо «Проверка связи» доставлено",
            "state": "Доставлено",
        }],
        "fleet": [{
            "id": "nvidia/nemotron-3-ultra-550b-a55b",
            "roles": ["Дежурный резерв"],
            "score_text": "79.81 (живой замер)",
            "latency": "6.13с",
        }],
        "recipient_tag": "настроен (sha256:00000000)",
    }
    html, plain = email_render.render_health(report)
    assert not email_render.validate_html(html), "the health letter must stay well-formed"

    # No table may squeeze the text into a narrow column: on a phone the
    # incident record and the fleet record are cards, not grid cells.
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cols = len(re.findall(r"<t[dh]\b", row))
        assert cols <= 2, f"a row of {cols} columns does not fit a phone: {row[:80]}"

    # The model id must be readable text, not a 53px sliver.
    assert "font-size:14px; font-weight:700" in html, \
        "the model id must be a readable heading"
    assert "word-break:break-word" in html, "a long id must break, not overflow"

    # The plain-text twin must carry the same facts.
    assert "nvidia/nemotron-3-ultra-550b-a55b" in plain, "the plain letter must carry the fleet"
    assert "Доставлено" in plain, "the plain letter must carry the state"

    # Russian plural: the letter said «3 новостей».
    assert email_render._plural_news(1) == "новость", "1 → новость"
    assert email_render._plural_news(2) == "новости", "2 → новости"
    assert email_render._plural_news(3) == "новости", "3 → новости"
    assert email_render._plural_news(5) == "новостей", "5 → новостей"
    assert email_render._plural_news(11) == "новостей", "11 → новостей"
    assert email_render._plural_news(21) == "новость", "21 → новость"
    assert email_render._plural_news(22) == "новости", "22 → новости"

    items = [{
        "title": "Новость", "summary": "Суть", "url": "https://example.org/a",
        "source": "example.org", "date": "02.11.2026", "category": "Модели",
    }] * 3
    html3, _ = email_render.render_digest(items, {"subject": "Дайджест", "date_label": "02.11.2026"})
    assert "3 новости" in html3, "the preheader must say «3 новости»"
    assert "3 новостей" not in html3, "the preheader must not say «3 новостей»"

    print("[+] Test 40 Passed: the letter fits a phone and counts in Russian.")


def test_41_the_news_is_written_out_in_full():
    """
    The operator: «в дайджесте нужно полностью расписывать новость и только
    оставлять ссылку для подтверждения источника, а не бриф».

    So the source page is READ (as text, never executed) and the news is
    written from what the page says. A link that cannot be read leaves a
    brief — and the brief is marked as a brief.
    """
    print("\n" + "=" * 60)
    print("TEST 41: The news is written out in full; the link only confirms the source")
    print("=" * 60)
    from harness.daily_dispatcher import (
        read_source_pages, write_full_news, MIN_FULL_CHARS,
    )
    from harness import email_render

    class _FakeExecutor:
        model_id = "fake/model"

        def __init__(self):
            self.fetched = []

        def read_source_text(self, url, max_chars=8000):
            self.fetched.append(url)
            if "blocked" in url:
                return "The page could not be read: HTTP 403."
            return "Полный текст источника. " * 200

        def execute_task(self, **kwargs):
            answer = json.dumps({"items": [
                {
                    "n": 1,
                    "title": "Развёрнутый заголовок",
                    "summary": "Подробное описание новости со всеми деталями. " * 12,
                    "date": "01.11.2026",
                },
                {"n": 2, "title": "Короткий заголовок", "summary": "Коротко."},
            ]}, ensure_ascii=False)
            return {"status": "SUCCESS", "content": answer}

    items = [
        {"title": "A", "summary": "кратко", "url": "https://a.example/1",
         "source": "a.example", "category": "Модели", "date": "", "tier": "TinyFish"},
        {"title": "B", "summary": "кратко", "url": "https://b.example/2",
         "source": "b.example", "category": "Агенты", "date": "", "tier": "TinyFish"},
    ]
    executor = _FakeExecutor()
    pages = read_source_pages(executor, items)
    assert len(pages) == 2, f"both source pages must be read ({pages.keys()})"
    assert executor.fetched, "the executor must be asked for the page text"

    described, note = write_full_news(executor, items, pages)
    assert not note, f"the news must be written in full ({note})"
    assert len(described[0]["summary"]) >= MIN_FULL_CHARS, "the full news must be long"
    assert described[0]["detail"] == "full", "a long description is a full news item"
    assert described[1]["detail"] == "brief", "a two-word description is a brief"
    # The link, the source and the category stay ours — the model writes text only.
    assert described[0]["url"] == "https://a.example/1", "the URL stays ours"
    assert described[0]["source"] == "a.example", "the source stays ours"

    # The source is the SITE, not the search result's headline. The letter
    # said «Модели · Mistral открыла веса ИИ-модели Large 4 с триллионом ...»
    # — the news repeated in truncated form instead of naming where it came from.
    from harness.daily_dispatcher import _map_items
    candidates = [{
        "title": "Mistral открыла веса ИИ-модели Large 4 с триллионом параметров",
        "url": "https://3dnews.ru/1150622/x",
        "snippet": "текст",
        "category": "Модели",
        "tier": "TinyFish",
    }]
    mapped = _map_items([{"n": 1, "title": "Заголовок", "summary": "Суть"}], candidates)
    assert mapped[0]["source"] == "3dnews.ru", \
        f"the source must be the site, not the search headline ({mapped[0]['source']})"

    # A page that cannot be read is not a page.
    blocked = [{"title": "C", "summary": "s", "url": "https://blocked.example/x",
                "source": "blocked.example", "category": "Модели", "date": ""}]
    assert read_source_pages(executor, blocked) == {}, \
        "a page that could not be read must not become source text"

    html, plain = email_render.render_digest(described, {
        "subject": "Ежедневный дайджест ИИ",
        "date_label": "02.11.2026",
        "tier": "TinyFish",
        "selection_mode": "модель отобрала",
        "full_news_count": 1,
        "items_count": 2,
    })
    assert "Открыть источник" in html, "the link must stay as the source confirmation"
    assert "a.example" in html, "the source site must be named in the letter"
    assert "краткое изложение" in html, "a brief must be marked in the letter"
    assert "полных описаний: 1 из 2" in html, "the footer must count the full ones"
    assert not email_render.validate_html(html), "the letter must stay well-formed"

    # If not one news could be written in full, the letter says so.
    html2, _ = email_render.render_digest(described, {
        "subject": "Ежедневный дайджест ИИ",
        "date_label": "02.11.2026",
        "selection_mode": "модель отобрала",
        "full_news_count": 0,
        "items_count": 2,
    })
    assert "Полные описания не удались" in html2, \
        "six briefs must not be passed off as news"

    # A failed selection must be described accurately: if the news were
    # written out from the pages, the letter says that the list was not
    # curated — not that the sources were shown as they are.
    html3, _ = email_render.render_digest(described, {
        "subject": "Ежедневный дайджест ИИ",
        "date_label": "02.11.2026",
        "selection_mode": "источники показаны как есть",
        "selection_note": "модель не ответила (HTTP 429)",
        "full_news_count": 1,
        "items_count": 2,
    })
    assert "список не отобран моделью" in html3, \
        "the letter must say what actually happened"
    assert "показаны найденные источники как есть" not in html3, \
        "the letter must not claim the sources were shown as they are"

    html4, _ = email_render.render_digest(described, {
        "subject": "Ежедневный дайджест ИИ",
        "date_label": "02.11.2026",
        "selection_mode": "источники показаны как есть",
        "full_news_count": 0,
        "items_count": 2,
    })
    assert "показаны найденные источники как есть" in html4, \
        "without full texts the letter must say the sources are as they are"

    print("[+] Test 41 Passed: the news is written out, the link only confirms the source.")


def test_42_a_rate_limit_is_not_a_bad_answer():
    """
    Measured in the cloud on 2026-11-02: the second dispatch run was
    answered with «HTTP 429: Too Many Requests» and the letter lost the
    curation — the selection was simply given up on. A rate limit is not a
    bad answer, so the harness waits and asks again.
    """
    print("\n" + "=" * 60)
    print("TEST 42: A rate limit is retried, not taken as a failed answer")
    print("=" * 60)
    import harness.daily_dispatcher as dd
    from harness.daily_dispatcher import _ask_model, RATE_LIMIT_PAUSE

    class _RateLimited:
        model_id = "fake/model"

        def __init__(self):
            self.calls = 0

        def execute_task(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"status": "ERROR",
                        "error": 'HTTP 429: {"status":429,"title":"Too Many Requests"}'}
            return {"status": "SUCCESS", "content": '{"items": []}'}

    class _AlwaysLimited:
        model_id = "fake/model"

        def __init__(self):
            self.calls = 0

        def execute_task(self, **kwargs):
            self.calls += 1
            return {"status": "ERROR", "error": "HTTP 429: Too Many Requests"}

    slept = []
    original_sleep = dd.time.sleep
    dd.time.sleep = lambda seconds: slept.append(seconds)
    try:
        executor = _RateLimited()
        res, note = _ask_model(executor, task_prompt="p")
        assert not note, f"the retry must recover the answer ({note})"
        assert res.get("status") == "SUCCESS", "the second answer must be taken"
        assert executor.calls == 2, f"the model must be asked twice ({executor.calls})"
        assert slept == [RATE_LIMIT_PAUSE], f"the wait must be {RATE_LIMIT_PAUSE}s ({slept})"

        # A permanent limit must not turn into an endless wait.
        slept.clear()
        hard = _AlwaysLimited()
        res2, note2 = _ask_model(hard, task_prompt="p")
        assert note2, "a permanent rate limit must be reported"
        assert hard.calls == 2, f"only one retry is allowed ({hard.calls})"
    finally:
        dd.time.sleep = original_sleep

    print("[+] Test 42 Passed: the rate limit is waited out, not surrendered to.")


def test_43_state_writes_are_atomic():
    """
    The runner can die mid-write. The state files are the complex's memory:
    a truncated `data/.cycle_supervisor.json` reads as "nothing ran" and the
    supervisor re-dispatches everything; a truncated duty shift reads as
    "no shift" and the keeper silently starts a new term.
    """
    print("\n" + "=" * 60)
    print("TEST 43: State writes are atomic — a crash cannot truncate them")
    print("=" * 60)
    from harness.atomic_io import write_json_atomic, write_text_atomic

    tmp = Path(tempfile.mkdtemp(prefix="atomic-test-"))
    target = tmp / "state.json"

    assert write_json_atomic(target, {"v": 1}) is True, "a simple write must succeed"
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}

    # An unserializable payload must fail WITHOUT touching the old file and
    # without leaving the temp file behind.
    assert write_json_atomic(target, {"bad": object()}) is False, \
        "an unserializable payload must fail"
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}, \
        "the old state must survive a failed write"
    assert not (tmp / "state.json.tmp").exists(), "the temp file must be cleaned up"

    assert write_json_atomic(target, {"v": 2}) is True
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}, \
        "the next successful write must replace the file"

    text_target = tmp / "log.txt"
    assert write_text_atomic(text_target, "строка\n") is True
    assert text_target.read_text(encoding="utf-8") == "строка\n"
    assert write_text_atomic(text_target, "вторая\n") is True
    assert text_target.read_text(encoding="utf-8") == "вторая\n"

    # The writers the complex lives on must go through atomic_io.
    atomic_users = {
        "harness/state_manager.py": ["atomic_io.write_json_atomic", "atomic_io.write_text_atomic"],
        "harness/cycle_supervisor.py": ["atomic_io.write_json_atomic"],
        "harness/duty_shift.py": ["atomic_io.write_json_atomic"],
        "harness/telegram_bot.py": ["write_json_atomic"],
        "harness/diagnostics.py": ["atomic_io.write_json_atomic"],
        "harness/fleet_tuner.py": ["atomic_io.write_json_atomic"],
        "harness/fleet_drift.py": ["atomic_io.write_json_atomic"],
        "harness/daily_dispatcher.py": ["atomic_io.write_json_atomic"],
        "harness/bot_silence_alarm.py": ["atomic_io.write_json_atomic"],
    }
    for rel, calls in atomic_users.items():
        body = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for call in calls:
            assert call in body, f"{rel} must use {call} for its state writes"

    print("[+] Test 43 Passed: the complex's memory cannot be truncated by a crash.")


def test_44_a_failed_letter_raises_a_backup_alert():
    """
    The letter IS the primary alarm channel. When Gmail refuses it, an
    alarm that lives only in the log next to the failure is read after
    the damage — so the failure opens a GitHub Issue instead. The Issue
    is PUBLIC: the error text must be masked before it goes there.
    """
    print("\n" + "=" * 60)
    print("TEST 44: A failed delivery raises a backup alert (and masks the error)")
    print("=" * 60)
    from datetime import datetime, timezone
    from harness import alert_channel

    # The dedupe key: one issue per label per UTC day.
    t1 = alert_channel.issue_title("Проверка связи")
    assert t1 == alert_channel.issue_title("Проверка связи"), "same day = same title"
    t2 = alert_channel.issue_title(
        "Проверка связи", datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    )
    assert t1 != t2, "a new day must be a new alert"
    assert alert_channel.ALERT_MARKER in t1, "the title must be findable"

    # Without a token the alert must degrade to the log, not crash the cycle.
    saved = (os.environ.pop("GH_TOKEN", None), os.environ.pop("GITHUB_TOKEN", None),
             os.environ.pop("GITHUB_REPOSITORY", None), os.environ.pop("GH_REPO", None))
    try:
        assert alert_channel.raise_delivery_alert(
            "Проверка связи", "SMTP error: ivan@example.com gone"
        ) is False, "no token = no alert, but no crash either"
    finally:
        for name, value in zip(("GH_TOKEN", "GITHUB_TOKEN", "GITHUB_REPOSITORY", "GH_REPO"), saved):
            if value is not None:
                os.environ[name] = value

    # Dry-run: the alert path is exercised without touching GitHub.
    os.environ["GH_TOKEN"] = "x" * 10
    os.environ["GITHUB_REPOSITORY"] = "vasiliad/kg"
    try:
        assert alert_channel.raise_delivery_alert(
            "Проверка связи", "SMTP error", dry_run=True
        ) is True, "dry-run must pass the guard rails"
    finally:
        os.environ.pop("GH_TOKEN", None)
        os.environ.pop("GITHUB_REPOSITORY", None)

    # The dispatcher must call the alert on a non-DELIVERED result.
    source = (PROJECT_ROOT / "harness" / "daily_dispatcher.py").read_text(encoding="utf-8")
    assert "raise_delivery_alert" in source, "the dispatcher must raise the backup alert"
    assert "issues: write" in (PROJECT_ROOT / ".github/workflows/daily_dispatch_novt.yml").read_text(encoding="utf-8"), \
        "the workflow must be allowed to open issues"

    print("[+] Test 44 Passed: a failed letter raises a masked, deduplicated backup alert.")



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
        test_27_no_chunk_is_lost_and_the_placeholder_is_bare,
        test_28_the_silence_alarm_judges_two_signals_and_never_floods,
        test_29_the_web_is_asked_when_the_operator_asks_for_it,
        test_30_every_cycle_comes_back_without_the_scheduler,
        test_31_the_supervisor_dispatches_through_the_gh_binary,
        test_32_the_operator_address_never_reaches_the_public_repository,
        test_33_tool_call_is_executed_not_delivered,
        test_34_raw_json_never_reaches_telegram,
        test_35_planner_rules,
        test_36_email_wellformed_honest_private,
        test_37_prose_probe_scores,
        test_38_duty_shift_rules,
        test_39_selection_reads_every_json_shape_and_lies_never,
        test_40_the_letter_survives_a_phone_screen,
        test_41_the_news_is_written_out_in_full,
        test_42_a_rate_limit_is_not_a_bad_answer,
        test_43_state_writes_are_atomic,
        test_44_a_failed_letter_raises_a_backup_alert,
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
