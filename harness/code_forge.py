#!/usr/bin/env python3
"""
The code forge — the local model writes and RUNS programs.

The operator's question (2026-11-03): «может ли резервная локальная модель
писать какие-то программы сама или по приказу заказчика — с целью улучшения
своих условий или решения задач заказчика, прямо из своего облака?»

Two flows, one constitution:

  ORDER (по приказу). The operator asks in Telegram; the duty model (or the
  sovereign take-over) writes a Python script; the forge runs it in a
  SANDBOX and returns the measured output. Sandbox rules are the repo's own
  discipline: an isolated empty directory (never the working tree), the
  interpreter in isolated mode (`python -I` — no user site, no env, no cwd
  on sys.path), a hard timeout, and no secrets in the environment.

  SELF-IMPROVEMENT (сама). The model proposes a patch to its OWN harness.
  The patch is only ever APPLIED through the full offline suite: the patch
  is written to a scratch copy, the suite must pass there, and only then
  does the patch reach the working tree as a branch the operator (or the
  cloud verification) judges. A model that breaks the suite breaks NOTHING.

Honesty contract: the forge reports the exit code, the timeout, and the
output AS MEASURED. A script that printed nothing did not succeed
quietly; a patch that was not suite-tested is never called "applied".
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The sandbox's hard edges.
FORGE_TIMEOUT_SEC = float(os.getenv("FORGE_TIMEOUT_SEC", "60"))
FORGE_MAX_OUTPUT = 8000
FORGE_TESTS_TIMEOUT_SEC = float(os.getenv("FORGE_TESTS_TIMEOUT_SEC", "900"))
# The forge's inner suite run carries this marker: the suite-in-the-scratch
# must not forge ITSELF (measured 2026-11-03: the inner test_51 cloned again
# and choked on the already-applied patch — an endless recursion of smiths).
FORGE_SKIP_SELF_ENV = "FORGE_SKIP_SELF"
# Measured 2026-11-03: this host has a SYSTEM core.autocrlf=true. The working
# tree therefore holds CRLF while the blobs hold LF — a patch written from the
# working tree (LF) does not match the clone's checkout (CRLF) and `git apply`
# refuses it. The forge pins LF for every git call it makes: the gate must
# depend on the CONTENT of the patch, not on the host's line-ending policy.
GIT = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf"]
FORBIDDEN_IN_PATCH = (
    "GH_TOKEN", "GITHUB_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_PASSWORD",
    "GMAIL_APP_PASSWORD", "NVIDIA_API_KEY", "DISPATCH_RECIPIENT_EMAIL",
    "sk-tinyfish", "ghp_", "nvapi-",
)


def run_script(
    code: str,
    args: Optional[List[str]] = None,
    timeout_sec: Optional[float] = None,
    allow_imports_of: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs the model's script in the sandbox and returns the MEASURED result.

    The script lands in its own empty directory; the interpreter runs in
    isolated mode; the working tree of the complex is never the sandbox.
    """
    timeout_sec = float(timeout_sec if timeout_sec is not None else FORGE_TIMEOUT_SEC)
    workdir = Path(tempfile.mkdtemp(prefix="forge-run-"))
    script = workdir / "script.py"
    script.write_text(code, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),   # Windows python needs it
        "PYTHONIOENCODING": "utf-8",
        "HOME": str(workdir),
    }
    # -I: isolated (no user site, no env, no cwd on sys.path).
    # -X utf8: forced UTF-8 — `-I` ignores PYTHONIOENCODING, so without -X
    # a Windows child prints cp1251 and the forge reads mojibake (measured).
    cmd = [sys.executable, "-I", "-X", "utf8", str(script), *(args or [])]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),          # the script's own dir, never the repo
            env=env,                   # no secrets reach the sandbox
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        elapsed = round(time.time() - started, 2)
        out = (proc.stdout or "")[:FORGE_MAX_OUTPUT]
        err = (proc.stderr or "")[:FORGE_MAX_OUTPUT]
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "stdout": out,
            "stderr": err,
            "timed_out": False,
            "workdir": str(workdir),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "exit_code": None, "elapsed_sec": round(time.time() - started, 2),
            "stdout": "", "stderr": f"превышен таймаут {timeout_sec:.0f}с — процесс остановлен",
            "timed_out": True, "workdir": str(workdir),
        }
    except Exception as e:
        return {
            "ok": False, "exit_code": None, "elapsed_sec": round(time.time() - started, 2),
            "stdout": "", "stderr": f"{type(e).__name__}: {e}",
            "timed_out": False, "workdir": str(workdir),
        }


def _patch_is_clean(patch_text: str) -> Tuple[bool, str]:
    """
    The patch must never touch secrets: a self-improvement that copies a
    key somewhere is the one improvement that must never happen.
    """
    for forbidden in FORBIDDEN_IN_PATCH:
        if forbidden.lower() in (patch_text or "").lower():
            return False, f"патч упоминает «{forbidden}» — секреты вне закона"
    return True, ""


# ---------------------------------------------------------------------------
# The ORDER: free-form detection, extraction, and the two prompts
# ---------------------------------------------------------------------------

# A code ORDER is not a code TOPIC. «напиши программу» is an order;
# «напиши письмо про код» is a letter; «запусти поиск» is a search. The
# marker must name BOTH the act and the artifact, and the artifact must be
# the DIRECT object of the act — otherwise the forge answers a question
# nobody asked it.
CODE_VERB_RE = re.compile(
    r"(напиши|написать|создай|создай|сделай|запусти|выполни|исполни|"
    r"write|create|make|run|execute)",
    re.I,
)
CODE_NOUN_RE = re.compile(
    r"(код\w*|скрипт\w*|программ\w*|функци\w*|модул\w*|пайтон|python|"
    r"bash|shell|скриптов)",
    re.I,
)
# The words that mean "not the forge": the direct object is a letter, an
# article or a note — and the mention of code comes later, as a topic.
NOT_CODE_OBJECT_RE = re.compile(
    r"(письм\w*|стать[ьяю]|доклад\w*|отч[её]т\w*|новост[ьию]|сообщени\w+|"
    r"заметк\w+|резюме|список|план\w*|инструкци\w+)",
    re.I,
)
# Self-improvement: the order to improve the complex itself.
SELF_MARKER_RE = re.compile(
    r"(улучш\w*\s+себя|улучш\w*\s+свой|улучш\w*\s+сво[её]|самоулучш\w*|"
    r"self[- ]?improv\w*|поправь\s+(свой|свою|сво[её])|"
    r"почини\s+(свой|свою|сво[её])|предложи\s+(патч|изменени\w+|улучшени\w+))",
    re.I,
)

# Applying a proposal that ALREADY EXISTS: the operator's «да» to a
# specific patch the complex wrote down. It is not a request to imagine
# something new — it is a request to run the stored diff through the same
# constitution.
APPLY_PROPOSAL_RE = re.compile(
    r"(примен\w*\s+(?:последн\w*\s+|эт\w*\s+|предложен\w*\s+|готов\w*\s+)?"
    r"(?:улучшен\w*|патч\w*|предложен\w*)|"
    r"apply\s+(?:the\s+)?(?:last\s+)?(?:proposal|improvement|patch))",
    re.I,
)

FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
PY_HINT_RE = re.compile(r"^[ \t]*(import\s|from\s|def\s|class\s|print\(|if\s__name__)", re.M)


def looks_like_code_order(text: str) -> bool:
    """Is this message an ORDER to write or run code? (free-form, no @role)"""
    t = (text or "").strip()
    if not t:
        return False
    if t == "/code" or t == "/forge" or t.startswith("/code ") or t.startswith("/forge "):
        return True
    if SELF_MARKER_RE.search(t):
        return True
    if APPLY_PROPOSAL_RE.search(t):
        return True
    verb = CODE_VERB_RE.search(t)
    if not verb:
        return False
    window = t[verb.end(): verb.end() + 80]
    noun = CODE_NOUN_RE.search(window)
    if not noun:
        return False
    other = NOT_CODE_OBJECT_RE.search(window)
    if other and other.start() < noun.start():
        return False          # «напиши письмо про код» — the object is a letter
    return True


def is_self_improvement_order(text: str) -> bool:
    """The order to improve the complex itself (a patch, suite-gated)."""
    return bool(SELF_MARKER_RE.search((text or "").strip()))


def is_apply_proposal_order(text: str) -> bool:
    """The order to apply a proposal that already exists (the autopilot's)."""
    return bool(APPLY_PROPOSAL_RE.search((text or "").strip()))


def extract_code(text: str) -> str:
    """The python code out of a model's answer — measured, not guessed."""
    raw = (text or "").strip()
    if not raw:
        return ""
    blocks = [b.strip() for b in FENCE_RE.findall(raw) if b.strip()]
    if blocks:
        return max(blocks, key=len)
    if PY_HINT_RE.search(raw):
        return raw
    return ""


def extract_diff(text: str) -> str:
    """The unified diff out of a model's answer (the self-improvement path)."""
    raw = (text or "").strip()
    if not raw:
        return ""
    start = raw.find("diff --git")
    if start < 0:
        start = raw.find("--- a/")
    if start < 0:
        return ""
    tail = raw[start:].strip()
    fence_end = tail.find("```")     # the closing fence is not part of the diff
    if fence_end >= 0:
        tail = tail[:fence_end].strip()
    return tail


# The two prompts. Both demand ONE artifact and nothing else — a model that
# answers with prose is a model that did not do the work.
CODE_ORDER_PROMPT = (
    "Напиши ОДИН самостоятельный Python-скрипт, который решает задачу оператора.\n"
    "Правила ответа:\n"
    "1. ТОЛЬКО код в одном блоке ```python ... ``` — без пояснений до и после;\n"
    "2. только стандартная библиотека (без pip и внешних сервисов);\n"
    "3. скрипт обязан сам завершиться и напечатать результат (print);\n"
    "4. никаких ключей, паролей и токенов — никаких секретов в коде;\n"
    "5. если данных не хватает — скрипт всё равно должен отработать и честно "
    "сказать об этом в выводе."
)

SELF_IMPROVEMENT_PROMPT = (
    "Ты — инженер самого автономного комплекса. Предложи ОДИН минимальный патч "
    "к его собственному коду (unified diff, формат `diff --git a/... b/...`).\n"
    "Правила ответа:\n"
    "1. ТОЛЬКО патч в одном блоке ```diff ... ``` — без пояснений;\n"
    "2. патч меняет ОДИН файл и решает ОДНУ задачу;\n"
    "3. никаких ключей, паролей и токенов в патче;\n"
    "4. патч не должен ломать существующие тесты;\n"
    "5. если задача не требует изменения кода — верни пустой ответ."
)


def apply_self_improvement(
    patch_text: str,
    suite_command: Optional[List[str]] = None,
    dry_run: bool = False,
    target_repo: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    The self-improvement flow: patch → scratch copy → FULL suite → branch.

    The patch NEVER lands in the working tree un-tested. The suite is the
    constitution: 50+ offline suites must pass in the scratch copy before
    the patch becomes a branch; the branch (not main) is what the operator
    or the cloud verification judges next.

    The branch lands in the COMPLEX'S OWN repository (a local branch, not
    pushed): a branch that lives only in a temp folder is a fiction nobody
    can review.
    """
    # The recursion guard lives in the CODE, not only in the test: a patch
    # that rewrites the test must not be able to restart the forge inside
    # the forge. Fail-closed — the inner run then fails the suite, and a
    # patch that breaks the suite never reaches a branch.
    if os.getenv(FORGE_SKIP_SELF_ENV) == "1":
        return {"ok": False, "stage": "guarded",
                "error": "forge внутри forge — рекурсия запрещена"}

    clean, why = _patch_is_clean(patch_text)
    if not clean:
        return {"ok": False, "stage": "vetting", "error": why}

    scratch = Path(tempfile.mkdtemp(prefix="forge-patch-"))
    if dry_run:
        return {"ok": True, "stage": "dry_run", "scratch": str(scratch)}

    try:
        subprocess.run(GIT + ["clone", "--shared", str(PROJECT_ROOT), str(scratch)],
                       check=True, capture_output=True, timeout=300)
    except Exception as e:
        return {"ok": False, "stage": "clone", "error": f"{type(e).__name__}: {e}"}

    # The patch is a unified diff; it is applied in the scratch copy only.
    # The line endings are normalised first: the model writes text, and CRLF
    # in a diff is a transport accident, not an intention. The file is
    # written in BYTES on purpose — Windows' text mode would put the CRLF
    # back (measured: the patch then refuses to apply to its own clone).
    patch_file = scratch / "proposed.patch"
    patch_file.write_bytes((patch_text or "").replace("\r\n", "\n").encode("utf-8"))
    try:
        proc = subprocess.run(GIT + ["apply", "--check", str(patch_file)],
                              cwd=str(scratch), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
        if proc.returncode != 0:
            return {"ok": False, "stage": "apply_check",
                    "error": (proc.stderr or "патч не накладывается")[:500]}
        subprocess.run(GIT + ["apply", str(patch_file)], cwd=str(scratch),
                       check=True, capture_output=True, timeout=60)
    except Exception as e:
        return {"ok": False, "stage": "apply", "error": f"{type(e).__name__}: {e}"}

    suite = suite_command or [sys.executable, "tests/test_resilience_suite.py"]
    try:
        # The gate is the OFFLINE constitution: the live-provider parts are
        # the dispatch cycle's work (measured in the cloud: the live digest
        # build alone took 17 minutes), and a patch must be judged by the
        # offline suite, not by a provider's mood.
        forge_env = {
            **os.environ,
            FORGE_SKIP_SELF_ENV: "1",
            "SUITE_OFFLINE": "1",
        }
        proc = subprocess.run(suite, cwd=str(scratch), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              env=forge_env, timeout=FORGE_TESTS_TIMEOUT_SEC)
        tail = (proc.stdout or "")[-1500:]
        if proc.returncode != 0:
            return {
                "ok": False, "stage": "suite", "error": "suite упал в песочнице",
                "suite_tail": tail,
            }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "suite", "error": "suite не уложился в таймаут"}

    branch = f"forge/improvement-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        subprocess.run(GIT + ["config", "user.name", "vasiliad"], cwd=str(scratch),
                       check=True, capture_output=True)
        subprocess.run(GIT + ["config", "user.email",
                              "vasiliadi@users.noreply.github.com"], cwd=str(scratch),
                       check=True, capture_output=True)
        subprocess.run(GIT + ["add", "-A"], cwd=str(scratch), check=True, capture_output=True)
        subprocess.run(GIT + ["commit", "-m",
                              "forge: model-proposed improvement (suite green in scratch)"],
                       cwd=str(scratch), check=True, capture_output=True, timeout=120)
        subprocess.run(GIT + ["branch", branch], cwd=str(scratch),
                       check=True, capture_output=True)
    except Exception as e:
        return {"ok": False, "stage": "commit", "error": f"{type(e).__name__}: {e}"}

    # The branch becomes VISIBLE in the complex's own repository (local only:
    # nothing is pushed — the operator or the cloud verification decides).
    repo = Path(target_repo) if target_repo else PROJECT_ROOT
    try:
        subprocess.run(GIT + ["fetch", str(scratch), f"{branch}:{branch}"],
                       cwd=str(repo), check=True, capture_output=True, timeout=120)
    except Exception as e:
        return {"ok": False, "stage": "publish_branch",
                "error": f"{type(e).__name__}: {e}", "branch": branch}

    return {
        "ok": True,
        "stage": "branched",
        "branch": branch,
        "repo": str(repo),
        "scratch": str(scratch),
        "suite_tail": tail,
        "note": "патч прошёл весь suite в копии; ветка ждёт оператора или облачную проверку",
    }
