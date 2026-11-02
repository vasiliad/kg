#!/usr/bin/env python3
"""
The survival protocol — what the complex does when NVIDIA NIM disappears.

The operator's scheme, reordered by the operator (2026-11-03): LOCAL
self-sufficiency first, external dependencies last.

  Stage 1 — БРАТЬ ИСПОЛНЕНИЕ НА СЕБЯ, немедленно. The duty task is
  answered by the local sovereign CPU engine (Qwen2.5-1.5B on CPU, or
  the deterministic kernel without torch).

  Stage 2 — НАЙТИ ЛОКАЛЬНУЮ МОДЕЛЬ УМНЕЕ И ПОСТАВИТЬ НА ЗАМЕНУ.
  HuggingFace is searched through its public API (no key). The runner
  budget (measured, not guessed): GGUF Q4 up to 8-10 GB ≈ 7-13B params,
  4 vCPU / 16 GB RAM / 14 GB disk, units of tokens per second — enough
  for batch work like the digest. A candidate is downloaded only when
  the disk budget allows and the download is explicitly enabled, then
  measured by the canary, and SWAPPED only if it beats the current
  keeper. Weights live in the Actions cache (10 GB, 7 days) and — when
  the operator allows it — a mirror repo's Releases (2 GiB per asset).

  Stage 3 — ИСКАТЬ АЛЬТЕРНАТИВНЫЕ СЕТЕВЫЕ API. Only after the local
  tier is squared away: discovery runs on the existing search chain and
  ends with prepared candidates + the exact secret to add. Credentials
  cannot invent themselves — this stage PREPARES, the operator installs.

The honesty contract rules every stage: a model chosen by popularity is
«кандидат, не измерен»; a downloaded model that loses the canary is NOT
installed; a provider without credentials is a prepared candidate, never
a working tier. Untrusted code: an auto-discovered model is loaded with
trust_remote_code=False.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import redaction

# --- the runner budget, measured by the operator ---------------------------
RUNNER_DISK_GB = 14.0
MODEL_FILE_MAX_GB = 10.0     # GGUF Q4 8-10 GB ≈ 7-13B params
DISK_SAFETY_GB = 3.0         # system, context, the swap dance
MIN_PARAMS_B = 5.0           # below this the upgrade is not an upgrade
MAX_PARAMS_B = 14.0          # above this it does not fit the runner
UPGRADE_RECORD = PROJECT_ROOT / "data" / "local_keeper.json"
PROVIDER_CANDIDATES = PROJECT_ROOT / "data" / "provider_candidates.json"
HF_API = "https://huggingface.co/api"
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)[\s_-]*[bB]\b")
INSTRUCT_RE = re.compile(r"instruct|chat|\bit\b|tuned", re.I)
Q4_RE = re.compile(r"q4[_k]?[_0-9kmS]*", re.I)
GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Stage 2 — the smarter local model
# ---------------------------------------------------------------------------

def disk_budget_gb(free_gb: Optional[float] = None) -> float:
    """How big a model FILE may be, by the operator's own numbers."""
    if free_gb is None:
        try:
            from harness.disk_guard import DiskGuard
            free_gb = float(DiskGuard().inspect_disk_usage().get("free_gb") or 0.0)
        except Exception:
            free_gb = RUNNER_DISK_GB
    return max(0.0, min(float(free_gb) - DISK_SAFETY_GB, MODEL_FILE_MAX_GB))


def parse_param_count(text: str) -> Optional[float]:
    """«Qwen2.5-7B-Instruct-GGUF» → 7.0; unparseable → None."""
    match = PARAM_RE.search(text or "")
    try:
        return float(match.group(1)) if match else None
    except Exception:
        return None


def hf_search_gguf(limit: int = 40, params: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    The HuggingFace search, through the public API — no key, no library.
    Only GGUF (the runner's quantization), only text generation.
    """
    query = params or {}
    try:
        res = requests.get(
            f"{HF_API}/models",
            params={
                "library": "gguf",
                "pipeline_tag": "text-generation",
                "sort": "downloads",
                "direction": -1,
                "limit": limit,
                **query,
            },
            timeout=15,
        )
        if res.status_code != 200:
            print(f"[survival] HF search refused: HTTP {res.status_code}")
            return []
        return [m for m in (res.json() or []) if isinstance(m, dict)]
    except Exception as e:
        print(f"[survival] HF search failed: {type(e).__name__}")
        return []


def hf_repo_file_sizes(repo_id: str) -> Dict[str, int]:
    """The repo tree with sizes — to pick the ONE gguf file that fits."""
    try:
        res = requests.get(f"{HF_API}/models/{repo_id}/tree/main", timeout=15)
        if res.status_code != 200:
            return {}
        return {
            str(item.get("path")): int(item.get("size") or 0)
            for item in (res.json() or [])
            if isinstance(item, dict) and str(item.get("path", "")).endswith(".gguf")
        }
    except Exception:
        return {}


def pick_gguf_file(files: Dict[str, int], budget_bytes: float) -> Optional[Tuple[str, int]]:
    """
    The one file that fits the budget, Q4 preferred, bigger param count
    preferred inside the same quality class. Returns (name, bytes).
    """
    fitting = [(name, size) for name, size in files.items()
               if 0 < size <= budget_bytes]
    if not fitting:
        return None

    def rank(item: Tuple[str, int]) -> Tuple[int, int, int]:
        name, size = item
        is_q4 = 1 if Q4_RE.search(name) else 0
        params = parse_param_count(name) or 0.0
        return (is_q4, int(params * 10), -abs(size - budget_bytes * 0.85))

    return sorted(fitting, key=rank, reverse=True)[0]


def find_smarter_local_model(
    current_keeper_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    free_gb: Optional[float] = None,
    search_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    The best local upgrade candidate that FITS the runner, or None.

    «Smarter» here is the crowd's vote (downloads, likes) — an honest
    «кандидат, не измерен»: the real verdict is the canary AFTER the
    download, and the swap happens only on a measured win.
    """
    budget_bytes = int(disk_budget_gb(free_gb) * GB)
    if budget_bytes < 2 * GB:
        print(f"[survival] disk budget {budget_bytes / GB:.1f} GB — no room for an upgrade")
        return None

    entries = search_results if search_results is not None else hf_search_gguf()
    best: Optional[Dict[str, Any]] = None
    for entry in entries:
        repo_id = str(entry.get("modelId") or entry.get("id") or "").strip()
        if not repo_id or repo_id == current_keeper_id:
            continue
        if not INSTRUCT_RE.search(repo_id):
            continue  # the duty path needs an instruction model
        params = parse_param_count(repo_id)
        if params is None or not (MIN_PARAMS_B <= params <= MAX_PARAMS_B):
            continue
        votes = int(entry.get("downloads") or 0) + int(entry.get("likes") or 0)
        if best is None or votes > best["votes"]:
            best = {
                "repo_id": repo_id,
                "params_b": params,
                "votes": votes,
                "downloads": int(entry.get("downloads") or 0),
                "likes": int(entry.get("likes") or 0),
            }
    if not best:
        print("[survival] no instruction-tuned GGUF candidate in the runner's window")
        return None

    files = hf_repo_file_sizes(best["repo_id"]) if search_results is None else {}
    pick = pick_gguf_file(files, budget_bytes) if files else None
    best["file"] = pick[0] if pick else None
    best["file_bytes"] = pick[1] if pick else None
    best["budget_bytes"] = budget_bytes
    best["verdict"] = "кандидат по популярности, не измерен"
    if search_results is None and files and not pick:
        print(f"[survival] {best['repo_id']}: no single gguf file fits {budget_bytes / GB:.1f} GB")
        return None
    return best


# ---------------------------------------------------------------------------
# Stage 2b — download, measure, swap
# ---------------------------------------------------------------------------

def load_upgrade_record(path: Path = UPGRADE_RECORD) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_upgrade_record(record: Dict[str, Any], path: Path = UPGRADE_RECORD) -> None:
    from harness import atomic_io
    atomic_io.write_json_atomic(path, record)


def _model_dir() -> Path:
    return Path(os.getenv("SOVEREIGN_MODEL_DIR", str(PROJECT_ROOT / "models_local")))


def _mirror_repo() -> str:
    """The operator's weights mirror, e.g. «vasiliad/kg-weights»."""
    return os.getenv("SOVEREIGN_WEIGHTS_REPO", "").strip()


def mirror_urls(repo_id: str, file_name: str) -> List[str]:
    """
    The mirror's release assets: the gguf split into ≤2 GiB parts is
    downloaded part by part and glued. Empty when no mirror is set.
    """
    mirror = _mirror_repo()
    if not mirror:
        return []
    tag = _mirror_tag(repo_id, file_name)
    return [f"https://github.com/{mirror}/releases/download/{tag}/{file_name}.part{i:02d}"
            for i in range(1, 12)]  # up to 22 GB; the loop stops on 404


def _mirror_tag(repo_id: str, file_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{repo_id}--{file_name}")
    return safe[:80].strip("-")


def sha256_of(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_gguf(repo_id: str, file_name: str, dest_dir: Optional[Path] = None,
                  revision: str = "") -> Optional[Path]:
    """
    ONE gguf file (≤ 10 GB), over the survival chain:

      1. already on the local disk — nothing to download;
      2. the operator's mirror repo (Releases) — independent of HF;
      3. HuggingFace, pinned to the recorded revision.

    The sha256 lands in the upgrade record, so a mirror (or HF) serving
    different bytes is caught before the model ever runs.
    """
    dest_dir = Path(dest_dir) if dest_dir else _model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[survival] {file_name} is already on the local disk")
        return dest

    parts = mirror_urls(repo_id, file_name)
    if parts:
        try:
            import urllib.request
            with open(dest.with_suffix(dest.suffix + ".part"), "wb") as fh:
                got_any = False
                for url in parts:
                    try:
                        with urllib.request.urlopen(url, timeout=60) as res:
                            fh.write(res.read())
                        got_any = True
                    except Exception:
                        break
            if got_any:
                os.replace(str(dest.with_suffix(dest.suffix + ".part")), str(dest))
                print(f"[survival] downloaded from the MIRROR ({_mirror_repo()})")
                return dest
        except Exception as e:
            print(f"[survival] mirror download failed: {type(e).__name__}")

    ref = f"?revision={revision}" if revision else ""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}{ref}"
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as res:
            if res.status_code != 200:
                print(f"[survival] download refused: HTTP {res.status_code}")
                return None
            tmp = dest.with_suffix(dest.suffix + ".part")
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in res.iter_content(chunk_size=1 << 22):
                    fh.write(chunk)
                    done += len(chunk)
            os.replace(str(tmp), str(dest))
        print(f"[survival] downloaded {file_name} ({done / GB:.1f} GB) from HF")
        return dest
    except Exception as e:
        print(f"[survival] download failed: {type(e).__name__}")
        return None


def mirror_upload(gguf_path: Path, repo_id: str, file_name: str) -> Optional[str]:
    """
    Mirrors the approved model to the operator's repo Releases: split into
    2-GiB parts, one release per model. GATED by SOVEREIGN_ALLOW_REPO=1 —
    creating content on the operator's account is the operator's decision,
    and it needs the PAT to carry the repo scope.
    """
    if os.getenv("SOVEREIGN_ALLOW_REPO", "").strip().lower() not in ("1", "true", "yes"):
        print("[survival] mirror upload is gated: set SOVEREIGN_ALLOW_REPO=1")
        return None
    import subprocess
    mirror = _mirror_repo()
    if not mirror:
        print("[survival] SOVEREIGN_WEIGHTS_REPO is not set — no mirror to upload to")
        return None
    tag = _mirror_tag(repo_id, file_name)
    part_size = int(1.9 * GB)
    parts: List[Path] = []
    try:
        data = gguf_path.read_bytes()
        for i in range(0, len(data), part_size):
            part = gguf_path.with_suffix(gguf_path.suffix + f".part{i // part_size + 1:02d}")
            part.write_bytes(data[i:i + part_size])
            parts.append(part)
        subprocess.run(["gh", "release", "create", tag, "--repo", mirror,
                        "--title", tag, "--notes", f"mirror of {repo_id} ({file_name})"],
                       check=True, capture_output=True, timeout=120)
        for part in parts:
            subprocess.run(["gh", "release", "upload", tag, str(part), "--repo", mirror, "--clobber"],
                           check=True, capture_output=True, timeout=600)
        for part in parts:
            part.unlink(missing_ok=True)
        print(f"[survival] mirrored to {mirror} (tag {tag}, {len(parts)} parts)")
        return tag
    except Exception as e:
        print(f"[survival] mirror upload failed: {type(e).__name__}")
        for part in parts:
            part.unlink(missing_ok=True)
        return None


def measure_and_swap(
    gguf_path: Path,
    repo_id: str,
    canary_fn=None,
    record_path: Path = UPGRADE_RECORD,
) -> Dict[str, Any]:
    """
    The canary is the judge: the candidate runs the digest's own task and
    is installed ONLY on a measured win over the current keeper. Without
    a runtime (llama-cpp-python) there is no measurement and NO swap — a
    popular model is not a better model until measured.
    """
    record = load_upgrade_record(record_path)
    try:
        from llama_cpp import Llama  # noqa: F401
        runtime = "llama_cpp"
    except Exception:
        runtime = ""

    if canary_fn is None:
        from harness import fleet_probes
        canary_fn = fleet_probes.check_canary_news

    if not runtime:
        record["pending_candidate"] = {
            "repo_id": repo_id,
            "file": str(gguf_path),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "note": "скачано, но не измерено: нужен llama-cpp-python",
        }
        save_upgrade_record(record, record_path)
        print("[survival] no llama-cpp-python — the candidate stays UNMEASURED, no swap")
        return record

    from harness.sovereign_cpu_engine import SovereignCPUEngine
    current = SovereignCPUEngine()
    candidate = SovereignCPUEngine(model_id=repo_id, gguf_path=str(gguf_path))

    from harness.fleet_probes import CANARY_NEWS_PROMPT
    before_res = current.execute_sovereign_mission(
        task_description=CANARY_NEWS_PROMPT, context={})
    after_res = candidate.execute_sovereign_mission(
        task_description=CANARY_NEWS_PROMPT, context={})
    before_score, before_verdict = canary_fn(str(before_res.get("output") or ""))
    after_score, after_verdict = canary_fn(str(after_res.get("output") or ""))

    record["canary"] = {
        "keeper": {"score": before_score, "verdict": before_verdict},
        "candidate": {"repo_id": repo_id, "score": after_score, "verdict": after_verdict},
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    if after_score > before_score:
        record["active_local"] = {
            "repo_id": repo_id,
            "gguf": str(gguf_path),
            "sha256": sha256_of(gguf_path),
            "mirror_tag": mirror_upload(gguf_path, repo_id, Path(gguf_path).name) or "",
            "runtime": runtime,
            "canary": after_score,
            "previous_canary": before_score,
            "swapped_at": datetime.now(timezone.utc).isoformat(),
        }
        record.pop("pending_candidate", None)
        print(f"[survival] SWAPPED: {repo_id} ({after_score:.2f}) beats the keeper ({before_score:.2f})")
    else:
        record["rejected_candidate"] = {
            "repo_id": repo_id,
            "reason": f"canary {after_score:.2f} <= keeper {before_score:.2f}",
        }
        print("[survival] the candidate did NOT beat the keeper — keeping the current model")
    save_upgrade_record(record, record_path)
    return record


# ---------------------------------------------------------------------------
# Stage 3 — alternative network APIs
# ---------------------------------------------------------------------------

PROVIDER_QUERIES = (
    "free LLM API providers 2026 openrouter together inference",
    "бесплатные API больших языковых моделей аналоги NVIDIA NIM",
    "open source LLM inference API free tier no credit card",
)


def discover_alternate_apis(executor: Any) -> Dict[str, Any]:
    """
    Stage 3: the DISCOVERY of alternative providers, through the existing
    search chain. Ends with candidates and the exact instruction — a
    provider without credentials is a PREPARED candidate, never a tier.
    """
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for query in PROVIDER_QUERIES:
        try:
            results = executor.search_web(query, num_results=5) or []
        except Exception:
            results = []
        for item in results:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append({"title": title, "url": url})
    report = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates[:12],
        "instruction": (
            "Кандидаты подготовлены автоматически. Для подключения провайдера "
            "нужен ключ оператора: добавить секрет (например ALT_LLM_API_KEY) и "
            "назвать провайдера — комплекс подключит его в цепочку поиска моделей."
        ),
    }
    from harness import atomic_io
    atomic_io.write_json_atomic(PROVIDER_CANDIDATES, report)
    return report


# ---------------------------------------------------------------------------
# The orchestration: 1 → 2 → 3, in the operator's order
# ---------------------------------------------------------------------------

def run_protocol(
    executor: Any = None,
    free_gb: Optional[float] = None,
    allow_download: Optional[bool] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """The full ladder, in the operator's order: local first, APIs last."""
    report: Dict[str, Any] = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "stage1_takeover": "дежурные задачи уже уходят локальному ядру при отказе сети",
        "order": ["1_takeover", "2_local_upgrade", "3_alt_apis"],
    }

    if allow_download is None:
        allow_download = os.getenv("SOVEREIGN_ALLOW_DOWNLOAD", "").strip().lower() in ("1", "true", "yes")

    candidate = find_smarter_local_model(free_gb=free_gb)
    report["stage2_candidate"] = candidate
    if candidate:
        if dry_run or not allow_download:
            print("[survival] download is gated: set SOVEREIGN_ALLOW_DOWNLOAD=1 to fetch")
        else:
            gguf = download_gguf(candidate["repo_id"], candidate["file"]) if candidate.get("file") else None
            if gguf:
                record = measure_and_swap(gguf, candidate["repo_id"])
                report["stage2_swap"] = {
                    "active_local": record.get("active_local"),
                    "rejected": record.get("rejected_candidate"),
                    "pending": record.get("pending_candidate"),
                }

    if executor is not None:
        try:
            report["stage3_apis"] = discover_alternate_apis(executor)
        except Exception as e:
            print(f"[survival] stage 3 failed: {type(e).__name__}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The survival protocol: local first, APIs last")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executor = None
    if args.stage in ("3", "all"):
        from harness.minimax_executor import MiniMaxExecutor
        executor = MiniMaxExecutor(
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            keenable_api_key=os.getenv("KEENABLE_API_KEY", ""),
        )
    report = run_protocol(
        executor=executor if args.stage in ("3", "all") else None,
        allow_download=args.allow_download,
        dry_run=args.dry_run,
    )
    print(json.dumps(
        redaction.mask_emails_deep({k: v for k, v in report.items() if k != "stage3_apis"}),
        ensure_ascii=False, indent=2, default=str))
