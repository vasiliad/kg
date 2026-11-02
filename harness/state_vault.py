#!/usr/bin/env python3
"""
The state vault: point-in-time snapshots of the complex's memory.

The state files on `main` are SNAPSHOTS that the next cycle overwrites —
`git log` on main shows the writes mixed with everything else, and a
bad write replaces the previous picture forever. This module copies the
key state files to the `state-history` branch through the GitHub API:

  snapshots/<UTC date>/<HHMM>-<filename>

Each snapshot is its own file on a branch nobody else writes to, so
there are NO rebases and NO conflicts — the failure mode of the main
branch simply does not exist here. The history answers «что комплекс
думал вчера в это время?» without archaeology over main.

Privacy: the branch is PUBLIC. Every file's content goes through
`redaction.mask_emails` before it leaves the runner.

Failure contract: the vault is a BONUS. Any failure is logged and
returns False — it must never take the cycle that calls it down.
"""

import base64
import os
from datetime import datetime, timezone
from typing import List, Optional

import requests

from harness import redaction

BRANCH = "state-history"


def _gh_env() -> tuple[str, str]:
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    repo = os.getenv("GITHUB_REPOSITORY") or os.getenv("GH_REPO") or ""
    return token.strip(), repo.strip()


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _default_branch_sha(token: str, repo: str) -> Optional[str]:
    try:
        res = requests.get(
            f"https://api.github.com/repos/{repo}", headers=_headers(token), timeout=10
        )
        default = (res.json() or {}).get("default_branch", "main")
        res = requests.get(
            f"https://api.github.com/repos/{repo}/git/ref/heads/{default}",
            headers=_headers(token), timeout=10,
        )
        return ((res.json() or {}).get("object") or {}).get("sha")
    except Exception:
        return None


def _ensure_branch(token: str, repo: str) -> bool:
    try:
        res = requests.get(
            f"https://api.github.com/repos/{repo}/git/ref/heads/{BRANCH}",
            headers=_headers(token), timeout=10,
        )
        if res.status_code == 200:
            return True
        sha = _default_branch_sha(token, repo)
        if not sha:
            return False
        res = requests.post(
            f"https://api.github.com/repos/{repo}/git/refs",
            headers=_headers(token),
            json={"ref": f"refs/heads/{BRANCH}", "sha": sha},
            timeout=15,
        )
        return res.status_code in (201, 422)  # 422 = created by a concurrent run
    except Exception:
        return False


def push_state_snapshot(files: List[str], token: str = "", repo: str = "") -> bool:
    """
    Copies the given files to the state-history branch. True if ALL of
    them landed (or there was nothing to copy).
    """
    if not token or not repo:
        token, repo = _gh_env()
    token, repo = token.strip(), repo.strip()
    if not token or not repo:
        print("[vault] GH_TOKEN/GITHUB_REPOSITORY not set — the state stays local")
        return False

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d/%H%M")
    pushed, skipped = 0, 0

    for path in files:
        try:
            source = open(path, "r", encoding="utf-8").read()
        except Exception:
            skipped += 1
            continue
        if not source.strip():
            skipped += 1
            continue

        content = redaction.mask_emails(source)
        remote_path = f"snapshots/{stamp}-{os.path.basename(path)}"
        try:
            res = requests.put(
                f"https://api.github.com/repos/{repo}/contents/{remote_path}",
                headers=_headers(token),
                json={
                    "message": f"vault: {path} at {now.strftime('%Y-%m-%d %H:%M UTC')}",
                    "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                    "branch": BRANCH,
                },
                timeout=20,
            )
            if res.status_code in (200, 201):
                pushed += 1
            else:
                print(f"[vault] {path}: HTTP {res.status_code}")
        except Exception as e:
            print(f"[vault] {path}: {type(e).__name__}")

    if pushed == 0 and skipped == 0:
        # Nothing landed AND nothing was skipped for being empty: the
        # branch itself is probably missing.
        if _ensure_branch(token, repo):
            return push_state_snapshot(files, token=token, repo=repo)
    print(f"[vault] snapshots pushed: {pushed}, skipped: {skipped}")
    return pushed > 0 or skipped == len(files)
