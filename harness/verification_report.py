#!/usr/bin/env python3
"""
The verification suite writes down WHAT it measured.

The cloud verification is the cycle that notices when a supplier changes
something the complex depends on — a provider's API shape, a model's
disappearance, a search tier that stops answering. Until it left an
ARTEFACT, the supervisor had nothing to watch: the suite could be skipped
by the scheduler for weeks and nobody would have noticed, because a cycle
without a proof of life is a cycle without a heartbeat.

The report says three things: when the check happened, which suites ran,
and whether they passed. It is written even when a suite FAILS — a failed
verification is exactly the evidence the operator must see.

Usage:
    python harness/verification_report.py --offline PASS|FAIL --memory PASS|FAIL
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT = PROJECT_ROOT / "data" / "cloud_tests_report.json"


def _outcome(value: str) -> str:
    text = str(value or "").strip().lower()
    return "PASS" if text in ("pass", "0", "success", "true") else "FAIL"


def build_report(offline: str, memory: str, checked_at: str) -> Dict[str, Any]:
    offline_status = _outcome(offline)
    memory_status = _outcome(memory)
    return {
        "checked_at": checked_at,
        "offline_suite": offline_status,
        "memory_suite": memory_status,
        "ok": offline_status == "PASS" and memory_status == "PASS",
        "suites": {
            "offline": "tests/test_resilience_suite.py",
            "memory": "tests/test_eternal_memory_suite.py",
        },
    }


def write_report(offline: str, memory: str) -> Dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    report = build_report(offline, memory, checked_at)
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[verification] the report could not be written: {e}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", default="FAIL")
    parser.add_argument("--memory", default="FAIL")
    args = parser.parse_args()
    report = write_report(args.offline, args.memory)
    print("[verification] " + json.dumps(report, ensure_ascii=False))
    # The report is evidence, not a verdict: the WORKFLOW decides whether the
    # job is green. This script never masks a failure with a zero exit code.
    sys_exit = 0 if report.get("ok") else 1
    raise SystemExit(sys_exit)


if __name__ == "__main__":
    main()
