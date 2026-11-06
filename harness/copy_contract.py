#!/usr/bin/env python3
"""
The contract of ONE copy of the record — the rule in one place.

The complex keeps more than one copy of its own record in the clouds it
owns: the MEGA archive and the Proton Drive. Both are judged by the SAME
rule, and both verdicts end up in the operator's letter. If each cycle
carried its own copy of the rule, they would drift — and the letter would
show two copies judged by two standards, which is the same lie as a green
run that was never measured.

So the rule lives here, with no network and no account: it can be tested
in a second, on a laptop, with no cloud at all.

Two clauses, both learned the hard way:

  1. THE VAULT CLAUSE. A byte-identical copy of the sealed vault is not a
     backup — it is a file — unless the copy can still be OPENED. The GCM
     tag is the only thing that can tell the two apart: a wrong salt or a
     tampered ciphertext fails the tag instead of yielding garbage. So a
     copy whose vault holds entries that do not open is RED: that
     credential is already lost, however faithfully the bytes travelled.

     One exception, and it is the difference between honesty and noise: a
     cycle that holds the WRONG salt cannot tell a damaged copy from a
     fine one. That verdict is UNKNOWN — a named note, not a red. The
     witness is the original: if the original opens with the salt at hand
     and the copy does not, the copy is damaged; if neither opens, this
     cycle simply cannot read the vault. «I cannot read it» must never
     become «it is broken».

  2. THE STALENESS CLAUSE. A verdict is only about the moment it was
     measured. The supervisor's own rule is that a copy older than its
     cycle's age (26h for both cycles) is owed a fresh run — so a verdict
     older than that describes a copy nobody has looked at lately, and it
     must not be painted green.

Usage:
    from harness.copy_contract import copy_verdict, COPY_STALE_AFTER_H
"""

from typing import Any, Dict, List, Optional

# How old a copy's verdict may be before it stops counting as current.
# The number is the supervisor's own rule (cycle_supervisor.py: both the
# archive and the drive cycles are owed a fresh run after 26h), not a
# number invented for the letter.
COPY_STALE_AFTER_H = 26.0


def copy_verdict(count: int, verified: int, errors: List[str],
                 vault: Optional[Dict[str, Any]] = None) -> str:
    """
    The verdict of ONE copy.

      ok      — everything we had is there, reads back identical, and the
                copy of the credentials can be restored.
      partial — something did not make it, did not read back, or the vault
                does not open. RED: the promise was broken.
      failed  — nothing was archived at all. RED.

    An empty vault does NOT make the copy red: a vault with no entries has
    nothing to restore, which is a different fact from a vault whose
    entries do not open. Neither does an UNKNOWN vault verdict: that says
    this cycle cannot read the vault, not that the copy is damaged.
    """
    vault = vault or {}
    vault_broken = (bool(vault.get("entries")) and not vault.get("ok")
                    and not vault.get("unknown"))
    if count and verified == count and not errors and not vault_broken:
        return "ok"
    return "partial" if count else "failed"
