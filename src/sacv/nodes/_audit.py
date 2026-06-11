"""
nodes/_audit.py
===============
Convenience helpers for creating workflow audit trail entries.

Import and call at the end of each node's return dict to populate
the ``workflow_audit_trail`` field that LangGraph merges via the
``_append_audit`` reducer.
"""
from __future__ import annotations

import time

from sacv.orchestration.state import AuditEntry


def make_audit_entry(
    node: str,
    decision: str,
    key_values: dict[str, object],
) -> AuditEntry:
    """
    Create a single AuditEntry.

    Usage::

        return {
            ...
            "workflow_audit_trail": [make_audit_entry(
                "tdd_gate",
                "tests_written",
                {"test_paths": paths, "attempt": attempt},
            )],
        }
    """
    return AuditEntry(
        timestamp_ms=time.time() * 1000,
        node=node,
        decision=decision,
        key_values=key_values,
    )
