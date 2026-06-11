"""
tests/unit/test_audit_helper.py
===============================
Unit tests for the _audit.make_audit_entry helper.
"""
from __future__ import annotations

import time

from sacv.nodes._audit import make_audit_entry


class TestMakeAuditEntry:

    def test_returns_entry_with_all_fields(self):
        entry = make_audit_entry("test_node", "test_decision", {"key": "value"})
        assert entry["node"] == "test_node"
        assert entry["decision"] == "test_decision"
        assert entry["key_values"] == {"key": "value"}
        assert isinstance(entry["timestamp_ms"], float)
        assert entry["timestamp_ms"] > 0

    def test_timestamp_is_recent(self):
        before = time.time() * 1000
        entry = make_audit_entry("node", "decision", {})
        after = time.time() * 1000
        assert before <= entry["timestamp_ms"] <= after + 1000

    def test_key_values_serializable(self):
        entry = make_audit_entry(
            "tdd_gate",
            "tests_written",
            {"test_paths": ["tests/e2e/features/f1.spec.ts"], "attempt": 0},
        )
        assert entry["key_values"]["test_paths"] == ["tests/e2e/features/f1.spec.ts"]
        assert entry["key_values"]["attempt"] == 0

    def test_key_values_accepts_nested_dicts(self):
        entry = make_audit_entry(
            "preflight",
            "violations_found",
            {"lsp_errors": 2, "arch_violations": [{"rule": "layer", "file": "x.java"}]},
        )
        assert isinstance(entry["key_values"]["arch_violations"], list)

    def test_multiple_entries_have_different_timestamps(self):
        e1 = make_audit_entry("node", "decision1", {})
        time.sleep(0.01)
        e2 = make_audit_entry("node", "decision2", {})
        assert e1["timestamp_ms"] != e2["timestamp_ms"]
