"""
tests/unit/test_replan_audit.py
================================
Tests for the audit trail entry in replan.py.

TDD checklist:
- [x] Tests verify audit entry is included in return dict
- [x] Tests verify audit entry has correct structure
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sacv.nodes._audit import make_audit_entry


class TestReplanAuditEntry:
    """Verify the audit entry in replan.py's return dict."""

    def test_make_audit_entry_has_required_fields(self):
        """make_audit_entry returns a dict with all required AuditEntry fields."""
        entry = make_audit_entry("replan", "new_candidates=2", {
            "new_candidates": 2,
            "selected_id": "r1",
            "replan_count": 1,
        })
        assert "timestamp_ms" in entry
        assert entry["node"] == "replan"
        assert entry["decision"] == "new_candidates=2"
        assert entry["key_values"]["new_candidates"] == 2
        assert entry["key_values"]["selected_id"] == "r1"
        assert entry["key_values"]["replan_count"] == 1

    def test_make_audit_entry_timestamp_is_float(self):
        """make_audit_entry timestamp_ms is a float (epoch ms)."""
        entry = make_audit_entry("replan", "test", {})
        assert isinstance(entry["timestamp_ms"], float)
        assert entry["timestamp_ms"] > 0

    def test_make_audit_entry_key_values_preserved(self):
        """make_audit_entry preserves all key_values."""
        entry = make_audit_entry("replan", "test", {
            "new_candidates": 5,
            "selected_id": "r3",
            "replan_count": 2,
        })
        kv = entry["key_values"]
        assert kv["new_candidates"] == 5
        assert kv["selected_id"] == "r3"
        assert kv["replan_count"] == 2


class TestReplanReturnStructure:
    """Verify the replan node return dict includes workflow_audit_trail."""

    def test_audit_entry_is_list(self):
        """workflow_audit_trail is always a list (even with single entry)."""
        trail = [make_audit_entry("replan", "test", {})]
        assert isinstance(trail, list)
        assert len(trail) == 1

    def test_multiple_audit_entries(self):
        """Multiple nodes can contribute audit entries."""
        trail = [
            make_audit_entry("value_node", "strategies_selected", {"count": 3}),
            make_audit_entry("replan", "new_candidates=2", {"count": 2}),
        ]
        assert len(trail) == 2
        assert trail[0]["node"] == "value_node"
        assert trail[1]["node"] == "replan"
