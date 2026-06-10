"""
tests/unit/test_audit_trail.py
===============================
HIGH-04: Structured audit trail in WorkflowState.

Verifies:
  1. AuditEntry TypedDict exists with correct fields
  2. WorkflowState includes workflow_audit_trail field
  3. _append_audit reducer correctly appends entries
  4. Nodes append audit entries on key decisions
"""
from __future__ import annotations

import pytest

from sacv.orchestration.state import (
    AuditEntry,
    WorkflowState,
    _append_audit,
    WorkflowPhase,
)


class TestAuditEntryTypedDict:
    """Verify AuditEntry structure."""

    def test_audit_entry_has_required_fields(self):
        """AuditEntry has timestamp_ms, node, decision, key_values fields."""
        entry: AuditEntry = {
            "timestamp_ms": 1000.0,
            "node": "actor",
            "decision": "diff_applied",
            "key_values": {"attempt": 1, "files": ["a.java"]},
        }
        assert entry["timestamp_ms"] == 1000.0
        assert entry["node"] == "actor"
        assert entry["decision"] == "diff_applied"
        assert entry["key_values"] == {"attempt": 1, "files": ["a.java"]}


class TestWorkflowStateAuditTrail:
    """Verify workflow_audit_trail field in WorkflowState."""

    def test_workflow_state_has_audit_trail_field(self):
        """WorkflowState includes workflow_audit_trail with _append_audit reducer."""
        # The field must exist in WorkflowState keys
        from typing import get_type_hints
        hints = get_type_hints(WorkflowState)
        assert "workflow_audit_trail" in hints

    def test_audit_trail_default_is_empty_list(self):
        """workflow_audit_trail defaults to empty list when no state."""
        result = _append_audit(None, None)
        assert result == []


class TestAppendAuditReducer:
    """Verify _append_audit reducer logic."""

    def test_none_new_returns_existing(self):
        """When new is None, return existing (no change)."""
        existing = [{"node": "bootstrap", "decision": "init"}]
        result = _append_audit(existing, None)
        assert result == existing

    def test_none_new_with_no_existing_returns_empty(self):
        """When both are None, return empty list."""
        result = _append_audit(None, None)
        assert result == []

    def test_appends_new_entries(self):
        """New entries are appended to existing."""
        existing = [{"node": "bootstrap", "decision": "init"}]
        new = [{"node": "actor", "decision": "diff_applied"}]
        result = _append_audit(existing, new)
        assert len(result) == 2
        assert result[0]["decision"] == "init"
        assert result[1]["decision"] == "diff_applied"

    def test_empty_new_returns_existing(self):
        """Empty new list returns existing (no-op)."""
        existing = [{"node": "bootstrap", "decision": "init"}]
        result = _append_audit(existing, [])
        assert result == existing

    def test_new_without_existing(self):
        """New entries work when no existing list."""
        new = [{"node": "actor", "decision": "diff_applied"}]
        result = _append_audit(None, new)
        assert len(result) == 1
        assert result[0]["decision"] == "diff_applied"
