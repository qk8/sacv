"""
tests/unit/test_no_subagent.py
================================
Verifies that AgentProvider does NOT define create_subagent.

MEDIUM-008: The create_subagent method in ClaudeAgentAdapter stored config
but never applied it — dead code. This test ensures the interface contract
does not require it, preventing future accidental re-addition.
"""
from __future__ import annotations

import inspect
import pytest
from sacv.interfaces.agent_provider import AgentProvider
from sacv.adapters.claude.claude_agent_adapter import ClaudeAgentAdapter
from sacv.testing.stub_providers import StubAgentProvider
from sacv.testing.vcr_recorder import VCRAgentProvider


class TestNoCreateSubagent:
    """create_subagent has been removed from the AgentProvider contract."""

    def test_interface_has_no_create_subagent(self):
        """AgentProvider ABC does not declare create_subagent."""
        abstract_methods = getattr(AgentProvider, "__abstractmethods__", set())
        assert "create_subagent" not in abstract_methods

    def test_interface_has_only_run_task(self):
        """AgentProvider only exposes run_task as abstract method."""
        abstract_methods = getattr(AgentProvider, "__abstractmethods__", set())
        assert abstract_methods == {"run_task"}

    def test_claude_agent_adapter_has_no_create_subagent(self):
        """ClaudeAgentAdapter does not define create_subagent."""
        assert not hasattr(ClaudeAgentAdapter, "create_subagent")

    def test_stub_agent_provider_has_no_create_subagent(self):
        """StubAgentProvider does not define create_subagent."""
        assert not hasattr(StubAgentProvider, "create_subagent")

    def test_vcr_agent_provider_has_no_create_subagent(self):
        """VCRAgentProvider does not define create_subagent."""
        assert not hasattr(VCRAgentProvider, "create_subagent")

    def test_claude_agent_adapter_only_has_run_task(self):
        """ClaudeAgentAdapter only implements run_task from the interface."""
        public_methods = {
            m for m in dir(ClaudeAgentAdapter)
            if not m.startswith("_") and callable(getattr(ClaudeAgentAdapter, m, None))
        }
        assert "create_subagent" not in public_methods
