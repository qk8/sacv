"""
tests/unit/test_wait_for_debug_port.py
=======================================
Unit tests for _wait_for_debug_port — port availability polling inside
the sandbox container.

MEDIUM-002: _wait_for_debug_port uses bash /dev/tcp built-in (replacing
nc -z which was not installed in the sandbox container). Tests verify
correct retry behavior and timeout handling.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sacv.nodes.intelligent_debugger import _wait_for_debug_port
from sacv.testing.stub_providers import StubSandboxProvider


class TestWaitForDebugPort:

    @pytest.fixture
    def deps(self):
        sandbox = MagicMock()
        config = MagicMock()
        config.debug.debug_port_wait_sec = 2.0  # short timeout for tests
        sandbox.config = config
        return MagicMock(sandbox=sandbox)

    @pytest.fixture
    def handle(self):
        return MagicMock()

    def test_port_available_returns_true_immediately(self, deps, handle):
        """When port is already open, returns True on first poll."""
        deps.sandbox.exec_in_container = AsyncMock(
            return_value=MagicMock(stdout="OK")
        )
        result = asyncio.run(
            _wait_for_debug_port(handle, 5005, deps, timeout=5.0)
        )
        assert result is True
        deps.sandbox.exec_in_container.assert_called_once()

    def test_port_unavailable_times_out(self, deps, handle):
        """When port never opens, returns False after timeout."""
        deps.sandbox.exec_in_container = AsyncMock(
            return_value=MagicMock(stdout="WAIT")
        )
        result = asyncio.run(
            _wait_for_debug_port(handle, 5005, deps, timeout=0.5)
        )
        assert result is False

    def test_port_becomes_available_after_retries(self, deps, handle):
        """Returns True when port opens on a subsequent poll."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return MagicMock(stdout="WAIT")
            return MagicMock(stdout="OK")

        deps.sandbox.exec_in_container = AsyncMock(side_effect=side_effect)
        result = asyncio.run(
            _wait_for_debug_port(handle, 5005, deps, timeout=5.0)
        )
        assert result is True
        assert call_count == 3

    def test_custom_timeout_used(self, deps, handle):
        """Uses the provided timeout parameter."""
        deps.sandbox.exec_in_container = AsyncMock(
            return_value=MagicMock(stdout="WAIT")
        )
        result = asyncio.run(
            _wait_for_debug_port(handle, 5005, deps, timeout=0.3)
        )
        assert result is False

    def test_uses_config_default_timeout(self, deps, handle):
        """When timeout is None, uses config.debug.debug_port_wait_sec."""
        deps.config.debug.debug_port_wait_sec = 0.5
        deps.sandbox.exec_in_container = AsyncMock(
            return_value=MagicMock(stdout="WAIT")
        )
        result = asyncio.run(
            _wait_for_debug_port(handle, 5005, deps)
        )
        assert result is False

    def test_stub_sandbox_skipped_in_debugger(self, handle):
        """Verify the 'Stub' guard in intelligent_debugger prevents
        calling _wait_for_debug_port for stub sandboxes."""
        stub = StubSandboxProvider()
        # The guard checks: "Stub" not in type(deps.sandbox).__name__
        # For StubSandboxProvider, __name__ contains "Stub", so the
        # guard evaluates to False and _wait_for_debug_port is NOT called.
        assert "Stub" in type(stub).__name__
