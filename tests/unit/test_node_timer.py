"""
tests/unit/test_node_timer.py
===============================
Tests for the per-node execution timing context manager.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from sacv.nodes._node_timer import node_timer


@pytest.fixture(autouse=True)
def _patch_default_logger():
    """Prevent tests from emitting real logs via the default logger."""
    with patch("sacv.nodes._node_timer.log") as mock_log:
        yield mock_log


class TestNodeTimer:

    async def test_logs_duration_ms_on_success(self):
        """node_timer logs duration_ms on successful exit."""
        mock_logger = MagicMock()
        async with node_timer("test_node", logger=mock_logger) as timing:
            timing["files_changed"] = 3
            await asyncio.sleep(0.02)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        # First positional arg is the event string
        assert call_args[0][0] == "test_node.timing"
        kwargs = call_args[1]
        assert kwargs["files_changed"] == 3
        assert "duration_ms" in kwargs
        assert kwargs["duration_ms"] >= 15

    async def test_logs_extra_fields(self):
        """Extra fields yielded into the context manager appear in the log."""
        mock_logger = MagicMock()
        async with node_timer("extra_test", logger=mock_logger) as timing:
            timing["attempt"] = 2
            timing["branch"] = "feature-x"

        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "extra_test.timing"
        kwargs = call_args[1]
        assert kwargs["attempt"] == 2
        assert kwargs["branch"] == "feature-x"

    async def test_logs_exc_info_on_error(self):
        """node_timer logs exc_info=True when the node raises."""
        mock_logger = MagicMock()
        with pytest.raises(ValueError, match="boom"):
            async with node_timer("error_node", logger=mock_logger) as timing:
                timing["extra"] = "data"
                raise ValueError("boom")

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[0][0] == "error_node.timing_error"
        kwargs = call_args[1]
        assert kwargs["extra"] == "data"
        assert kwargs["exc_info"] is True

    async def test_uses_default_logger_when_not_provided(self, _patch_default_logger):
        """When no logger is passed, the module default logger is used."""
        async with node_timer("default_node"):
            await asyncio.sleep(0.01)

        _patch_default_logger.info.assert_called_once()
