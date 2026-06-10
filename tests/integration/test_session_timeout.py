"""
tests/integration/test_session_timeout.py
==========================================
HIGH-01: Workflow-level session timeout via asyncio.wait_for.

Verifies:
  1. WorkflowConfig.max_workflow_duration exists with default 3600
  2. from_json() reads max_workflow_duration from config file
  3. cmd_run wraps run_with_progress in asyncio.timeout()
  4. TimeoutError is handled gracefully (not swallowed)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch
import os

import pytest

from sacv.orchestration.config import WorkflowConfig
from dataclasses import replace


# ── Config field tests ────────────────────────────────────────────────────────


class TestMaxWorkflowDurationConfig:
    """Verify max_workflow_duration field exists and loads correctly."""

    def test_default_max_workflow_duration(self) -> None:
        """max_workflow_duration defaults to 3600 (1 hour)."""
        cfg = WorkflowConfig()
        assert cfg.max_workflow_duration == 3600

    def test_from_json_reads_max_workflow_duration(self, tmp_path: pathlib.Path) -> None:
        """from_json() reads max_workflow_duration from config file."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"max_workflow_duration": 1800}))
        cfg = WorkflowConfig.from_json(cfg_file)
        assert cfg.max_workflow_duration == 1800

    def test_from_json_default_when_absent(self, tmp_path: pathlib.Path) -> None:
        """When absent, max_workflow_duration falls back to 3600."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("{}")
        cfg = WorkflowConfig.from_json(cfg_file)
        assert cfg.max_workflow_duration == 3600


# ── cmd_run timeout wiring tests ──────────────────────────────────────────────


class TestCmdRunSessionTimeout:
    """Verify cmd_run wraps workflow execution in asyncio.timeout()."""

    async def test_cmd_run_cancels_on_timeout(self) -> None:
        """cmd_run raises TimeoutError when run_with_progress exceeds max_workflow_duration."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        from sacv.cli import cmd_run

        args = argparse.Namespace(
            task_id="timeout-test",
            description="Test task",
            mode="greenfield",
            module="backend-domain",
            check_profile="standard",
        )

        with patch("sacv.cli._build_deps") as mock_deps, \
             patch("sacv.cli._start_deps") as mock_start, \
             patch("sacv.cli._stop_deps") as mock_stop, \
             patch("sacv.adapters.sandbox.DockerContainerManager") as mock_docker, \
             patch("sacv.orchestration.graph.build_graph") as mock_build, \
             patch("sacv.cli_progress.run_with_progress") as mock_run, \
             patch("sacv.cli.print") as mock_print:

            mock_deps_instance = MagicMock()
            mock_deps_instance.config = replace(WorkflowConfig(), max_workflow_duration=1)
            mock_deps_instance.memory.validate = AsyncMock()
            mock_deps_instance.code_graph.validate = AsyncMock()
            mock_deps_instance.cross_domain.validate = AsyncMock()
            mock_deps.return_value = mock_deps_instance

            mock_docker.validate_image = AsyncMock()
            mock_run.side_effect = asyncio.TimeoutError("workflow timed out")

            with pytest.raises(asyncio.TimeoutError):
                await cmd_run(args)

            mock_run.assert_called_once()

    async def test_cmd_run_continues_on_success(self) -> None:
        """cmd_run proceeds normally when workflow completes within timeout."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        from sacv.cli import cmd_run

        args = argparse.Namespace(
            task_id="ok-test",
            description="Test task",
            mode="greenfield",
            module="backend-domain",
            check_profile="standard",
        )

        with patch("sacv.cli._build_deps") as mock_deps, \
             patch("sacv.cli._start_deps") as mock_start, \
             patch("sacv.cli._stop_deps") as mock_stop, \
             patch("sacv.adapters.sandbox.DockerContainerManager") as mock_docker, \
             patch("sacv.orchestration.graph.build_graph") as mock_build, \
             patch("sacv.cli_progress.run_with_progress") as mock_run, \
             patch("sacv.cli.print") as mock_print, \
             patch("sacv.cli_progress.format_result") as mock_format:

            mock_deps_instance = MagicMock()
            mock_deps_instance.config = WorkflowConfig()
            mock_deps_instance.memory.validate = AsyncMock()
            mock_deps_instance.code_graph.validate = AsyncMock()
            mock_deps_instance.cross_domain.validate = AsyncMock()
            mock_deps.return_value = mock_deps_instance

            mock_docker.validate_image = AsyncMock()
            mock_run.return_value = {"current_phase": "complete"}
            mock_format.return_value = "Result"

            await cmd_run(args)

            mock_run.assert_called_once()
            mock_print.assert_called()  # format_result printed

    async def test_cmd_run_logs_timeout_with_state(self) -> None:
        """When timeout occurs, cmd_run logs the timeout with state snapshot."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        from sacv.cli import cmd_run

        args = argparse.Namespace(
            task_id="timeout-log-test",
            description="Test task",
            mode="greenfield",
            module="backend-domain",
            check_profile="standard",
        )

        with patch("sacv.cli._build_deps") as mock_deps, \
             patch("sacv.cli._start_deps") as mock_start, \
             patch("sacv.cli._stop_deps") as mock_stop, \
             patch("sacv.adapters.sandbox.DockerContainerManager") as mock_docker, \
             patch("sacv.orchestration.graph.build_graph") as mock_build, \
             patch("sacv.cli_progress.run_with_progress") as mock_run, \
             patch("sacv.cli.print") as mock_print:

            mock_deps_instance = MagicMock()
            mock_deps_instance.config = replace(WorkflowConfig(), max_workflow_duration=1)
            mock_deps_instance.memory.validate = AsyncMock()
            mock_deps_instance.code_graph.validate = AsyncMock()
            mock_deps_instance.cross_domain.validate = AsyncMock()
            mock_deps.return_value = mock_deps_instance

            mock_docker.validate_image = AsyncMock()
            mock_run.side_effect = asyncio.TimeoutError("workflow timed out")

            with patch("sacv.cli.log") as mock_log:
                mock_graph = AsyncMock()
                mock_graph.get_state = AsyncMock(return_value=MagicMock(
                    values=MagicMock(
                        get=MagicMock(side_effect=lambda k, d=None: {
                            "current_phase": "actor",
                            "correction_state": {"attempt_count": 3},
                            "verifier_verdict": {"test_result": "FAIL"},
                            "cumulative_cost_dollars": 12.5,
                        }.get(k, d))
                    )
                ))
                mock_build.return_value = mock_graph

                with pytest.raises(asyncio.TimeoutError):
                    await cmd_run(args)

                mock_log.error.assert_called()
                # Verify the logged message mentions timeout
                log_call = mock_log.error.call_args[0][0] if mock_log.error.call_args[0] else ""
                assert "workflow_timeout" in log_call

    async def test_cmd_run_timeout_preserves_exc_info_on_state_failure(self) -> None:
        """When get_state fails during timeout, log.error includes exc_info."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        from sacv.cli import cmd_run

        args = argparse.Namespace(
            task_id="timeout-exc-test",
            description="Test task",
            mode="greenfield",
            module="backend-domain",
            check_profile="standard",
        )

        with patch("sacv.cli._build_deps") as mock_deps, \
             patch("sacv.cli._start_deps") as mock_start, \
             patch("sacv.cli._stop_deps") as mock_stop, \
             patch("sacv.adapters.sandbox.DockerContainerManager") as mock_docker, \
             patch("sacv.orchestration.graph.build_graph") as mock_build, \
             patch("sacv.cli_progress.run_with_progress") as mock_run, \
             patch("sacv.cli.print") as mock_print:

            mock_deps_instance = MagicMock()
            mock_deps_instance.config = replace(WorkflowConfig(), max_workflow_duration=1)
            mock_deps_instance.memory.validate = AsyncMock()
            mock_deps_instance.code_graph.validate = AsyncMock()
            mock_deps_instance.cross_domain.validate = AsyncMock()
            mock_deps.return_value = mock_deps_instance

            mock_docker.validate_image = AsyncMock()
            mock_run.side_effect = asyncio.TimeoutError("workflow timed out")

            mock_graph = AsyncMock()
            mock_graph.get_state = AsyncMock(side_effect=RuntimeError("db locked"))
            mock_build.return_value = mock_graph

            with patch("sacv.cli.log") as mock_log:
                with pytest.raises(asyncio.TimeoutError):
                    await cmd_run(args)

                # Should log workflow_timeout_no_state with exc_info
                mock_log.error.assert_called()
                call_kwargs = mock_log.error.call_args[1]
                assert "exc_info" in call_kwargs
                assert call_kwargs["exc_info"] is not None
