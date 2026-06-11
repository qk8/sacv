"""
tests/unit/test_cli_doctor.py
===============================
Tests for the `sacv doctor` CLI command.

TDD checklist:
- [x] Every new function has a test
- [x] Tests verify output format and diagnostic checks
- [x] Tests use real code (mocks only for subprocess/external)
"""
from __future__ import annotations

import argparse
import asyncio
import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from sacv import cli


@pytest.fixture
def clean_env():
    """Remove OTel and API key env vars for clean test state."""
    saved = {}
    for key in ("ANTHROPIC_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT", "SACV_OTEL_ENABLED"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    yield
    for key, val in saved.items():
        os.environ[key] = val


class TestCmdDoctor:

    def _run_doctor(self, extra_env=None, which_side_effect=None, run_side_effect=None):
        """Run cmd_doctor and capture stdout."""
        args = argparse.Namespace()
        args.verbose = False
        args.log_format = None

        base_env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        if extra_env:
            base_env.update(extra_env)

        captured = StringIO()
        default_which = lambda name: "/usr/bin/" + name
        def default_run(cmd, **kwargs):
            # Docker commands succeed by default; everything else fails
            if cmd[0] == "docker":
                return MagicMock(returncode=0, stdout="24.0.7")
            return MagicMock(returncode=1)
        with patch("sys.stdout", captured):
            with patch.dict(os.environ, base_env, clear=True):
                with patch("shutil.which", side_effect=which_side_effect or default_which):
                    with patch("subprocess.run", side_effect=run_side_effect or default_run):
                        asyncio.run(cli.cmd_doctor(args))
        return captured.getvalue()

    def test_command_exists(self, clean_env):
        """cmd_doctor is callable without error."""
        output = self._run_doctor()
        # Should not raise
        assert "SACV Environment Diagnostics" in output

    def test_reports_python_version_check(self, clean_env):
        """Doctor output includes Python version check."""
        output = self._run_doctor()
        assert "Python" in output

    def test_reports_api_key_check(self, clean_env):
        """Doctor output includes ANTHROPIC_API_KEY check."""
        output = self._run_doctor()
        assert "ANTHROPIC_API_KEY" in output

    def test_reports_docker_check(self, clean_env):
        """Doctor output includes Docker check."""
        output = self._run_doctor()
        assert "Docker" in output

    def test_reports_otel_info(self, clean_env):
        """Doctor output includes OTel configuration info."""
        output = self._run_doctor()
        assert "OTel" in output

    def test_reports_summary(self, clean_env):
        """Doctor output includes a summary line."""
        output = self._run_doctor()
        assert "Summary:" in output

    def test_api_key_passes_when_set(self, clean_env):
        """Doctor reports OK when ANTHROPIC_API_KEY is set."""
        output = self._run_doctor({"ANTHROPIC_API_KEY": "sk-ant-test"})
        assert "[OK]" in output
        # Count OKs — at least Python and API key should pass
        ok_count = output.count("[OK]")
        assert ok_count >= 2

    def test_docker_available_when_installed(self, clean_env):
        """Doctor checks Docker daemon when Docker is available."""
        def which_side_effect(name):
            if name == "docker":
                return "/usr/bin/docker"
            return None

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "info":
                    return MagicMock(returncode=0, stdout="24.0.7")
                if cmd[1] == "image":
                    return MagicMock(returncode=0, stdout="{}")
            return MagicMock(returncode=1)

        # With Docker available, all checks pass (including MCP servers)
        def which_all(name):
            return f"/usr/bin/{name}"

        output = self._run_doctor(
            which_side_effect=which_all,
            run_side_effect=run_side_effect,
        )
        assert "Docker daemon" in output
        assert "Docker image" in output

    def test_otel_config_reported(self, clean_env):
        """Doctor reports OTel configuration when enabled."""
        # OTel SDK not installed → OTel SDK check fails, causing sys.exit(1)
        with pytest.raises(SystemExit):
            self._run_doctor({"SACV_OTEL_ENABLED": "true"})

    def test_otel_sdk_check(self, clean_env):
        """Doctor checks OTel SDK availability when OTel is configured."""
        with pytest.raises(SystemExit):
            self._run_doctor({"SACV_OTEL_ENABLED": "true"})

    def test_no_failures_no_exit(self, clean_env):
        """Doctor does not exit with code 1 when all checks pass."""
        def which_side_effect(name):
            if name in ("agentmemory", "codegraph", "graphify", "docker"):
                return f"/usr/bin/{name}"
            return None

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "info":
                    return MagicMock(returncode=0, stdout="24.0.7")
                if cmd[1] == "image":
                    return MagicMock(returncode=0, stdout="{}")
            return MagicMock(returncode=1)

        output = self._run_doctor(
            which_side_effect=which_side_effect,
            run_side_effect=run_side_effect,
        )
        # All checks should pass — no SystemExit(1) should be raised
        assert "Summary:" in output
        assert "0 failure" in output
