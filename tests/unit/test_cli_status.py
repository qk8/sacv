"""
tests/unit/test_cli_status.py
==============================
DBG-001: Verify sacv status and sacv dump-state CLI commands exist
and produce expected output formats.
"""
from __future__ import annotations

import json
import subprocess
import sys


class TestCliStatusCommand:

    def test_status_subcommand_exists(self):
        """sacv status should be a valid subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "status", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"status --help failed: {result.stderr}"
        assert "task-id" in result.stdout.lower() or "task_id" in result.stdout.lower()

    def test_dump_state_subcommand_exists(self):
        """sacv dump-state should be a valid subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "dump-state", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"dump-state --help failed: {result.stderr}"
        assert "task-id" in result.stdout.lower() or "task_id" in result.stdout.lower()

    def test_status_requires_task_id(self):
        """sacv status without --task-id should fail with error."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "status"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_dump_state_requires_task_id(self):
        """sacv dump-state without --task-id should fail with error."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "dump-state"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_status_no_db_shows_error(self, tmp_path):
        """sacv status with no workflow DB should show a clear error."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "status", "--task-id", "nonexistent"],
            capture_output=True, text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode != 0
        assert "no workflow database" in result.stderr.lower() or \
               "no state found" in result.stderr.lower() or \
               "error" in result.stderr.lower()

    def test_dump_state_no_db_shows_error(self, tmp_path):
        """sacv dump-state with no workflow DB should show a clear error."""
        result = subprocess.run(
            [sys.executable, "-m", "sacv.cli", "dump-state", "--task-id", "nonexistent"],
            capture_output=True, text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode != 0
        assert "no workflow database" in result.stderr.lower() or \
               "no state found" in result.stderr.lower() or \
               "error" in result.stderr.lower()
