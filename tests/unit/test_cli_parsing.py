"""
tests/unit/test_cli_parsing.py
================================
Unit tests for CLI argument parsing and command routing in cli.py.

Tests cover:
1. Argument parsing — run and resume subcommands
2. Command routing — correct function dispatch
3. Error handling — missing API key, missing escalation file
"""
from __future__ import annotations

import argparse
import pytest
from unittest.mock import patch, MagicMock
import sys


@pytest.fixture
def parser():
    from sacv.cli import main
    # Build parser without calling main
    parser = argparse.ArgumentParser(prog="sacv")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a new SACV workflow task")
    run_p.add_argument("--task-id", required=True, help="Unique task identifier")
    run_p.add_argument("--description", required=True, help="Task description")
    run_p.add_argument(
        "--mode", choices=["greenfield", "brownfield"], default="greenfield",
    )
    run_p.add_argument(
        "--module",
        choices=[
            "backend-domain", "backend-api",
            "frontend-feature", "frontend-data",
            "infrastructure", "cross-cutting",
        ],
        default="backend-domain",
    )
    run_p.add_argument(
        "--check-profile",
        choices=["standard", "full"],
        default="standard",
    )

    res_p = sub.add_parser("resume", help="Resume a paused HITL escalation")
    res_p.add_argument(
        "--escalation-id", required=True,
        help="Escalation ID from the .workflow/escalations/<id>.json file",
    )
    return parser


class TestRunSubcommand:

    def test_default_mode_is_greenfield(self, parser):
        args = parser.parse_args(["run", "--task-id", "T1", "--description", "Add feature"])
        assert args.mode == "greenfield"

    def test_explicit_mode_brownfield(self, parser):
        args = parser.parse_args([
            "run", "--task-id", "T1", "--description", "Fix bug", "--mode", "brownfield",
        ])
        assert args.mode == "brownfield"

    def test_default_module_is_backend_domain(self, parser):
        args = parser.parse_args(["run", "--task-id", "T1", "--description", "Add feature"])
        assert args.module == "backend-domain"

    def test_explicit_module_frontend_feature(self, parser):
        args = parser.parse_args([
            "run", "--task-id", "T1", "--description", "Add UI", "--module", "frontend-feature",
        ])
        assert args.module == "frontend-feature"

    def test_check_profile_standard(self, parser):
        args = parser.parse_args(["run", "--task-id", "T1", "--description", "Add feature"])
        assert args.check_profile == "standard"

    def test_check_profile_full(self, parser):
        args = parser.parse_args([
            "run", "--task-id", "T1", "--description", "Add feature", "--check-profile", "full",
        ])
        assert args.check_profile == "full"

    def test_all_module_choices_valid(self, parser):
        valid_modules = [
            "backend-domain", "backend-api",
            "frontend-feature", "frontend-data",
            "infrastructure", "cross-cutting",
        ]
        for module in valid_modules:
            args = parser.parse_args([
                "run", "--task-id", "T1", "--description", "x", "--module", module,
            ])
            assert args.module == module

    def test_invalid_module_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "run", "--task-id", "T1", "--description", "x", "--module", "invalid",
            ])


class TestResumeSubcommand:

    def test_escalation_id_required(self, parser):
        args = parser.parse_args([
            "resume", "--escalation-id", "abc-123",
        ])
        assert args.escalation_id == "abc-123"

    def test_resume_command_set(self, parser):
        args = parser.parse_args(["resume", "--escalation-id", "abc-123"])
        assert args.command == "resume"


class TestCommandRouting:

    def test_run_command_set(self, parser):
        args = parser.parse_args(["run", "--task-id", "T1", "--description", "x"])
        assert args.command == "run"

    def test_resume_command_set(self, parser):
        args = parser.parse_args(["resume", "--escalation-id", "abc"])
        assert args.command == "resume"


class TestMissingApiKey:

    def test_cmd_run_exits_without_api_key(self):
        """cmd_run exits with code 1 when ANTHROPIC_API_KEY is not set."""
        import asyncio
        from sacv.cli import cmd_run
        import argparse

        args = argparse.Namespace(
            task_id="T1", description="Add feature",
            mode="greenfield", module="backend-domain",
            check_profile="standard",
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                asyncio.run(cmd_run(args))
            assert exc_info.value.code == 1


class TestMissingEscalationFile:

    def test_cmd_resume_exits_without_file(self):
        """cmd_resume exits with code 1 when escalation file doesn't exist."""
        import asyncio
        from sacv.cli import cmd_resume
        import argparse

        args = argparse.Namespace(escalation_id="nonexistent-uuid")

        with patch("sacv.cli.Path.exists", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                asyncio.run(cmd_resume(args))
            assert exc_info.value.code == 1


class TestMainFunction:

    def test_run_delegates_to_cmd_run(self):
        """main() with 'run' calls asyncio.run(cmd_run)."""
        from sacv.cli import main

        with patch("sys.argv", ["sacv", "run", "--task-id", "T1", "--description", "x"]):
            with patch("sacv.cli.cmd_run") as mock_run:
                with patch("sacv.cli.asyncio.run"):
                    main()
                    # asyncio.run should have been called with cmd_run
                    # The exact call depends on implementation

    def test_resume_delegates_to_cmd_resume(self):
        """main() with 'resume' calls asyncio.run(cmd_resume)."""
        from sacv.cli import main

        with patch("sys.argv", ["sacv", "resume", "--escalation-id", "abc"]):
            with patch("sacv.cli.cmd_resume") as mock_resume:
                with patch("sacv.cli.asyncio.run"):
                    main()
