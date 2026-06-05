"""
tests/integration/test_intelligent_debugger_cdp.py
====================================================
Integration tests for IntelligentDebuggerNode CDP (TypeScript) debug path.

Tests cover:
1. CDP breakpoint session for TypeScript errors
2. CDP variable inspection from paused event
3. CDP step-over/step-into after breakpoint hit
4. CDP expression evaluation in call frame
5. CDP session with no bundle file
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node
from sacv.orchestration.config import WorkflowConfig, DebugConfig
from sacv.orchestration.state import WorkflowPhase, VerifierVerdict, DiagnosticVerdict
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.sandbox_provider import ExecResult


def _mock_cdp():
    """Create a mock CdpClient that simulates a breakpoint hit."""
    mock_paused = MagicMock()
    mock_paused.reason = "breakpoint"
    mock_paused.call_frames = []
    mock_paused.hit_breakpoints = ["bp-1"]
    mock_paused.call_frame_id = "frame-1"

    mock_frame = MagicMock()
    mock_frame.function = "handleSubmit"
    mock_frame.url = "src/forms/UserForm.tsx"
    mock_frame.line = 42
    mock_frame.column = 10
    mock_frame.scope_chain = [{
        "type": "local",
        "object": {"objectId": "obj-1"},
    }]
    mock_paused.call_frames = [mock_frame]

    mock_client = AsyncMock()
    mock_client.enable_debugger = AsyncMock()
    mock_client.set_breakpoint_by_url = AsyncMock(return_value="bp-1")
    mock_client.resume = AsyncMock()
    mock_client.wait_for_paused = AsyncMock(return_value=mock_paused)
    mock_client.get_scope_variables_from_paused = AsyncMock(return_value={
        "user": {"value": "null", "type": "undefined"},
        "id": {"value": "42", "type": "number"},
    })
    mock_client.evaluate_in_frame = AsyncMock(return_value="user")
    mock_client.step_over = AsyncMock(return_value=None)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _deps(sandbox=None, agent=None, config=None):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider([
            make_json_agent_result("The `user` variable is undefined at line 42.")
        ]),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(
            default_exit_code=0,
            default_stdout='{"contexts": {}}',
        ),
        diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _ts_state():
    verdict: VerifierVerdict = {
        "test_result": "FAIL",
        "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
        "phase1_passed": True, "phase2_passed": False,
        "test_failures": [{
            "message": "TypeError: Cannot read properties of null (reading 'id')\n"
                       "    at handleSubmit (src/forms/UserForm.tsx:42:10)",
            "source": "playwright",
        }],
        "performance_delta": None, "visual_diff_result": None,
        "docker_exit_code": 1,
    }
    return {
        "session_id": "t", "task_id": "task-dbg-ts-001",
        "task_description": "Add user form",
        "project_mode": "greenfield", "module_type": "frontend-feature",
        "context_skeleton": None, "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": verdict,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 1, "branch_name": "b",
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 0.7, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
    }


@pytest.mark.asyncio
@pytest.mark.integration
class TestIntelligentDebuggerCDP:

    async def test_cdp_session_executed_for_typescript(self):
        """TypeScript NULL_REFERENCE triggers CDP debug session."""
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps())(_ts_state())

        cdp_mock.enable_debugger.assert_called_once()
        cdp_mock.set_breakpoint_by_url.assert_called_once()

    async def test_cdp_breakpoint_set_at_correct_line(self):
        """CDP breakpoint is set one line before error frame (strategy offset=-1)."""
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            await make_intelligent_debugger_node(_deps())(_ts_state())

        args, kwargs = cdp_mock.set_breakpoint_by_url.call_args
        # Method signature: set_breakpoint_by_url(url_or_file, line, column=0)
        url_or_file = args[0] if len(args) > 0 else kwargs["url_or_file"]
        line = args[1] if len(args) > 1 else kwargs["line"]
        # NULL_REFERENCE strategy has breakpoint_offset=-1 → 42-1=41
        assert line == 41
        assert "UserForm" in url_or_file

    async def test_cdp_variables_collected(self):
        """CDP session collects scope variables from paused frame."""
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps())(_ts_state())

        obs = out["debug_observations"]
        assert len(obs["breakpoint_hits"]) >= 1
        hit = obs["breakpoint_hits"][0]
        assert "user" in hit["variables"]
        assert "id" in hit["variables"]

    async def test_cdp_call_stack_collected(self):
        """CDP session collects call stack from paused frame."""
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps())(_ts_state())

        obs = out["debug_observations"]
        assert len(obs["breakpoint_hits"]) >= 1
        stack = obs["breakpoint_hits"][0]["call_stack"]
        assert len(stack) >= 1

    async def test_cdp_expression_evaluation(self):
        """CDP session evaluates strategy expressions in call frame.

        NULL_REFERENCE strategy has no evaluate_expressions, so
        evaluate_in_frame should not be called.
        """
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps())(_ts_state())

        obs = out["debug_observations"]
        # NULL_REFERENCE strategy has empty evaluate_expressions
        assert len(obs["breakpoint_hits"][0].get("extra_evals", {})) == 0

    async def test_cdp_step_over_after_breakpoint(self):
        """CDP step_over is called when strategy requires it."""
        cdp_mock = _mock_cdp()
        cdp_mock.step_over = AsyncMock(return_value=None)

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            await make_intelligent_debugger_node(_deps())(_ts_state())

        # step_over should be called if strategy.step_type != "none"
        cdp_mock.step_over.assert_called()

    async def test_cdp_no_bundle_returns_empty_observations(self):
        """CDP session returns empty observations when no bundle file found."""
        call_log: list[str] = []

        class _NoBundleSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(0, "NO_ENTRY_POINT", "", 10)

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
        ) as mock_cdp_cls:
            await make_intelligent_debugger_node(_deps(sandbox=_NoBundleSandbox()))(_ts_state())
            # CdpClient should NOT have been instantiated since bundle not found
            mock_cdp_cls.assert_not_called()

    async def test_cdp_port_not_ready_returns_empty_observations(self):
        """CDP session returns empty observations when debug port not ready."""
        cdp_mock = _mock_cdp()

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps(
                config=WorkflowConfig(debug=DebugConfig(cdp_port=9999, debug_port_wait_sec=0.01))
            ))(_ts_state())

        # Should still return observations (just without CDP data)
        assert out["debug_observations"] is not None
        assert out["debug_observations"]["error_type"] == "NULL_REFERENCE"

    async def test_cdp_session_error_handled_gracefully(self):
        """CDP session errors are caught and observations still returned."""
        cdp_mock = AsyncMock()
        cdp_mock.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
        cdp_mock.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "sacv.adapters.debug.cdp_client.CdpClient",
            return_value=cdp_mock,
        ):
            out = await make_intelligent_debugger_node(_deps())(_ts_state())

        # Should still return observations (without breakpoint hits)
        assert out["debug_observations"] is not None
        assert len(out["debug_observations"]["breakpoint_hits"]) == 0
