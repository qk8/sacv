"""
tests/integration/test_intelligent_debugger.py
===============================================
Integration tests for IntelligentDebuggerNode.
Uses stubs — no Docker, no live JVM, no network.
"""
from __future__ import annotations
import json
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node
from sacv.orchestration.config import WorkflowConfig, DebugConfig
from sacv.orchestration.state import WorkflowPhase, VerifierVerdict, DiagnosticVerdict
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.sandbox_provider import ExecResult


def _mock_jdwp():
    """Create a mock JdwpClient that simulates a breakpoint hit."""
    mock_hit = MagicMock()
    mock_hit.file = "UserService.java"
    mock_hit.line = 42
    mock_hit.method_name = "findById"
    mock_hit.class_name = "com.example.service.UserService"
    mock_hit.thread_name = "main"

    # Use plain objects with attributes for JSON serialization compatibility
    class _Var:
        def __init__(self, name, value, type_):
            self.name = name
            self.value = value
            self.type = type_

    mock_vars = [
        _Var("user", "null", "User"),
        _Var("id", "1", "Long"),
    ]

    mock_client = AsyncMock()
    mock_client.set_breakpoint_at_line = AsyncMock()
    mock_client.run = AsyncMock()
    mock_client.wait_for_breakpoint_hit = AsyncMock(return_value=mock_hit)
    mock_client.get_local_variables = AsyncMock(return_value=mock_vars)
    mock_client.get_call_stack = AsyncMock(return_value=[
        "com.example.service.UserService.findById(UserService.java:42)",
    ])
    mock_client.step_over = AsyncMock(return_value=None)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _deps(sandbox=None, agent=None, config=None):
    from sacv.orchestration.graph import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider([
            make_json_agent_result("The variable `user` is null at line 42.")
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


def _state(
    module="backend-domain",
    failure_msg=(
        "java.lang.NullPointerException: Cannot invoke "
        '"com.example.User.getId()"\n'
        "\tat com.example.service.UserService.findById(UserService.java:42)"
    ),
):
    verdict: VerifierVerdict = {
        "test_result": "FAIL",
        "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
        "phase1_passed": True, "phase2_passed": False,
        "test_failures": [{"message": failure_msg, "source": "junit"}],
        "performance_delta": None, "visual_diff_result": None,
        "critic_findings": [], "docker_exit_code": 1,
    }
    return {
        "session_id": "t", "task_id": "task-dbg-001",
        "task_description": "Add findById to UserService",
        "project_mode": "greenfield", "module_type": module,
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
class TestIntelligentDebugger:

    async def test_returns_debug_observations(self):
        """Node must always return debug_observations dict."""
        out = await make_intelligent_debugger_node(_deps())(_state())
        assert "debug_observations" in out
        assert out["debug_observations"] is not None

    async def test_reports_intelligent_debugger_phase(self):
        """During debug session, current_phase must be INTELLIGENT_DEBUGGER."""
        out = await make_intelligent_debugger_node(_deps())(_state())
        assert out["current_phase"] == WorkflowPhase.INTELLIGENT_DEBUGGER.value

    async def test_error_type_classified(self):
        """NPE error text → NULL_REFERENCE classification."""
        out = await make_intelligent_debugger_node(_deps())(_state())
        obs = out["debug_observations"]
        assert obs["error_type"] == "NULL_REFERENCE"

    async def test_pruned_stack_populated(self):
        """Pruned stack must contain user-code frames (not framework lines)."""
        out = await make_intelligent_debugger_node(_deps())(_state())
        stack = out["debug_observations"]["pruned_stack"]
        assert isinstance(stack, list)
        # All frames must reference user code (not Spring/Hibernate)
        for frame in stack:
            assert "springframework" not in frame.get("method", "").lower()

    async def test_bean_error_triggers_actuator_query(self):
        """BeanCreationException → sandbox exec for /actuator/beans."""
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(0, '{"contexts": {"beans": []}}', "", 10)

        deps = _deps(sandbox=_TrackedSandbox())
        failure = (
            "org.springframework.beans.factory.BeanCreationException: "
            "Error creating bean 'paymentService'\n"
            "\tat com.example.payment.PaymentService.<init>(PaymentService.java:18)"
        )
        state = _state(failure_msg=failure)
        await make_intelligent_debugger_node(deps)(state)

        actuator_calls = [c for c in call_log if "actuator" in c.lower()]
        assert len(actuator_calls) >= 1

    async def test_validation_error_triggers_delta_debug(self):
        """Validation error with payload → delta debug executed."""
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(1, "validation failed", "", 10)

        deps  = _deps(sandbox=_TrackedSandbox())
        failure = (
            'java.lang.ConstraintViolationException: {"email": null, "name": ""}\n'
            '\tat com.example.service.UserValidator.validate(UserValidator.java:15)'
        )
        state = _state(failure_msg=failure)
        # Include an API endpoint in task_description so _extract_endpoint works
        state["task_description"] = "Add user via POST /api/v1/users"
        await make_intelligent_debugger_node(deps)(state)
        # At least one curl/test call for delta debug
        assert len(call_log) >= 1

    async def test_one_llm_call_for_hypothesis(self):
        """Node makes exactly one LLM call (root-cause hypothesis)."""
        agent = StubAgentProvider([
            make_json_agent_result("The user object is null.")
        ])
        deps = _deps(agent=agent)
        with patch(
            "sacv.adapters.debug.jdwp_client.JdwpClient",
            return_value=_mock_jdwp(),
        ):
            await make_intelligent_debugger_node(deps)(_state())
        assert len(agent.calls) == 1
        assert agent.calls[0][0] == "debug_analyst"

    async def test_root_cause_in_observations(self):
        """Hypothesis from LLM must appear in debug_observations.root_cause."""
        agent = StubAgentProvider([
            make_json_agent_result("The user is null because lazy loading failed.")
        ])
        with patch(
            "sacv.adapters.debug.jdwp_client.JdwpClient",
            return_value=_mock_jdwp(),
        ):
            out = await make_intelligent_debugger_node(_deps(agent=agent))(_state())
        assert "null" in out["debug_observations"]["root_cause"].lower()

    async def test_no_proposal_skips_strategy(self):
        """No diff_proposal → observations still returned (graceful)."""
        state = _state()
        state["verifier_verdict"]["test_failures"] = []
        out = await make_intelligent_debugger_node(_deps())(state)
        assert out["debug_observations"] is not None

    async def test_typescript_error_classified(self):
        """TypeScript NPE classified correctly."""
        state = _state(
            module="frontend-feature",
            failure_msg="TypeError: Cannot read properties of null (reading 'id')"
        )
        out = await make_intelligent_debugger_node(_deps())(state)
        assert out["debug_observations"]["error_type"] == "NULL_REFERENCE"

    async def test_frontend_does_not_use_jdwp(self):
        """Frontend errors must not trigger JDWP debug session (CDP only)."""
        jdb_launched = False

        class _WatchSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                nonlocal jdb_launched
                if "jdwp" in cmd.lower() or "jdb" in cmd.lower():
                    jdb_launched = True
                return ExecResult(0, "", "", 10)

        state = _state(
            module="frontend-feature",
            failure_msg="TypeError: Cannot read properties of undefined (reading 'balance')"
        )
        await make_intelligent_debugger_node(_deps(sandbox=_WatchSandbox()))(state)
        assert not jdb_launched, "JDWP must not be used for TypeScript errors"
