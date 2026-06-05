"""
tests/integration/test_intelligent_debugger_delta_debug.py
============================================================
Integration tests for IntelligentDebuggerNode delta-debug path.

Tests cover:
1. Delta debug binary search on request payload
2. Delta debug payload extraction from failure message
3. Delta debug with no extractable payload
4. Delta debug endpoint extraction
5. Delta debug with shallow payload (1-2 fields)
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, VerifierVerdict, DiagnosticVerdict
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.sandbox_provider import ExecResult


def _deps(sandbox=None, agent=None, config=None):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider([
            make_json_agent_result("The request payload contains invalid fields.")
        ]),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(
            default_exit_code=1,
            default_stdout="validation failed",
        ),
        diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _validation_state(payload_in_message=True):
    failure_msg = (
        'java.lang.ConstraintViolationException: '
        '{"email":"test@example.com","name":"","age":25,"role":"admin"}\n'
        '\tat com.example.service.UserValidator.validate(UserValidator.java:15)'
    ) if payload_in_message else "ConstraintViolationException: validation failed"

    verdict: VerifierVerdict = {
        "test_result": "FAIL",
        "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
        "phase1_passed": True, "phase2_passed": False,
        "test_failures": [{"message": failure_msg, "source": "junit"}],
        "performance_delta": None, "visual_diff_result": None,
        "docker_exit_code": 1,
    }
    return {
        "session_id": "t", "task_id": "task-dbg-dd-001",
        "task_description": "Add user via POST /api/v1/users",
        "project_mode": "greenfield", "module_type": "backend-api",
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
class TestIntelligentDebuggerDeltaDebug:

    async def test_delta_debug_executed_for_validation_error(self):
        """Validation error triggers delta debug strategy."""
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(1, "error", "", 10)

        out = await make_intelligent_debugger_node(_deps(sandbox=_TrackedSandbox()))(
            _validation_state()
        )

        # At least one curl/test call for delta debug
        curl_calls = [c for c in call_log if "curl" in c.lower()]
        assert len(curl_calls) >= 1

    async def test_delta_debug_minimal_payload_extracted(self):
        """Delta debug finds minimal failing subset of request payload."""
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                # First call fails, subsequent calls fail too
                return ExecResult(1, "error", "", 10)

        out = await make_intelligent_debugger_node(_deps(sandbox=_TrackedSandbox()))(
            _validation_state()
        )

        obs = out["debug_observations"]
        assert obs["minimal_payload"] is not None
        # Minimal payload should be a subset of the original fields
        assert isinstance(obs["minimal_payload"], dict)

    async def test_delta_debug_no_payload_returns_empty(self):
        """Delta debug returns empty observations when no payload can be extracted."""
        verdict: VerifierVerdict = {
            "test_result": "FAIL",
            "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
            "phase1_passed": True, "phase2_passed": False,
            "test_failures": [{"message": "ConstraintViolationException: validation failed",
                               "source": "junit"}],
            "performance_delta": None, "visual_diff_result": None,
            "docker_exit_code": 1,
        }
        state = {
            "session_id": "t", "task_id": "task-dbg-dd-002",
            "task_description": "Add user via POST /api/v1/users",
            "project_mode": "greenfield", "module_type": "backend-api",
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

        out = await make_intelligent_debugger_node(_deps())(state)
        obs = out["debug_observations"]
        assert obs["minimal_payload"] is None

    async def test_delta_debug_endpoint_extracted_from_task(self):
        """Delta debug uses endpoint from task description."""
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(1, "error", "", 10)

        state = _validation_state()
        state["task_description"] = "Implement POST /api/v2/orders endpoint"

        await make_intelligent_debugger_node(_deps(sandbox=_TrackedSandbox()))(state)

        # curl calls should target the extracted endpoint
        curl_calls = [c for c in call_log if "curl" in c.lower()]
        assert len(curl_calls) >= 1
        # The endpoint should be in the curl command
        any_call_has_endpoint = any("/api/v2/orders" in c for c in curl_calls)
        assert any_call_has_endpoint

    async def test_delta_debug_with_single_field_payload(self):
        """Delta debug handles single-field payloads correctly."""
        verdict: VerifierVerdict = {
            "test_result": "FAIL",
            "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
            "phase1_passed": True, "phase2_passed": False,
            "test_failures": [{"message": 'ConstraintViolationException: {"name": ""}', "source": "junit"}],
            "performance_delta": None, "visual_diff_result": None,
            "docker_exit_code": 1,
        }
        state = {
            "session_id": "t", "task_id": "task-dbg-dd-003",
            "task_description": "Add user via POST /api/v1/users",
            "project_mode": "greenfield", "module_type": "backend-api",
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

        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, cmd, env=None, timeout=120):
                call_log.append(cmd)
                return ExecResult(1, "error", "", 10)

        out = await make_intelligent_debugger_node(_deps(sandbox=_TrackedSandbox()))(state)
        obs = out["debug_observations"]
        # Should still produce a minimal payload (even if single field)
        assert obs["minimal_payload"] is not None

    async def test_delta_debug_deeply_nested_payload(self):
        """Delta debug handles deeply nested JSON payloads."""
        verdict: VerifierVerdict = {
            "test_result": "FAIL",
            "diagnostic": DiagnosticVerdict.AMBIGUOUS.value,
            "phase1_passed": True, "phase2_passed": False,
            "test_failures": [{
                "message": 'ConstraintViolationException: {"user":{"name":"","email":"x"},"settings":{"theme":"dark"}}',
                "source": "junit"
            }],
            "performance_delta": None, "visual_diff_result": None,
            "docker_exit_code": 1,
        }
        state = {
            "session_id": "t", "task_id": "task-dbg-dd-004",
            "task_description": "Update user via POST /api/v1/users",
            "project_mode": "greenfield", "module_type": "backend-api",
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

        out = await make_intelligent_debugger_node(_deps())(state)
        obs = out["debug_observations"]
        assert obs["error_type"] == "VALIDATION_ERROR"

    async def test_delta_debug_root_cause_synthesized(self):
        """Root cause hypothesis mentions the validation issue."""
        out = await make_intelligent_debugger_node(_deps())(_validation_state())
        obs = out["debug_observations"]
        assert obs["root_cause"] is not None
        assert len(obs["root_cause"]) > 0
        assert "validation" in obs["root_cause"].lower() or "payload" in obs["root_cause"].lower()
