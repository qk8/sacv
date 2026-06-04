"""
tests/integration/test_replan.py
=================================
Integration tests for ReplanNode.
Uses stubs — no Docker, no live LLM, no network.
"""
from __future__ import annotations

import pytest
from sacv.nodes.replan import make_replan_node
from sacv.orchestration.deps import NodeDeps
from sacv.orchestration.config import WorkflowConfig
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _deps():
    agent = StubAgentProvider([
        make_json_agent_result([{
            "strategy_id": "r1",
            "description": "Retry approach",
            "affected_files": ["A.java"],
            "avoids": "avoids the previous error",
        }])
    ])
    return NodeDeps(
        agent=agent,
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "t", "task_description": "test",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": "verifier",
        "tdd_gate_attempts": 3,  # at max — replan must reset this
        "replan_count": 0,
        "strategy_candidates": [{"strategy_id": "s1"}],
        "exhausted_branches": ["agent-task-abc-s1"],
        "verifier_verdict": {"diagnostic": "FIX_IMPL", "test_failures": []},
        "critic_findings": [], "preflight_result": None,
        "correction_state": {
            "attempt_count": 3, "branch_name": "b",
            "last_error_hash": "abc", "error_history": [],
            "stagnation_pattern": "none",
        },
        "context_skeleton": None, "blast_radius_map": None,
        "agents_md_context": None, "test_inventory_paths": [],
        "diff_proposal": None, "debug_observations": None,
        "confidence_score": 1.0, "active_branches": [],
        "speculative_stash_ref": None, "escalation_payload": None,
        "procedural_constraints": [], "lesson_learned": None,
        "arch_rules_updated": False,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replan_resets_tdd_gate_attempts():
    """replan must reset tdd_gate_attempts to allow TDD gate to run again."""
    out = await make_replan_node(_deps())(_state())
    assert out["tdd_gate_attempts"] == 0, (
        "replan must reset tdd_gate_attempts to allow TDD gate to run again"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replan_resets_correction_state():
    """replan resets attempt_count, stagnation, and error history."""
    out = await make_replan_node(_deps())(_state())
    cs = out["correction_state"]
    assert cs["attempt_count"] == 0
    assert cs["stagnation_pattern"] == "none"
    assert cs["error_history"] == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replan_resets_active_branches():
    """replan clears active_branches for a fresh cycle."""
    out = await make_replan_node(_deps())(_state())
    assert out["active_branches"] == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replan_increments_replan_count():
    """replan increments replan_count."""
    s = _state(replan_count=2)
    out = await make_replan_node(_deps())(s)
    assert out["replan_count"] == 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replan_clears_verdict_and_preflight():
    """replan clears stale verifier_verdict and preflight_result."""
    s = _state(
        verifier_verdict={"test_result": "FAIL"},
        preflight_result={"passed": False},
    )
    out = await make_replan_node(_deps())(s)
    assert out["verifier_verdict"] is None
    assert out["preflight_result"] is None
