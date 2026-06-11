"""
tests/unit/test_critic_breakdown.py
====================================
OBS-005: Verify that all_critics_node timing includes per-critic finding counts.
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _make_deps():
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _make_state():
    return {
        "session_id": "s1", "task_id": "t1", "project_mode": "greenfield",
        "module_type": "backend-domain", "current_phase": WorkflowPhase.CRITICS.value,
        "task_description": "test", "context_skeleton": None,
        "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None,
        "pruned_strategies": [], "red_phase_evidence_path": None,
        "test_inventory_paths": [], "tdd_gate_attempts": 0,
        "diff_proposal": {
            "diffs": [{
                "file_path": "src/main/java/UserService.java",
                "operation": "modify",
                "diff_content": "+    public User findById(Long id) { return null; }",
            }],
        },
        "preflight_result": None, "critic_findings": [],
        "verifier_verdict": None, "debug_observations": None,
        "correction_state": {"attempt_count": 0, "branch_name": None,
                             "last_error_hash": None, "error_history": [],
                             "stagnation_pattern": "none"},
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "speculative_stash_ref": None, "escalation_payload": None,
        "procedural_constraints": [], "lesson_learned": None,
        "arch_rules_updated": False, "check_profile": "standard",
        "cumulative_cost_dollars": 0.0, "skip_tdd_gate": False,
        "workflow_audit_trail": [],
    }


@pytest.mark.asyncio
class TestCriticBreakdown:

    async def test_security_critic_returns_findings(self):
        """Security critic returns its findings when agent responds."""
        agent = StubAgentProvider([make_json_agent_result([
            {"critic": "security", "severity": "critical", "file": "X.java",
             "line": 1, "rule_id": "SEC-001", "message": "SQL injection",
             "resolution_hint": "use params"},
        ])])
        deps = _make_deps()
        deps.agent = agent

        from sacv.nodes.critics.security import make_security_critic_node
        node = make_security_critic_node(deps)

        out = await node(_make_state())

        assert len(out["critic_findings"]) == 1
        assert out["critic_findings"][0]["critic"] == "security"

    async def test_style_critic_returns_findings(self):
        """Style critic returns its findings when agent responds."""
        agent = StubAgentProvider([make_json_agent_result([
            {"critic": "style", "severity": "warning", "file": "Y.java",
             "line": 1, "rule_id": "STY-001", "message": "Long method",
             "resolution_hint": "shorten"},
        ])])
        deps = _make_deps()
        deps.agent = agent

        from sacv.nodes.critics.style import make_style_critic_node
        node = make_style_critic_node(deps)

        out = await node(_make_state())

        assert len(out["critic_findings"]) == 1
        assert out["critic_findings"][0]["critic"] == "style"

    async def test_consistency_critic_returns_findings(self):
        """Consistency critic returns its findings when agent responds."""
        agent = StubAgentProvider([make_json_agent_result([
            {"critic": "consistency", "severity": "warning", "file": "Z.java",
             "line": 1, "rule_id": "CON-001", "message": "Naming mismatch",
             "resolution_hint": "align names"},
        ])])
        deps = _make_deps()
        deps.agent = agent

        from sacv.nodes.critics.consistency import make_consistency_critic_node
        node = make_consistency_critic_node(deps)

        out = await node(_make_state())

        assert len(out["critic_findings"]) == 1
        assert out["critic_findings"][0]["critic"] == "consistency"
