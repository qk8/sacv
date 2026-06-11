"""
tests/unit/test_actor_untested_paths.py
========================================
Unit tests for untested actor.py code paths.

Tests cover:
1. _format_preflight with passed=False and zero errors (returns "")
2. Branch-not-found fallback — creates fresh branch when existing branch missing
3. Outcome stagnation triggered by critic findings only (no preflight problems)
4. _format_preflight includes repair_suggestions
5. _format_preflight with zero errors, zero arch, zero suggestions (all empty)
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, CRITIC_RESET
from sacv.nodes.actor import make_actor_node, _format_preflight
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _deps(agent=None, git=None, diff=None, config=None, sandbox=None):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=git or StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=diff or StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-actor-fmt-001",
        "session_start_ms": None,
        "task_description": "Add findById to UserService",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.ACTOR.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": "Follow DDD conventions.",
        "strategy_candidates": [],
        "selected_strategy": {"strategy_id": "s1", "description": "Add findById", "affected_files": ["UserService.java"]},
        "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "empty_diff_retries": 0,
        "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
        "check_profile": "standard", "cumulative_cost_dollars": 0.0,
    }
    base.update(kw)
    return base


# ── _format_preflight edge cases ──────────────────────────────────────────────

class TestFormatPreflightEdgeCases:

    def test_passed_false_with_zero_errors_returns_empty(self):
        """passed=False but no errors/arch/suggestions → empty string.

        This can happen when preflight_result has passed=False but all error
        lists are empty (e.g., all checks were non-required).
        """
        result = _format_preflight({
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [],
            "repair_suggestions": [],
        })
        assert result == ""

    def test_includes_repair_suggestions(self):
        """Repair suggestions from preflight are included in feedback."""
        result = _format_preflight({
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [],
            "repair_suggestions": [
                {"category": "compile", "text": "Fix imports: missing symbol UserService"},
                {"category": "architecture", "text": "no-dep: UserService -> UserRepository"},
            ],
        })
        assert "[FIX] compile" in result
        assert "Fix imports" in result
        assert "[FIX] architecture" in result
        assert "no-dep" in result

    def test_limits_lsp_errors_and_arch_violations_independently(self):
        """Both lsp_errors and arch_violations are limited to 5 each."""
        lsp_errors = [{"file": f"L{i}.java", "line": 1, "code": "E", "message": "err"} for i in range(10)]
        arch_violations = [{"rule": f"R{i}", "message": f"violation {i}"} for i in range(10)]
        result = _format_preflight({
            "passed": False,
            "lsp_errors": lsp_errors,
            "arch_violations": arch_violations,
        })
        # First 5 LSP errors
        assert "L0.java" in result
        assert "L4.java" in result
        assert "L5.java" not in result
        # First 5 arch violations
        assert "R0" in result
        assert "R4" in result
        assert "R5" not in result

    def test_only_repair_suggestions_no_errors(self):
        """Only repair_suggestions present, no lsp_errors or arch_violations."""
        result = _format_preflight({
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [],
            "repair_suggestions": [{"category": "blast_radius", "text": "Too many files"}],
        })
        assert "[FIX] blast_radius" in result
        assert "Too many files" in result


# ── Branch-not-found fallback ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestActorBranchNotFound:

    async def test_branch_not_found_creates_fresh_branch(self):
        """When stored branch_name doesn't exist, creates a fresh branch."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+method",
            "operation": "modify", "language": "java",
        }])])
        git = StubGitProvider()
        # Do NOT pre-populate the branch — list_branches will return empty
        deps = _deps(agent=agent, git=git)
        state = _state(
            correction_state={
                "attempt_count": 1,
                "branch_name": "agent-task-actor-fmt-001-a1",  # doesn't exist
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            }
        )
        node = make_actor_node(deps)

        out = await node(state)

        # Should have created a fresh branch
        assert out["diff_proposal"] is not None
        # list_branches called with correction's branch_name
        assert git.calls[0] == ("list_branches", "agent-task-actor-fmt-001-a1")
        # create_branch called because branch wasn't found, uses generated name
        assert git.calls[1] == ("create_branch", "agent-task-task-act-a1", "HEAD")
        # checkout uses the reassigned (generated) branch_name
        assert git.calls[2] == ("checkout", "agent-task-task-act-a1")
    async def test_no_branch_name_creates_branch(self):
        """When branch_name is None (first attempt), creates a new branch."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/X.java",
            "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        git = StubGitProvider()
        deps = _deps(agent=agent, git=git)
        state = _state(
            correction_state={
                "attempt_count": 0, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            }
        )
        node = make_actor_node(deps)

        await node(state)

        # list_branches called first (guard check), then create_branch
        assert git.calls[0] == ("list_branches", "agent-task-task-act-a0")
        assert git.calls[1] == ("create_branch", "agent-task-task-act-a0", "HEAD")
        # checkout uses the same generated branch_name
        assert git.calls[2] == ("checkout", "agent-task-task-act-a0")

    async def test_existing_branch_skips_create(self):
        """When branch_name exists in list_branches, skips create_branch."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java",
            "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        git = StubGitProvider()
        git._branches.add("agent-task-actor-fmt-001-a1")
        deps = _deps(agent=agent, git=git)
        state = _state(
            correction_state={
                "attempt_count": 1,
                "branch_name": "agent-task-actor-fmt-001-a1",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            }
        )
        node = make_actor_node(deps)

        await node(state)

        # list_branches called with correction's branch_name
        assert git.calls[0] == ("list_branches", "agent-task-actor-fmt-001-a1")
        # checkout uses the same (correction's) branch_name, no reassignment
        assert git.calls[1] == ("checkout", "agent-task-actor-fmt-001-a1")
        # No create_branch call
        create_calls = [c for c in git.calls if c[0] == "create_branch"]
        assert len(create_calls) == 0


# ── Outcome stagnation via critic findings ────────────────────────────────────

@pytest.mark.asyncio
class TestActorOutcomeStagnationCritic:
    """Outcome stagnation triggered by critic findings (not preflight)."""

    async def test_outcome_stagnation_via_critic_findings_only(self):
        """When preflight is clean but critical critic findings exist,
        outcome stagnation is checked via critic signature."""
        from sacv.nodes._stagnation import compute_outcome_signature

        # Clean preflight — no problems there
        preflight = {"passed": True, "lsp_errors": [], "arch_violations": []}
        # But critical critic findings exist
        critic_findings = [
            {"critic": "security", "severity": "critical", "rule_id": "SEC-001",
             "file": "X.java", "line": 10, "message": "injection",
             "resolution_hint": "use params"},
        ]
        outcome_sig = compute_outcome_signature(preflight, critic_findings)

        agent = StubAgentProvider([])  # Should NOT be called
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        state = _state(
            preflight_result=preflight,
            critic_findings=critic_findings,
            correction_state={
                "attempt_count": 1,
                "branch_name": "agent-task-task-act-a1",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
        )

        with patch("sacv.nodes.actor.check_outcome_stagnation", return_value=True):
            out = await node(state)

        # Should short-circuit to stagnation without calling agent
        assert out["verifier_verdict"]["test_result"] == "FAIL"
        assert out["verifier_verdict"]["diagnostic"] == "STAGNATION"
        assert out["diff_proposal"] is None
        assert len(agent.calls) == 0

    async def test_no_stagnation_when_no_problems(self):
        """When preflight is clean AND no critical critic findings,
        outcome stagnation check is skipped entirely."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java",
            "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        # Clean preflight + no critical findings
        state = _state(
            preflight_result={"passed": True},
            critic_findings=[
                {"critic": "style", "severity": "info", "rule_id": "STY-1",
                 "file": "X.java", "line": 1, "message": "naming",
                 "resolution_hint": "rename"},
            ],
            correction_state={
                "attempt_count": 0, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
        )

        out = await node(state)

        # Should proceed normally (agent called, diff produced)
        assert out["diff_proposal"] is not None
        assert out.get("verifier_verdict") is None  # no synthetic verdict
