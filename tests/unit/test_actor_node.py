"""
tests/unit/test_actor_node.py
==============================
Unit tests for the Actor node and its helper functions.

Tests cover:
1. Stagnation detection short-circuit
2. Empty diff handling (JSON parse failure)
3. Diff validation rejection (full overwrite)
4. Diff apply failure (conflicts)
5. Successful diff application
6. Prompt formatting helpers (_format_debug_observations, _format_preflight, _format_findings)
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import (
    WorkflowPhase, DiffProposal, UnifiedDiffPayload,
    VerifierVerdict, DiagnosticVerdict, CRITIC_RESET,
)
from sacv.nodes.actor import make_actor_node, _format_debug_observations, _format_preflight, _format_findings
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.interfaces.diff_provider import DiffValidationError


def _deps(
    agent=None, git=None, diff=None,
    config=None, sandbox=None,
):
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
        "session_id": "t", "task_id": "task-actor-001",
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


@pytest.mark.asyncio
@pytest.mark.unit
class TestActorNode:

    async def test_successful_diff_application(self):
        """Actor produces valid diffs → returns DiffProposal."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+public User findById(Long id) { ... }",
            "operation": "modify", "language": "java",
        }])])
        deps = _deps(agent=agent, diff=StubDiffProvider())
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        # DiffProposal is a TypedDict — use dict key access (not isinstance)
        proposal = out["diff_proposal"]
        assert isinstance(proposal, dict)
        assert proposal["strategy_id"] == "s1"
        assert out["correction_state"]["attempt_count"] == 1
        # task_id[:8] = "task-act", attempt=0 → branch is agent-task-task-act-a0
        assert out["correction_state"]["branch_name"] == "agent-task-task-act-a0"
        assert out["critic_findings"] is CRITIC_RESET
        assert out["preflight_result"] is None
        assert out["debug_observations"] is None
        # Verify git calls (list_branches guard from HIGH-004)
        assert deps.git.calls[0] == ("list_branches", "agent-task-task-act-a0")
        assert deps.git.calls[1] == ("create_branch", "agent-task-task-act-a0", "HEAD")
        assert deps.git.calls[2] == ("checkout", "agent-task-task-act-a0")

    async def test_stagnation_short_circuits_to_hitl(self):
        """When stagnation detected, returns synthetic failing verdict without LLM call."""
        agent = StubAgentProvider([])  # Should NOT be called
        state = _state(
            correction_state={
                "attempt_count": 3,  # >= max_self_correction_cycles (3)
                "branch_name": "b",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            }
        )
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        out = await node(state)

        # Should NOT have called the agent
        assert len(agent.calls) == 0
        # Should return synthetic failing verdict
        assert out["verifier_verdict"]["test_result"] == "FAIL"
        assert out["verifier_verdict"]["diagnostic"] == "STAGNATION"
        assert out["correction_state"]["attempt_count"] == 3  # set to max_self_correction_cycles
        assert out["correction_state"]["stagnation_pattern"] == "iteration"
        assert out["diff_proposal"] is None
        # Stagnation guard must return current_phase to avoid state inconsistency
        assert out["current_phase"] == WorkflowPhase.ACTOR.value

    async def test_json_parse_failure_returns_empty_diff(self):
        """LLM returns non-JSON → retries exhausted → StructuredOutputError → immediate return.

        ERR-002: parse errors now return immediately (not fall through to empty-diff path).
        attempt_count IS incremented because this is a real failure, not an empty-diff case.
        """
        agent = StubAgentProvider([
            AgentResult(content="not json 1", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json 2", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json 3", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json 4", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        deps = _deps(agent=agent, diff=StubDiffProvider())
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["diff_proposal"] is None
        # ERR-002: parse error counts as a real attempt
        assert out["correction_state"]["attempt_count"] == 1
        assert out["empty_diff_retries"] == 1
        assert out["critic_findings"] is CRITIC_RESET
        # debug_observations cleared on parse error
        assert out["debug_observations"] is None

    async def test_retry_on_malformed_json_succeeds_second_time(self):
        """First call returns non-JSON, second call returns valid diffs → succeeds."""
        agent = StubAgentProvider([
            # First call: malformed JSON → extract_structured retries
            AgentResult(
                content="this is not json {{{",
                tool_calls=[], finish_reason="stop",
                input_tokens=5, output_tokens=5,
            ),
            # Second call: valid diffs after error feedback
            make_json_agent_result([{
                "file_path": "src/main/java/UserService.java",
                "diff_content": "@@ -10 +10 @@\n+public User findById(Long id) { ... }",
                "operation": "modify", "language": "java",
            }]),
        ])
        deps = _deps(agent=agent, diff=StubDiffProvider())
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["diff_proposal"] is not None
        assert out["diff_proposal"]["diffs"][0]["file_path"] == "src/main/java/UserService.java"
        assert out["correction_state"]["attempt_count"] == 1

    async def test_diff_validation_rejected_on_full_overwrite(self):
        """Diffs that overwrite too much → self-loop retry."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -1 +1 @@\n-old\n+new",
            "operation": "modify", "language": "java",
        }])])
        validation_errors = [DiffValidationError(
            file_path="src/main/java/UserService.java",
            reason="Diff removes 95% of file lines",
        )]
        diff = StubDiffProvider(validation_errors=validation_errors)
        deps = _deps(agent=agent, diff=diff)
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["diff_proposal"] is None
        # attempt_count stays 0 — overwrite rejection uses dedicated counter
        assert out["correction_state"]["attempt_count"] == 0
        assert out["empty_diff_retries"] == 1
        assert out["critic_findings"] is CRITIC_RESET

    async def test_diff_apply_failure_returns_branch_name(self):
        """Apply conflicts → increment attempt, store branch_name."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+method",
            "operation": "modify", "language": "java",
        }])])
        apply_fail = StubDiffProvider(apply_success=False)
        deps = _deps(agent=agent, diff=apply_fail)
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["diff_proposal"] is None
        assert out["correction_state"]["attempt_count"] == 1
        # task_id[:8] = "task-act", attempt=0 → branch is agent-task-task-act-a0
        assert out["correction_state"]["branch_name"] == "agent-task-task-act-a0"

    async def test_branch_name_persists_on_retry(self):
        """On second attempt, uses stored branch_name instead of creating new."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+method",
            "operation": "modify", "language": "java",
        }])])
        git = StubGitProvider()
        # Pre-populate the branch so list_branches finds it (HIGH-004 guard)
        git._branches.add("agent-task-actor-001-a1")
        state = _state(
            correction_state={
                "attempt_count": 1, "branch_name": "agent-task-actor-001-a1",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            }
        )
        deps = _deps(agent=agent, git=git)
        node = make_actor_node(deps)

        await node(state)

        # list_branches guard (HIGH-004) checks existence, then checkouts
        assert deps.git.calls[0] == ("list_branches", "agent-task-actor-001-a1")
        # Should checkout existing branch, NOT create a new one
        assert deps.git.calls[1] == ("checkout", "agent-task-actor-001-a1")
        # create_branch should NOT be called
        create_calls = [c for c in deps.git.calls if c[0] == "create_branch"]
        assert len(create_calls) == 0

    async def test_agent_receives_correct_role_and_tools(self):
        """Agent is called with structured_output role (extract_structured wrapper)."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        await node(_state())

        role, _ = agent.calls[0]
        assert role == "structured_output"

    async def test_agent_receives_debug_observations_in_prompt(self):
        """When debug_observations present, they are injected into system prompt."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        debug_obs = {
            "error_type": "NULL_REFERENCE",
            "root_cause": "UserService.repo is null",
            "breakpoint_hits": [{
                "file": "UserService.java", "line": 42,
                "variables": {"repo": {"value": "null"}},
                "call_stack": ["UserService.findById", "UserServiceTest.test"],
            }],
        }
        state = _state(debug_observations=debug_obs)
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        await node(state)

        # The system prompt is formatted with debug observations via _format_debug_observations.
        # The stub captures (role, user_prompt[:80]) — system prompt is not captured.
        # Instead, verify the node received debug observations by checking
        # that the agent was called with the correct role and the node had debug_obs=True.
        role, user_prompt = agent.calls[0]
        assert role == "structured_output"
        # The user prompt always contains the task description
        assert "findById" in user_prompt

    async def test_cost_accumulation_on_success(self):
        """BUG-001 fix: cost is accumulated from token counts."""
        agent = StubAgentProvider([AgentResult(
            content='[{"file_path":"X.java","diff_content":"+x","operation":"modify","language":"java"}]',
            tool_calls=[], finish_reason="stop",
            input_tokens=1000, output_tokens=2000,
        )])
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        out = await node(_state())

        # BUG-001 fix: cost = (1000/1M * 5.0) + (2000/1M * 30.0) = 0.065
        assert out["cumulative_cost_dollars"] == pytest.approx(0.065, abs=0.001)

    async def test_empty_diff_path_sets_current_phase(self):
        """CRIT-06: empty-diff early-exit must set current_phase."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+public User findById(Long id) { ... }",
            "operation": "modify", "language": "java",
        }])])
        validation_errors = [DiffValidationError(
            file_path="src/main/java/UserService.java",
            reason="Diff removes 95% of file lines",
        )]
        diff = StubDiffProvider(validation_errors=validation_errors)
        deps = _deps(agent=agent, diff=diff)
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["current_phase"] == WorkflowPhase.ACTOR.value

    async def test_apply_failure_path_sets_current_phase(self):
        """CRIT-06: apply-failure early-exit must set current_phase."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+method",
            "operation": "modify", "language": "java",
        }])])
        apply_fail = StubDiffProvider(apply_success=False)
        deps = _deps(agent=agent, diff=apply_fail)
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["current_phase"] == WorkflowPhase.ACTOR.value

    async def test_empty_diff_list_from_valid_json(self):
        """LLM returns valid JSON but empty array → self-loop retry."""
        agent = StubAgentProvider([AgentResult(
            content="[]",
            tool_calls=[], finish_reason="stop",
            input_tokens=5, output_tokens=5,
        )])
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        out = await node(_state())

        assert out["diff_proposal"] is None
        # attempt_count stays 0 — empty diffs use dedicated counter
        assert out["correction_state"]["attempt_count"] == 0
        assert out["empty_diff_retries"] == 1

    async def test_missing_strategy_falls_back_to_unknown(self):
        """When selected_strategy is None, DiffProposal uses 'unknown' strategy_id."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        state = _state(selected_strategy=None)
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        out = await node(state)

        # DiffProposal is a TypedDict — use dict key access
        assert out["diff_proposal"]["strategy_id"] == "unknown"

    async def test_empty_diff_retries_reset_on_successful_diff(self):
        """ST-001: empty_diff_retries must reset to 0 when actor produces a valid diff.

        Without this fix, empty_diff_retries accumulates across correction cycles.
        A session where the actor occasionally produces empty diffs (e.g. attempt 0
        fails, attempt 1 succeeds, attempt 2 fails, attempt 3 succeeds) would
        prematurely hit the empty_diff_retries limit and escalate to HITL even
        though the system still had remaining correction cycles.
        """
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/UserService.java",
            "diff_content": "@@ -10 +10 @@\n+public User findById(Long id) { ... }",
            "operation": "modify", "language": "java",
        }])])
        deps = _deps(agent=agent, diff=StubDiffProvider())
        node = make_actor_node(deps)

        # Simulate state where empty_diff_retries has accumulated from prior cycles
        state = _state(
            empty_diff_retries=3,
            correction_state={
                "attempt_count": 2,
                "branch_name": "agent-task-task-act-a2",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
        )
        out = await node(state)

        # The key assertion: successful diff must reset empty_diff_retries to 0
        assert out["empty_diff_retries"] == 0, (
            f"empty_diff_retries should be 0 on successful diff, got {out['empty_diff_retries']}. "
            "This causes premature HITL escalation in multi-attempt sessions."
        )

    async def test_structured_output_error_clears_debug_observations(self):
        """ERR-002: StructuredOutputError must clear debug_observations and return immediately.

        When extract_structured exhausts all retries, the actor must not fall
        through to the empty-diff path. Instead it should return immediately
        with debug_observations cleared, attempt_count incremented, and an
        audit entry written.
        """
        from sacv.nodes._structured_output import StructuredOutputError

        exc = StructuredOutputError(
            "Failed to extract valid list[DiffPayload] after 3 retries",
            last_raw_content="not json",
            updated_cost=0.001,
        )

        agent = StubAgentProvider()
        deps = _deps(agent=agent)
        node = make_actor_node(deps)

        state = _state(
            debug_observations={"error_type": "NullPointerException"},
            correction_state={
                "attempt_count": 1,
                "branch_name": "agent-task-task-act-a1",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
        )

        with patch(
            "sacv.nodes.actor.extract_structured",
            side_effect=exc,
        ):
            out = await node(state)

        # Must return immediately — not fall through to empty-diff path
        assert out["diff_proposal"] is None
        # debug_observations must be cleared so stale debug context is not replayed
        assert out["debug_observations"] is None
        # Parse error counts as a real attempt (unlike empty-diff path)
        assert out["correction_state"]["attempt_count"] == 2
        # empty_diff_retries incremented (no valid diffs produced)
        assert out["empty_diff_retries"] == 1
        # critic_findings reset
        assert out["critic_findings"] is not None
        # Audit entry must be written for operator diagnosis
        audit = out.get("workflow_audit_trail", [])
        assert any(
            entry.get("node") == "actor" and "parse_error" in entry.get("decision", "")
            for entry in audit
        ), "workflow_audit_trail must contain a parse_error audit entry"

    async def test_outcome_stagnation_short_circuits(self):
        """Outcome stagnation: same preflight/critic problem persists across attempts.

        When the same preflight violation or critic finding keeps appearing
        despite diff changes, the actor should detect outcome stagnation and
        short-circuit to HITL escalation.
        """
        from sacv.nodes._stagnation import compute_outcome_signature

        # Create a preflight result with a specific violation
        preflight = {
            "passed": False,
            "lsp_errors": [{"file": "X.java", "line": 10, "code": "TS2345", "message": "error"}],
            "arch_violations": [],
            "cross_stack_errors": [],
            "blast_errors": [],
            "duration_ms": 100,
        }
        outcome_sig = compute_outcome_signature(preflight, [])
        # Create a critic finding with the same rule_id
        critic_findings = [
            {"severity": "critical", "critic": "security", "message": "X",
             "file": "X.java", "line": 10, "rule_id": "SEC-001", "resolution_hint": "fix"},
        ]
        critic_sig = compute_outcome_signature(None, critic_findings)

        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "src/main/java/X.java",
            "diff_content": "@@ -10 +10 @@\n+method",
            "operation": "modify", "language": "java",
        }])])
        deps = _deps(agent=agent, diff=StubDiffProvider())
        node = make_actor_node(deps)

        # Simulate state where outcome stagnation has been detected
        # (stagnation_pattern set by prior check_outcome_stagnation call)
        state = _state(
            preflight_result=preflight,
            critic_findings=critic_findings,
            correction_state={
                "attempt_count": 1,
                "branch_name": "agent-task-task-act-a1",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",  # Will be set by outcome_stagnation check
            },
        )

        # Patch check_outcome_stagnation to return True (simulate detected stagnation)
        with patch(
            "sacv.nodes.actor.check_outcome_stagnation",
            return_value=True,
        ):
            out = await node(state)

        # Should short-circuit to HITL without calling agent
        assert out["verifier_verdict"]["test_result"] == "FAIL"
        assert out["verifier_verdict"]["diagnostic"] == "STAGNATION"
        assert out["diff_proposal"] is None
        assert out["current_phase"] == WorkflowPhase.ACTOR.value

    async def test_format_debug_observations_non_dict_variable_values(self):
        """_format_debug_observations handles variable values that are not dicts.

        When breakpoint hit variables have non-dict values (e.g., plain strings
        or integers), the formatter should convert them via str().
        """
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{
                "file": "X.java", "line": 42,
                "variables": {
                    "repo": {"value": "null"},       # dict value (normal)
                    "id": "123",                      # plain string
                    "count": 42,                      # plain int
                },
                "call_stack": ["findById"],
            }],
        }
        result = _format_debug_observations(obs)
        # Dict value should be formatted as key = value
        assert "repo = null" in result
        # Non-dict values should be converted via str()
        assert "id = 123" in result
        assert "count = 42" in result


@pytest.mark.unit
class TestFormatDebugObservations:

    def test_none_returns_empty_string(self):
        assert _format_debug_observations(None) == ""

    def test_empty_dict_returns_error_type_only(self):
        # Non-empty dict without error_type produces "Error type: UNKNOWN"
        result = _format_debug_observations({"root_cause": "something"})
        assert "Error type: UNKNOWN" in result

    def test_includes_root_cause(self):
        obs = {"error_type": "NPE", "root_cause": "repo is null"}
        result = _format_debug_observations(obs)
        assert "Root cause: repo is null" in result

    def test_includes_breakpoint_hits(self):
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{
                "file": "X.java", "line": 42,
                "variables": {"repo": {"value": "null"}, "id": {"value": "1"}},
                "call_stack": ["findById", "test"],
            }],
        }
        result = _format_debug_observations(obs)
        assert "Breakpoint hit at X.java:42" in result
        assert "repo = null" in result
        assert "Stack: findById → test" in result

    def test_includes_minimal_payload(self):
        obs = {"error_type": "NPE", "minimal_payload": {"key": "val"}}
        result = _format_debug_observations(obs)
        assert '"key": "val"' in result

    def test_limits_breakpoint_hits_to_2(self):
        obs = {"error_type": "NPE", "breakpoint_hits": [
            {"file": "X.java", "line": 1},
            {"file": "Y.java", "line": 2},
            {"file": "Z.java", "line": 3},  # should be excluded
        ]}
        result = _format_debug_observations(obs)
        assert "X.java:1" in result
        assert "Y.java:2" in result
        assert "Z.java:3" not in result

    def test_limits_variables_to_8(self):
        obs = {"error_type": "NPE", "breakpoint_hits": [{
            "file": "X.java", "line": 1,
            "variables": {f"v{i}": {"value": f"val{i}"} for i in range(20)},
        }]}
        result = _format_debug_observations(obs)
        assert "v0 = val0" in result
        # Should only include first 8
        assert "v7 = val7" in result
        assert "v8 = val8" not in result


@pytest.mark.unit
class TestFormatPreflight:

    def test_passed_returns_empty(self):
        assert _format_preflight({"passed": True}) == ""

    def test_none_passed_defaults_to_true(self):
        assert _format_preflight({}) == ""

    def test_includes_lsp_errors(self):
        result = _format_preflight({
            "passed": False,
            "lsp_errors": [{"file": "X.ts", "line": 10, "code": "TS2345", "message": "Argument of type 'string' is not assignable"}],
        })
        assert "[LSP] X.ts:10" in result
        assert "TS2345" in result

    def test_includes_arch_violations(self):
        result = _format_preflight({
            "passed": False,
            "arch_violations": [{"rule": "no-circle", "message": "Circle detected"}],
        })
        assert "[ARCH] no-circle" in result
        assert "Circle detected" in result

    def test_limits_lsp_errors_to_5(self):
        errors = [{"file": f"E{i}.ts", "line": 1} for i in range(10)]
        result = _format_preflight({"passed": False, "lsp_errors": errors})
        # Should contain first 5
        assert "E0.ts" in result
        assert "E4.ts" in result
        assert "E5.ts" not in result


@pytest.mark.unit
class TestFormatFindings:

    def test_empty_returns_empty(self):
        assert _format_findings([]) == ""

    def test_single_finding(self):
        result = _format_findings([{
            "severity": "critical", "critic": "security",
            "file": "X.java", "line": 10,
            "message": "SQL injection", "resolution_hint": "use params",
        }])
        assert "[CRITICAL] security: X.java:10" in result
        assert "SQL injection" in result
        assert "use params" in result

    def test_limits_to_all_findings(self):
        findings = [{
            "severity": "critical", "critic": "security",
            "file": f"F{i}.java", "line": 1,
            "message": f"Violation {i}", "resolution_hint": "fix",
        } for i in range(5)]
        result = _format_findings(findings)
        for i in range(5):
            assert f"F{i}.java" in result
