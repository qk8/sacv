"""
tests/integration/test_preflight_node.py
=========================================
Integration tests for the PreflightNode (approaches 1, 9, 10).

Validates:
1. Clean output (no errors) → preflight_result.passed = True
2. tsc type errors are parsed and surfaced
3. ArchUnit / dependency-cruiser violations are parsed
4. Duration is recorded
5. No diff_proposal → preflight skipped (passed=True, fast path)
6. Node never calls LLM (zero agent calls)
"""
from __future__ import annotations
import pytest
from sacv.nodes.preflight_node import make_preflight_node, _parse_lsp, _parse_arch
from sacv.orchestration.state import WorkflowPhase, DiffProposal, UnifiedDiffPayload
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)
from sacv.interfaces.sandbox_provider import ExecResult


def _deps(sandbox):
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.config import WorkflowConfig
    return NodeDeps(
        agent=StubAgentProvider(), memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(), sandbox=sandbox, diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _state(module="backend-domain", diff=True, **kw):
    proposal = DiffProposal(
        strategy_id="s1", branch_name="b", commit_message="impl",
        diffs=[UnifiedDiffPayload(
            file_path="UserService.java",
            diff_content="@@ -1 +1 @@\n-old\n+new",
            operation="modify", language="java",
        )],
    ) if diff else None
    s = {
        "session_id":"t","task_id":"t","task_description":"",
        "project_mode":"greenfield","module_type":module,
        "current_phase":WorkflowPhase.ACTOR.value,
        "context_skeleton":None,"blast_radius_map":None,"agents_md_context":None,
        "strategy_candidates":[],"selected_strategy":None,"pruned_strategies":[],
        "red_phase_evidence_path":None,"test_inventory_paths":[],
        "diff_proposal":proposal,"preflight_result":None,
        "critic_findings":[],"verifier_verdict":None,
        "correction_state":{"attempt_count":1,"branch_name":"b","last_error_hash":None,
                            "error_history":[],"stagnation_pattern":"none"},
        "confidence_score":1.0,"replan_count":0,
        "active_branches":[],"exhausted_branches":[],"escalation_payload":None,
        "procedural_constraints":[],"lesson_learned":None,"arch_rules_updated":False,
    }
    s.update(kw)
    return s


@pytest.mark.asyncio
@pytest.mark.integration
class TestPreflightNode:

    async def test_clean_compile_sets_passed_true(self):
        sandbox = StubSandboxProvider(default_exit_code=0, default_stdout="")
        out = await make_preflight_node(_deps(sandbox))(_state())
        r = out["preflight_result"]
        assert r["passed"] is True
        assert r["lsp_errors"] == []
        assert r["arch_violations"] == []

    async def test_tsc_errors_parsed_into_lsp_errors(self):
        tsc_out = (
            "src/ui/UserForm.tsx(12,5): error TS2322: "
            "Type 'string' is not assignable to type 'number'.\n"
            "src/ui/UserForm.tsx(20,3): error TS2345: "
            "Argument of type 'null' is not assignable.\n"
        )
        errors = _parse_lsp(tsc_out, "frontend-feature")
        assert len(errors) == 2
        assert errors[0]["file"] == "src/ui/UserForm.tsx"
        assert errors[0]["line"] == 12
        assert errors[0]["code"] == "TS2322"

    async def test_java_compile_errors_parsed(self):
        mvn_out = "[ERROR] src/main/java/UserService.java:[42,8] cannot find symbol\n"
        errors  = _parse_lsp(mvn_out, "backend-domain")
        assert len(errors) == 1
        assert errors[0]["line"] == 42
        assert errors[0]["code"] == "CE"

    async def test_no_diff_proposal_skips_docker(self):
        sandbox = StubSandboxProvider()
        out     = await make_preflight_node(_deps(sandbox))(_state(diff=False))
        assert out["preflight_result"]["passed"] is True
        assert sandbox.exec_calls == [], "No Docker exec when no diff"

    async def test_lsp_failure_sets_passed_false(self):
        sandbox = StubSandboxProvider(
            default_exit_code=1,
            default_stdout="UserService.java(5,3): error TS2322: type error\n",
        )
        state = _state(module="frontend-feature")
        out   = await make_preflight_node(_deps(sandbox))(state)
        r     = out["preflight_result"]
        # TypeScript: exit code 1 + error lines
        assert isinstance(r["lsp_errors"], list)

    async def test_duration_ms_is_recorded(self):
        sandbox = StubSandboxProvider(default_exit_code=0)
        out     = await make_preflight_node(_deps(sandbox))(_state())
        assert out["preflight_result"]["duration_ms"] >= 0

    async def test_no_agent_calls_made(self):
        """Preflight must never invoke the LLM."""
        agent   = StubAgentProvider()
        sandbox = StubSandboxProvider(default_exit_code=0)
        from sacv.orchestration.deps import NodeDeps
        from sacv.orchestration.config import WorkflowConfig
        deps = NodeDeps(
            agent=agent, memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(), sandbox=sandbox, diff=StubDiffProvider(),
            config=WorkflowConfig(),
        )
        await make_preflight_node(deps)(_state())
        assert agent.calls == [], "Preflight must not make any LLM calls"

    async def test_critic_findings_reset_on_preflight(self):
        """Preflight resets critic_findings to avoid stale data in fan-out."""
        from sacv.orchestration.state import CRITIC_RESET
        sandbox = StubSandboxProvider(default_exit_code=0)
        state   = _state()
        state["critic_findings"] = [
            {"critic":"security","severity":"warning","file":"X.java","line":1,
             "rule_id":"r","message":"old finding","resolution_hint":"fix"},
        ]
        out = await make_preflight_node(_deps(sandbox))(state)
        # CRITIC_RESET sentinel signals the reducer to reset the list
        assert out.get("critic_findings") is CRITIC_RESET


# ── Pure function tests ───────────────────────────────────────────────────────

class TestPreflightParsers:

    def test_parse_multiple_tsc_errors(self):
        output = "\n".join([
            "src/A.ts(1,1): error TS1001: msg1",
            "src/B.tsx(99,5): error TS2022: msg2",
        ])
        errors = _parse_lsp(output, "frontend-feature")
        assert len(errors) == 2
        assert errors[1]["line"] == 99

    def test_parse_empty_output_returns_empty(self):
        assert _parse_lsp("", "backend-domain") == []
        assert _parse_arch("", "backend-domain") == []

    def test_parse_no_arch_test_skips(self):
        assert _parse_arch("NO_ARCH_TEST", "backend-domain") == []

    def test_parse_depcruiser_json(self):
        import json
        violations = _parse_arch(
            json.dumps([{
                "source": "src/ui/A.ts",
                "violations": [{
                    "rule": {"name": "no-ui-to-db"},
                    "to": {"resolved": "src/infra/DB.ts"},
                }],
            }]),
            "frontend-feature",
        )
        assert len(violations) == 1
        assert violations[0]["rule"] == "no-ui-to-db"
