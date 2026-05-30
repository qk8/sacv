"""
tests/integration/test_two_phase_verifier.py
=============================================
Integration tests for the Two-Phase Guardrail (approach 8).

Validates:
1. Phase 1 failure (regression) blocks Phase 2 and returns FIX_IMPL.
2. Phase 1 pass + Phase 2 failure → FAIL with phase flags set correctly.
3. Both phases pass → PASS verdict.
4. Test deletion in diff → immediate rejection before Docker.
5. Critical critic findings block Docker entirely.
6. Blast-radius schema impact triggers additional API test run.
"""
from __future__ import annotations
import pytest
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import (
    WorkflowState, WorkflowPhase, DiffProposal, UnifiedDiffPayload,
)
from sacv.nodes.verifier import make_verifier_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)
from sacv.interfaces.sandbox_provider import ExecResult


def _deps(sandbox, config=None):
    from sacv.orchestration.graph import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(), memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(), sandbox=sandbox, diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _state(**kw):
    base = {
        "session_id":"t","task_id":"task-v2p-001","task_description":"Add findById",
        "project_mode":"greenfield","module_type":"backend-domain",
        "current_phase":WorkflowPhase.VERIFIER.value,
        "context_skeleton":None,"blast_radius_map":None,"agents_md_context":None,
        "strategy_candidates":[],"selected_strategy":None,"pruned_strategies":[],
        "red_phase_evidence_path":".workflow/tdd-evidence/task-v2p-001.json",
        "test_inventory_paths":["src/test/java/com/sacv/FindByIdTest.java"],
        "diff_proposal":DiffProposal(
            strategy_id="s1",branch_name="agent-task-abc-a1",
            commit_message="impl",
            diffs=[UnifiedDiffPayload(
                file_path="src/main/java/UserService.java",
                diff_content="@@ -10 +10 @@\n+findById",
                operation="modify",language="java",
            )],
        ),
        "preflight_result":{"passed":True,"lsp_errors":[],"arch_violations":[],"duration_ms":50},
        "critic_findings":[],
        "verifier_verdict":None,
        "correction_state":{"attempt_count":1,"branch_name":"agent-task-abc-a1",
                            "last_error_hash":None,"error_history":[],"stagnation_pattern":"none"},
        "confidence_score":0.8,"replan_count":0,
        "active_branches":[],"exhausted_branches":[],"escalation_payload":None,
        "procedural_constraints":[],"lesson_learned":None,"arch_rules_updated":False,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.integration
class TestTwoPhaseVerifier:

    async def test_both_phases_pass_returns_pass_verdict(self):
        """Phase 1 + Phase 2 both pass → PASS."""
        sandbox = _StagedSandbox(phase1_exit=0, phase2_exit=0)
        out     = await make_verifier_node(_deps(sandbox))(_state())
        v = out["verifier_verdict"]
        assert v["test_result"]  == "PASS"
        assert v["phase1_passed"] is True
        assert v["phase2_passed"] is True

    async def test_phase1_failure_stops_pipeline(self):
        """Phase 1 regression → FAIL immediately, Phase 2 not run."""
        sandbox = _StagedSandbox(phase1_exit=1, phase2_exit=0)
        out     = await make_verifier_node(_deps(sandbox))(_state())
        v = out["verifier_verdict"]
        assert v["test_result"]   == "FAIL"
        assert v["phase1_passed"] is False
        assert v["phase2_passed"] is False   # never ran

    async def test_phase1_pass_phase2_fail_returns_fail(self):
        """Phase 1 OK but new feature tests still fail → FAIL."""
        sandbox = _StagedSandbox(phase1_exit=0, phase2_exit=1)
        out     = await make_verifier_node(_deps(sandbox))(_state())
        v = out["verifier_verdict"]
        assert v["test_result"]   == "FAIL"
        assert v["phase1_passed"] is True
        assert v["phase2_passed"] is False

    async def test_test_deletion_rejected_before_docker(self):
        """If diff deletes a test file, abort before Docker is called."""
        state = _state(diff_proposal=DiffProposal(
            strategy_id="s1", branch_name="b", commit_message="impl",
            diffs=[UnifiedDiffPayload(
                file_path="tests/api/routes/user.spec.ts",
                diff_content="-all content",
                operation="delete", language="typescript",
            )],
        ))
        state["module_type"] = "frontend-feature"
        sandbox = _NeverCallSandbox()
        out     = await make_verifier_node(_deps(sandbox))(state)
        v = out["verifier_verdict"]
        assert v["test_result"]   == "FAIL"
        assert not sandbox.was_called, "Docker must not be invoked when tests are deleted"

    async def test_critical_finding_blocks_docker(self):
        """Critical critic finding skips Docker entirely."""
        state = _state(critic_findings=[{
            "critic":"security","severity":"critical","file":"X.java","line":5,
            "rule_id":"SEC-001","message":"SQL injection","resolution_hint":"use params",
        }])
        sandbox = _NeverCallSandbox()
        out     = await make_verifier_node(_deps(sandbox))(state)
        assert out["verifier_verdict"]["test_result"] == "FAIL"
        assert not sandbox.was_called

    async def test_schema_impact_triggers_extra_api_run(self):
        """blast_radius_map.schema_impact non-empty → extra API test run."""
        state = _state(blast_radius_map={
            "entry_files":[],"affected_files":["User.java"],
            "dependency_depth":1,"cross_service_impact":[],
            "schema_impact":["users_table"],
            "risk_score":0.6,
        })
        call_log: list[str] = []

        class _TrackedSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, command, env=None, timeout=120):
                call_log.append(command[:60])
                return ExecResult(0, "BUILD SUCCESS\nTests run: 3, Failures: 0", "", 100)

        out = await make_verifier_node(_deps(_TrackedSandbox()))(_state(
            blast_radius_map=state["blast_radius_map"]
        ))
        # At least 3 exec calls: phase1, phase2, and schema-impact API run
        assert len(call_log) >= 3


# ── Sandbox helpers ───────────────────────────────────────────────────────────

class _StagedSandbox(StubSandboxProvider):
    def __init__(self, phase1_exit: int, phase2_exit: int):
        super().__init__()
        self._phase1_exit = phase1_exit
        self._phase2_exit = phase2_exit
        self._call = 0

    async def exec_in_container(self, handle, command, env=None, timeout=120):
        self._call += 1
        exit_code = self._phase1_exit if self._call == 1 else self._phase2_exit
        stdout = (
            "BUILD SUCCESS\nTests run: 5, Failures: 0"
            if exit_code == 0 else
            "BUILD FAILURE\nTests run: 5, Failures: 2"
        )
        return ExecResult(exit_code, stdout, "", 150)


class _NeverCallSandbox(StubSandboxProvider):
    was_called = False

    async def exec_in_container(self, handle, command, env=None, timeout=120):
        self.was_called = True
        return ExecResult(0, "", "", 0)
