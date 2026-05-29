"""
Integration tests for the three-critic parallel fan-out/fan-in.
Updated for new WorkflowState fields.
"""
from __future__ import annotations
import asyncio, json
import pytest
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowState, WorkflowPhase, DiffProposal, UnifiedDiffPayload
from sacv.nodes.critics.security    import make_security_critic_node
from sacv.nodes.critics.style       import make_style_critic_node
from sacv.nodes.critics.consistency import make_consistency_critic_node
from sacv.nodes.critics.base        import make_aggregate_critics_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _deps(agent, config=None):
    from sacv.orchestration.graph import NodeDeps
    return NodeDeps(
        agent=agent, memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(), sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(), config=config or WorkflowConfig(),
    )


def _proposal():
    return DiffProposal(
        strategy_id="s1",
        diffs=[UnifiedDiffPayload(
            file_path="src/main/java/com/example/UserService.java",
            diff_content="@@ -10,6 +10,10 @@\n+    public User createUser(String email) {\n+        return userRepo.save(new User(email));\n+    }",
            operation="modify", language="java",
        )],
        branch_name="agent-task-abc12345-a1",
        commit_message="sacv: implement create-user",
    )


def _state(mode="greenfield"):
    return {
        "session_id":"t","task_id":"task-001","task_description":"Create user endpoint",
        "project_mode":mode,"module_type":"backend-domain","check_profile":"standard",
        "current_phase":WorkflowPhase.CRITICS.value,
        "context_skeleton":None,"blast_radius_map":None,"agents_md_context":None,
        "strategy_candidates":[],"selected_strategy":None,"pruned_strategies":[],
        "red_phase_evidence_path":"/p/e.json","test_inventory_paths":["tests/unit/UserTest.java"],
        "diff_proposal":_proposal(),"preflight_result":{"passed":True,"lsp_errors":[],"arch_violations":[],"duration_ms":50},
        "critic_findings":[],"verifier_verdict":None,
        "correction_state":{"attempt_count":1,"branch_name":"agent-task-abc12345-a1",
                            "last_error_hash":None,"error_history":[],"stagnation_pattern":"none"},
        "confidence_score":0.8,"replan_count":0,
        "active_branches":[],"exhausted_branches":[],"escalation_payload":None,
        "procedural_constraints":[],"lesson_learned":None,"arch_rules_updated":False,
    }


@pytest.mark.asyncio
@pytest.mark.integration
class TestCriticFanOut:

    async def test_all_critics_return_findings(self):
        agent = StubAgentProvider([
            make_json_agent_result([{"critic":"security","severity":"warning","file":"UserService.java","line":12,"rule_id":"SEC-001","message":"Missing @Valid","resolution_hint":"Add @Valid"}]),
            make_json_agent_result([{"critic":"style","severity":"info","file":"UserService.java","line":12,"rule_id":"DDD-003","message":"DDD issue","resolution_hint":"Use service"}]),
            make_json_agent_result([]),
        ])
        deps = _deps(agent)
        s    = _state()
        findings = (
            (await make_security_critic_node(deps)(s))["critic_findings"] +
            (await make_style_critic_node(deps)(s))["critic_findings"] +
            (await make_consistency_critic_node(deps)(s))["critic_findings"]
        )
        assert len(findings) == 2
        assert {f["critic"] for f in findings} == {"security", "style"}

    async def test_no_findings_returns_empty(self):
        for node_fn in [make_security_critic_node, make_style_critic_node, make_consistency_critic_node]:
            agent = StubAgentProvider([make_json_agent_result([])])
            out   = await node_fn(_deps(agent))(_state())
            assert out["critic_findings"] == []

    async def test_aggregate_advances_phase(self):
        agent = StubAgentProvider()
        out   = await make_aggregate_critics_node(_deps(agent))({**_state(), "critic_findings": []})
        assert out["current_phase"] == WorkflowPhase.VERIFIER.value

    async def test_concurrent_execution_no_deadlock(self):
        agent = StubAgentProvider([make_json_agent_result([])] * 3)
        deps  = _deps(agent)
        s     = _state()
        results = await asyncio.gather(
            make_security_critic_node(deps)(s),
            make_style_critic_node(deps)(s),
            make_consistency_critic_node(deps)(s),
        )
        assert len(results) == 3

    async def test_malformed_response_returns_empty(self):
        from sacv.interfaces.agent_provider import AgentResult
        agent = StubAgentProvider([AgentResult("not json", [], "stop", 5, 5)])
        out   = await make_security_critic_node(_deps(agent))(_state())
        assert out["critic_findings"] == []

    async def test_no_proposal_skips_llm(self):
        agent = StubAgentProvider()
        s     = {**_state(), "diff_proposal": None}
        out   = await make_security_critic_node(_deps(agent))(s)
        assert out["critic_findings"] == []
        assert not agent.calls

    async def test_brownfield_consistency_uses_different_rules(self):
        agent = StubAgentProvider([make_json_agent_result([])])
        out   = await make_consistency_critic_node(_deps(agent))(_state(mode="brownfield"))
        assert agent.calls[0][0] == "consistency"
