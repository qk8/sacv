"""
tests/unit/test_stub_providers.py
===================================
Unit tests for the stub provider implementations.

Tests cover:
1. StubAgentProvider — FIFO response queue, exhaustion error, enqueue
2. StubAgentProvider — call logging (role, prompt)
3. make_json_agent_result — correct AgentResult construction
4. StubMemoryProvider — store/retrieve episodic events
5. StubMemoryProvider — procedural constraints
6. StubCodeGraphProvider — blast radius, call graph, subgraph
7. StubCrossDomainProvider — map_code_to_schema, get_arch_alignment, get_sql_impact
8. StubDiffProvider — validation errors, apply success/failure
9. StubGitProvider — branch management call recording
10. StubSandboxProvider — default vs override results
"""
from __future__ import annotations

import pytest

from sacv.interfaces.agent_provider import AgentConfig, AgentResult
from sacv.interfaces.memory_provider import EpisodicEvent, ProceduralConstraint
from sacv.interfaces.code_graph_provider import CallGraph, BlastRadiusMap
from sacv.interfaces.diff_provider import DiffValidationError, UnifiedDiff
from sacv.interfaces.sandbox_provider import SandboxHandle, ExecResult
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


@pytest.mark.asyncio
@pytest.mark.unit
class TestStubAgentProvider:

    async def test_pops_responses_in_fifo_order(self):
        r1 = AgentResult(content="first", tool_calls=[], finish_reason="stop",
                         input_tokens=1, output_tokens=1)
        r2 = AgentResult(content="second", tool_calls=[], finish_reason="stop",
                         input_tokens=2, output_tokens=2)
        provider = StubAgentProvider([r1, r2])

        result1 = await provider.run_task("prompt1", {}, AgentConfig(role="test", system_prompt="", max_turns=1))
        result2 = await provider.run_task("prompt2", {}, AgentConfig(role="test", system_prompt="", max_turns=1))

        assert result1.content == "first"
        assert result2.content == "second"

    async def test_exhaustion_raises_assertion_error(self):
        provider = StubAgentProvider([])
        with pytest.raises(AssertionError, match="exhausted"):
            await provider.run_task("prompt", {}, AgentConfig(role="test", system_prompt="", max_turns=1))

    async def test_enqueued_responses_appended(self):
        provider = StubAgentProvider([])
        provider.enqueue(AgentResult(content="enqueued", tool_calls=[], finish_reason="stop",
                                     input_tokens=1, output_tokens=1))
        result = await provider.run_task("prompt", {}, AgentConfig(role="test", system_prompt="", max_turns=1))
        assert result.content == "enqueued"

    async def test_records_call_role_and_prompt(self):
        provider = StubAgentProvider([make_json_agent_result("x")])
        config = AgentConfig(role="build_agent", system_prompt="", max_turns=1)
        await provider.run_task("this is a long prompt that should be truncated", {}, config)

        assert len(provider.calls) == 1
        role, prompt = provider.calls[0]
        assert role == "build_agent"
        assert len(prompt) <= 80

    async def test_call_prompt_truncated_to_80_chars(self):
        provider = StubAgentProvider([make_json_agent_result("x")])
        config = AgentConfig(role="test", system_prompt="", max_turns=1)
        long_prompt = "x" * 200
        await provider.run_task(long_prompt, {}, config)

        _, prompt = provider.calls[0]
        assert len(prompt) == 80


@pytest.mark.unit
class TestMakeJsonAgentResult:

    def test_wraps_dict_as_json(self):
        result = make_json_agent_result({"key": "value"})
        assert result.content == '{"key": "value"}'
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_wraps_list_as_json(self):
        result = make_json_agent_result([1, 2, 3])
        assert result.content == "[1, 2, 3]"

    def test_wraps_string_as_json(self):
        result = make_json_agent_result("hello")
        assert result.content == '"hello"'

    def test_custom_token_count(self):
        result = make_json_agent_result({"key": "value"}, tokens=100)
        assert result.input_tokens == 100
        assert result.output_tokens == 100

    def test_default_token_count(self):
        result = make_json_agent_result({"key": "value"})
        assert result.input_tokens == 10
        assert result.output_tokens == 10


@pytest.mark.unit
class TestStubMemoryProvider:

    async def test_store_and_retrieve_episodic(self):
        provider = StubMemoryProvider()
        event = EpisodicEvent(
            session_id="s1", event_type="test",
            payload={"key": "value"},
            timestamp="2024-01-01T00:00:00Z",
        )
        await provider.store_episodic(event)
        assert len(provider.stored_events) == 1
        assert provider.stored_events[0] == event

    async def test_retrieve_procedural_returns_constraints(self):
        constraints = [
            ProceduralConstraint(constraint_id="c1", category="test",
                                 description="Test constraint", weight=1.0),
        ]
        provider = StubMemoryProvider(procedural=constraints)
        result = await provider.retrieve_procedural(["test"])
        assert result == constraints

    async def test_retrieve_procedural_ignores_context_tags(self):
        """Stub returns all stored constraints regardless of context_tags."""
        constraints = [
            ProceduralConstraint(constraint_id="c1", category="security",
                                 description="S1", weight=1.0),
            ProceduralConstraint(constraint_id="c2", category="performance",
                                 description="S2", weight=0.5),
        ]
        provider = StubMemoryProvider(procedural=constraints)
        result = await provider.retrieve_procedural(["tag1"])
        assert len(result) == 2

    async def test_purge_noise_records_session(self):
        provider = StubMemoryProvider()
        await provider.purge_noise("session-1")
        assert "session-1" in provider.purged_sessions

    async def test_empty_initial_state(self):
        provider = StubMemoryProvider()
        assert provider.stored_events == []
        assert provider.purged_sessions == []


@pytest.mark.unit
class TestStubCodeGraphProvider:

    @pytest.mark.asyncio
    async def test_get_blast_radius_returns_configured(self):
        blast = BlastRadiusMap(
            entry_files=["X.java"], affected_files=["X.java"],
            dependency_depth=1, cross_service_impact=[],
            schema_impact=[], risk_score=0.5,
        )
        provider = StubCodeGraphProvider(blast=blast)
        result = await provider.get_blast_radius(["X.java"])
        assert result.entry_files == ["X.java"]

    async def test_get_blast_radius_default_empty(self):
        provider = StubCodeGraphProvider()
        result = await provider.get_blast_radius(["X.java"])
        assert result.entry_files == []
        assert result.risk_score == 0.0

    async def test_get_call_graph_returns_configured(self):
        graph = CallGraph(".", ["node1"], ["edge1"])
        provider = StubCodeGraphProvider(graph=graph)
        result = await provider.get_call_graph(["X.java"])
        assert result.entry_point == "."
        assert result.nodes == ["node1"]

    async def test_get_call_graph_default_empty(self):
        provider = StubCodeGraphProvider()
        result = await provider.get_call_graph(["X.java"])
        assert result.nodes == []

    async def test_get_dependency_subgraph_returns_configured(self):
        provider = StubCodeGraphProvider(subgraph={"dep1": "value"})
        result = await provider.get_dependency_subgraph(["X.java"])
        assert result == {"dep1": "value"}

    async def test_get_dependency_subgraph_default_empty(self):
        provider = StubCodeGraphProvider()
        result = await provider.get_dependency_subgraph(["X.java"])
        assert result == {}


@pytest.mark.unit
class TestStubCrossDomainProvider:

    async def test_map_code_to_schema(self):
        provider = StubCrossDomainProvider()
        result = await provider.map_code_to_schema(["UserService", "UserRepo"])
        assert result == {"entities": ["UserService", "UserRepo"]}

    async def test_get_arch_alignment(self):
        provider = StubCrossDomainProvider()
        result = await provider.get_arch_alignment(["src/main/java"])
        assert result == {"aligned": True}

    async def test_get_sql_impact(self):
        provider = StubCrossDomainProvider()
        result = await provider.get_sql_impact(["User.java"])
        assert result == {"affected_tables": []}


@pytest.mark.unit
class TestStubDiffProvider:

    async def test_validate_returns_configured_errors(self):
        errors = [DiffValidationError(file_path="X.java", reason="too many lines")]
        provider = StubDiffProvider(validation_errors=errors)
        result = await provider.validate_no_full_overwrite([])
        assert result == errors

    async def test_validate_default_no_errors(self):
        provider = StubDiffProvider()
        result = await provider.validate_no_full_overwrite([])
        assert result == []

    async def test_apply_diffs_returns_success(self):
        provider = StubDiffProvider(apply_success=True)
        diffs = [UnifiedDiff(file_path="X.java", diff_content="+x",
                             operation="modify", language="java")]
        result = await provider.apply_diffs(diffs)
        assert result.success
        assert result.applied_files == ["X.java"]
        assert result.conflicts == []

    async def test_apply_diffs_returns_failure(self):
        provider = StubDiffProvider(apply_success=False)
        diffs = [UnifiedDiff(file_path="X.java", diff_content="+x",
                             operation="modify", language="java")]
        result = await provider.apply_diffs(diffs)
        assert not result.success
        assert result.applied_files == []
        assert len(result.conflicts) == 1

    async def test_applied_diffs_recorded(self):
        provider = StubDiffProvider(apply_success=True)
        diffs = [UnifiedDiff(file_path="X.java", diff_content="+x",
                             operation="modify", language="java")]
        await provider.apply_diffs(diffs)
        assert len(provider.applied) == 1
        assert provider.applied[0] == diffs

    async def test_generate_ast_diff(self):
        provider = StubDiffProvider()
        result = await provider.generate_ast_diff("old", "new", "java")
        assert result.file_path == "stub.java"
        assert "old" in result.diff_content
        assert "new" in result.diff_content
        assert result.operation == "modify"
        assert result.language == "java"


@pytest.mark.unit
class TestStubGitProvider:

    def test_create_branch_records_call(self):
        provider = StubGitProvider()
        provider.create_branch("feature-1")
        assert provider.calls == [("create_branch", "feature-1", "HEAD")]

    def test_checkout_records_call(self):
        provider = StubGitProvider()
        provider.checkout("feature-1")
        assert provider.calls == [("checkout", "feature-1")]

    def test_stash_records_call(self):
        provider = StubGitProvider()
        provider.stash("msg")
        assert provider.calls == [("stash", "msg")]

    def test_stash_pop_records_call(self):
        provider = StubGitProvider()
        provider.stash_pop("ref")
        assert provider.calls == [("stash_pop", "ref")]

    def test_reset_hard_records_call(self):
        provider = StubGitProvider()
        provider.reset_hard("abc123")
        assert provider.calls == [("reset_hard", "abc123")]

    def test_get_last_green_commit_returns_configured(self):
        provider = StubGitProvider(green_sha="custom_sha")
        assert provider.get_last_green_commit() == "custom_sha"

    def test_default_green_sha(self):
        provider = StubGitProvider()
        assert provider.get_last_green_commit() == "abc1234deadbeef"

    def test_record_green_commit_records_and_updates(self):
        provider = StubGitProvider()
        provider.record_green_commit("new_sha")
        assert provider.calls == [("record_green", "new_sha")]
        assert provider.get_last_green_commit() == "new_sha"

    def test_current_branch_returns_current(self):
        provider = StubGitProvider()
        assert provider.current_branch() == "main"

    def test_checkout_updates_current_branch(self):
        provider = StubGitProvider()
        provider.checkout("feature-1")
        assert provider.current_branch() == "feature-1"

    def test_uncommitted_files_returns_empty(self):
        provider = StubGitProvider()
        assert provider.uncommitted_files() == []

    def test_create_worktree_records_call(self):
        provider = StubGitProvider()
        provider.create_worktree("feature-1", "/path")
        assert provider.calls == [("create_worktree", "feature-1", "/path")]

    def test_remove_worktree_records_call(self):
        provider = StubGitProvider()
        provider.remove_worktree("/path")
        assert provider.calls == [("remove_worktree", "/path")]

    def test_stage_file_records_call(self):
        provider = StubGitProvider()
        provider.stage_file("X.java")
        assert provider.calls == [("stage_file", "X.java")]

    def test_head_sha_returns_green_sha(self):
        provider = StubGitProvider()
        assert provider.head_sha() == "abc1234deadbeef"

    def test_commit_records_call(self):
        provider = StubGitProvider()
        sha = provider.commit("test message")
        assert provider.calls == [("commit", "test message")]
        assert sha == "deadbeef00000000"


@pytest.mark.asyncio
@pytest.mark.unit
class TestStubSandboxProvider:

    async def test_warm_container_returns_handle(self):
        provider = StubSandboxProvider()
        handle = await provider.warm_container()
        assert handle.container_id == "stub-container-id"
        assert handle.working_dir == "/workspace"
        assert handle.warm is True

    async def test_default_exec_result(self):
        provider = StubSandboxProvider()
        handle = await provider.warm_container()
        result = await provider.exec_in_container(handle, "any command")
        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""

    async def test_custom_default_result(self):
        provider = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        handle = await provider.warm_container()
        result = await provider.exec_in_container(handle, "any command")
        assert result.exit_code == 1
        assert "FAIL" in result.stdout

    async def test_register_override(self):
        provider = StubSandboxProvider()
        provider.register("mvn test", ExecResult(1, "BUILD FAILURE", "", 100))
        handle = await provider.warm_container()
        result = await provider.exec_in_container(handle, "mvn test -q")
        assert result.exit_code == 1
        assert "BUILD FAILURE" in result.stdout

    async def test_override_falls_back_to_default(self):
        provider = StubSandboxProvider(default_exit_code=0)
        provider.register("mvn test", ExecResult(1, "FAIL", "", 100))
        handle = await provider.warm_container()
        # Command that doesn't match any override
        result = await provider.exec_in_container(handle, "npm test")
        assert result.exit_code == 0

    async def test_exec_calls_recorded(self):
        provider = StubSandboxProvider()
        handle = await provider.warm_container()
        await provider.exec_in_container(handle, "cmd1")
        await provider.exec_in_container(handle, "cmd2")
        assert provider.exec_calls == ["cmd1", "cmd2"]

    async def test_destroy_container_noop(self):
        provider = StubSandboxProvider()
        handle = await provider.warm_container()
        # Should not raise
        await provider.destroy_container(handle)

    async def test_create_isolated_instance(self):
        provider = StubSandboxProvider(default_exit_code=1, default_stdout="isolated")
        isolated = provider.create_isolated_instance("/host")
        assert isinstance(isolated, StubSandboxProvider)
        handle = await isolated.warm_container()
        result = await isolated.exec_in_container(handle, "any")
        assert result.exit_code == 1
        assert "isolated" in result.stdout
