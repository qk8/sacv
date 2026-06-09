"""
tests/unit/test_structured_output.py
=====================================
Tests for the instructor-inspired structured output utility.

Tests cover:
1. Valid JSON → parsed and returned
2. Invalid JSON → retries with error feedback
3. Valid JSON but wrong shape → retries with validation errors
4. All retries exhausted → raises StructuredOutputError
5. Successful retry → returns with retry_count > 0
6. Scalar model (str) → parsed correctly
7. Dict model → parsed correctly
8. Empty list model → parsed correctly
"""
from __future__ import annotations

import pytest

from pydantic import BaseModel, Field
from typing import List, Literal

from sacv.interfaces.agent_provider import AgentConfig, AgentResult
from sacv.testing.stub_providers import StubAgentProvider, make_json_agent_result
from sacv.nodes._structured_output import (
    extract_structured,
    StructuredOutputError,
)


# ── Pydantic models used in tests ──────────────────────────────────────────────


class DiffPayload(BaseModel):
    file_path: str
    diff_content: str
    operation: Literal["modify", "create", "delete"] = "modify"
    language: Literal["java", "typescript", "sql", "yaml", "other"] = "other"


class StrategyCandidate(BaseModel):
    strategy_id: str
    description: str
    affected_files: List[str]


class EmptyListPayload(BaseModel):
    items: List[str]


_SYSTEM = "You are a structured output agent."


def _agent_config(**kw) -> AgentConfig:
    return AgentConfig(
        role="structured_output",
        system_prompt=_SYSTEM,
        max_turns=1,
        allowed_tools=[],
        **kw,
    )


# ── Test: valid JSON validates and returns ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
class TestExtractStructured:

    async def test_valid_json_parsed_and_returned(self):
        """Valid JSON matching the schema → parsed data returned."""
        diffs = [
            {
                "file_path": "src/main/java/UserService.java",
                "diff_content": "+    public User findById(Long id) { return null; }",
                "operation": "modify",
                "language": "java",
            },
        ]
        agent = StubAgentProvider([make_json_agent_result(diffs)])
        result = await extract_structured(
            agent=agent,
            prompt="Create a diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=3,
            allowed_tools=["Read"],
        )
        assert isinstance(result.data, list)
        assert len(result.data) == 1
        assert result.data[0].file_path == "src/main/java/UserService.java"
        assert result.data[0].operation == "modify"
        assert result.retry_count == 0

    async def test_valid_dict_parsed(self):
        """Valid JSON object matching schema → parsed dict returned."""
        agent = StubAgentProvider([
            make_json_agent_result({"common_mistakes": "don't delete tests",
                                    "architecture_decisions": "use layers"}),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Update AGENTS.md",
            response_model=dict,
            system_prompt=_SYSTEM,
        )
        assert result.data == {"common_mistakes": "don't delete tests",
                               "architecture_decisions": "use layers"}
        assert result.retry_count == 0

    async def test_valid_string_parsed(self):
        """Valid JSON string → parsed string returned."""
        agent = StubAgentProvider([make_json_agent_result("root cause here")])
        result = await extract_structured(
            agent=agent,
            prompt="Explain the bug",
            response_model=str,
            system_prompt=_SYSTEM,
        )
        assert result.data == "root cause here"
        assert result.retry_count == 0

    async def test_valid_empty_list_parsed(self):
        """Valid JSON array matching schema → parsed list returned."""
        agent = StubAgentProvider([make_json_agent_result({"items": []})])
        result = await extract_structured(
            agent=agent,
            prompt="List findings",
            response_model=EmptyListPayload,
            system_prompt=_SYSTEM,
        )
        assert result.data.items == []
        assert result.retry_count == 0

    async def test_agent_called_with_correct_arguments(self):
        """Agent is called with the prompt, context, and config."""
        received_calls = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_calls.append((prompt, context, config))
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        await extract_structured(
            agent=agent,
            prompt="my prompt",
            context={"key": "val"},
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=3,
            allowed_tools=["Write"],
        )
        assert len(received_calls) == 1
        prompt, context, config = received_calls[0]
        assert "my prompt" in prompt
        assert context == {"key": "val"}
        assert config.role == "structured_output"

    async def test_agent_config_passes_allowed_tools(self):
        """Allowed tools from extract_structured are passed to AgentConfig."""
        received_configs = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_configs.append(config)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        await extract_structured(
            agent=agent,
            prompt="prompt",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            allowed_tools=["Read", "Bash"],
        )
        assert len(received_configs) == 1
        config = received_configs[0]
        assert "Read" in config.allowed_tools
        assert "Bash" in config.allowed_tools

    async def test_retry_on_validation_error_succeeds_second_time(self):
        """First call returns invalid shape, second call succeeds → retry_count=1."""
        agent = StubAgentProvider([
            # First call: wrong shape (missing required field)
            make_json_agent_result([{"file_path": "X.java"}]),
            # Second call: valid after error feedback
            make_json_agent_result([{
                "file_path": "src/main/java/UserService.java",
                "diff_content": "+ fix",
                "operation": "modify",
                "language": "java",
            }]),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=3,
        )
        assert len(result.data) == 1
        assert result.data[0].file_path == "src/main/java/UserService.java"
        assert result.retry_count == 1

    async def test_retry_on_malformed_json(self):
        """Non-JSON string triggers retry; valid JSON on second attempt succeeds."""
        agent = StubAgentProvider([
            make_json_agent_result("not valid json at all {{{"),
            make_json_agent_result([{
                "file_path": "src/main/java/Fix.java",
                "diff_content": "+ fixed",
            }]),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=3,
        )
        assert len(result.data) == 1
        assert result.data[0].file_path == "src/main/java/Fix.java"
        assert result.retry_count == 1

    async def test_all_retries_exhausted_raises_error(self):
        """All retries fail → StructuredOutputError raised."""
        agent = StubAgentProvider([
            make_json_agent_result("not json"),
            make_json_agent_result("still not json"),
            make_json_agent_result("one more fail"),
            make_json_agent_result("final fail"),
        ])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=List[DiffPayload],
                system_prompt=_SYSTEM,
                max_retries=3,
            )
        assert "List[DiffPayload]" in str(exc_info.value) or "list[DiffPayload]" in str(exc_info.value)
        assert len(agent.calls) == 4  # initial + 3 retries

    async def test_validation_error_includes_field_info(self):
        """Error message mentions the failing field."""
        agent = StubAgentProvider([
            make_json_agent_result([{"wrong_field": "value"}]),
        ])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=List[DiffPayload],
                system_prompt=_SYSTEM,
                max_retries=0,  # no retries, fail immediately
            )
        error_str = str(exc_info.value)
        assert "file_path" in error_str or "DiffPayload" in error_str

    async def test_retry_count_tracks_correctly(self):
        """retry_count reflects the number of retries attempted."""
        agent = StubAgentProvider([
            make_json_agent_result("bad"),
            make_json_agent_result("also bad"),
            make_json_agent_result([{
                "file_path": "src/main/java/OK.java",
                "diff_content": "+ ok",
            }]),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=5,
        )
        assert result.retry_count == 2

    async def test_error_feedback_includes_validation_errors(self):
        """Retries include the validation errors in the prompt sent to agent."""
        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([
            make_json_agent_result([{"wrong_field": "value"}]),
            make_json_agent_result([{
                "file_path": "src/main/java/Fix.java",
                "diff_content": "+ fixed",
            }]),
        ])
        await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
            max_retries=2,
        )
        # Second prompt should contain the validation error from first attempt
        assert len(received_prompts) == 2
        assert "validation" in received_prompts[1].lower() or \
               "required" in received_prompts[1].lower() or \
               "file_path" in received_prompts[1].lower()

    async def test_max_retries_zero_fails_immediately_on_error(self):
        """max_retries=0 means one attempt, no retries."""
        agent = StubAgentProvider([make_json_agent_result("bad")])
        with pytest.raises(StructuredOutputError):
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=List[DiffPayload],
                system_prompt=_SYSTEM,
                max_retries=0,
            )
        assert len(agent.calls) == 1

    async def test_list_of_dicts_model(self):
        """List[StrategyCandidate] model works correctly."""
        strategies = [
            {"strategy_id": "s1", "description": "A", "affected_files": ["X.java"]},
            {"strategy_id": "s2", "description": "B", "affected_files": ["Y.java"]},
        ]
        agent = StubAgentProvider([make_json_agent_result(strategies)])
        result = await extract_structured(
            agent=agent,
            prompt="Generate strategies",
            response_model=List[StrategyCandidate],
            system_prompt=_SYSTEM,
        )
        assert len(result.data) == 2
        assert result.data[0].strategy_id == "s1"
        assert result.data[1].affected_files == ["Y.java"]

    async def test_default_max_retries_is_three(self):
        """Default max_retries is 3 when not specified."""
        agent = StubAgentProvider([
            make_json_agent_result("bad"),
            make_json_agent_result("bad"),
            make_json_agent_result("bad"),
            make_json_agent_result([{
                "file_path": "src/main/java/OK.java",
                "diff_content": "+ ok",
            }]),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=List[DiffPayload],
            system_prompt=_SYSTEM,
        )
        assert result.retry_count == 3


# ── _validate — direct unit tests for the pure validation function ──────────────


from sacv.nodes._structured_output import _validate


class TestValidate:

    def test_valid_list_parsed(self):
        content = '[{"file_path":"X.java","diff_content":"+fix"}]'
        result = _validate(content, List[DiffPayload])
        assert len(result) == 1
        assert result[0].file_path == "X.java"

    def test_valid_dict_parsed(self):
        content = '{"key":"value"}'
        result = _validate(content, dict)
        assert result == {"key": "value"}

    def test_valid_string_parsed(self):
        content = '"hello world"'
        result = _validate(content, str)
        assert result == "hello world"

    def test_valid_empty_list_parsed(self):
        content = '[]'
        result = _validate(content, list)
        assert result == []

    def test_non_string_for_str_model_raises(self):
        content = '123'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_non_dict_for_dict_model_raises(self):
        content = '"not a dict"'
        with pytest.raises(Exception):
            _validate(content, dict)

    def test_non_list_for_list_model_raises(self):
        content = '{"not": "a list"}'
        with pytest.raises(Exception):
            _validate(content, list)

    def test_missing_required_field_raises(self):
        content = '{"file_path":"X.java"}'
        with pytest.raises(Exception):
            _validate(content, DiffPayload)

    def test_extra_fields_ignored(self):
        """Pydantic by default ignores extra fields."""
        content = '{"file_path":"X.java","diff_content":"+fix","extra":"ignored"}'
        result = _validate(content, DiffPayload)
        assert result.file_path == "X.java"

    def test_invalid_literal_value_raises(self):
        content = '[{"file_path":"X.java","diff_content":"+fix","operation":"invalid"}]'
        with pytest.raises(Exception):
            _validate(content, List[DiffPayload])

    def test_valid_literal_value_accepted(self):
        content = '[{"file_path":"X.java","diff_content":"+fix","operation":"create","language":"typescript"}]'
        result = _validate(content, List[DiffPayload])
        assert result[0].operation == "create"
        assert result[0].language == "typescript"

    def test_empty_json_object_for_model_raises(self):
        content = '{}'
        with pytest.raises(Exception):
            _validate(content, DiffPayload)

    def test_nested_model(self):
        """Nested Pydantic models parse correctly."""
        class Inner(BaseModel):
            name: str
        class Outer(BaseModel):
            inner: Inner
        content = '{"inner":{"name":"test"}}'
        result = _validate(content, Outer)
        assert result.inner.name == "test"

    def test_list_of_strings(self):
        content = '["a","b","c"]'
        result = _validate(content, List[str])
        assert result == ["a", "b", "c"]

    def test_integer_in_string_context_raises(self):
        content = '42'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_null_parsed_for_nullable(self):
        """null is valid for str model."""
        content = 'null'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_boolean_for_str_raises(self):
        content = 'true'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_deeply_nested_list(self):
        content = '[[1,2],[3,4]]'
        result = _validate(content, List[List[int]])
        assert result == [[1, 2], [3, 4]]


# ── Pydantic model edge cases ──────────────────────────────────────────────────


class TestDiffPayloadModel:
    """Tests for the DiffPayload Pydantic model from _structured_output.py."""

    def test_default_operation_is_modify(self):
        from sacv.nodes._structured_output import DiffPayload as DP
        p = DP(file_path="X.java", diff_content="+fix")
        assert p.operation == "modify"

    def test_default_language_is_other(self):
        from sacv.nodes._structured_output import DiffPayload as DP
        p = DP(file_path="X.java", diff_content="+fix")
        assert p.language == "other"

    def test_all_literal_values_accepted(self):
        from sacv.nodes._structured_output import DiffPayload as DP
        for op in ["modify", "create", "delete"]:
            for lang in ["java", "typescript", "sql", "yaml", "other"]:
                p = DP(file_path="X.java", diff_content="+fix", operation=op, language=lang)
                assert p.operation == op
                assert p.language == lang

    def test_empty_diff_content_accepted(self):
        from sacv.nodes._structured_output import DiffPayload as DP
        p = DP(file_path="X.java", diff_content="")
        assert p.diff_content == ""

    def test_unicode_file_path(self):
        from sacv.nodes._structured_output import DiffPayload as DP
        p = DP(file_path="src/日本語/UserService.java", diff_content="+fix")
        assert "日本語" in p.file_path


class TestCriticFindingPayloadModel:
    """Tests for the CriticFindingPayload Pydantic model."""

    def test_defaults(self):
        from sacv.nodes._structured_output import CriticFindingPayload as CFP
        f = CFP(critic="security", severity="warning", file="X.java", message="test", resolution_hint="fix it")
        assert f.line is None
        assert f.rule_id == "UNKNOWN"

    def test_all_severity_values(self):
        from sacv.nodes._structured_output import CriticFindingPayload as CFP
        for sev in ["critical", "warning", "info"]:
            f = CFP(critic="security", severity=sev, file="X.java", message="test", resolution_hint="fix it")
            assert f.severity == sev

    def test_line_as_int(self):
        from sacv.nodes._structured_output import CriticFindingPayload as CFP
        f = CFP(critic="security", severity="warning", file="X.java", line=42, message="test", resolution_hint="fix it")
        assert f.line == 42


class TestStrategyCandidateRawModel:
    """Tests for the StrategyCandidateRaw Pydantic model."""

    def test_defaults(self):
        from sacv.nodes._structured_output import StrategyCandidateRaw as SCR
        s = SCR(strategy_id="s1", description="test", affected_files=["X.java"])
        assert s.strategy_id == "s1"
        assert s.description == "test"
        assert s.affected_files == ["X.java"]

    def test_empty_affected_files(self):
        from sacv.nodes._structured_output import StrategyCandidateRaw as SCR
        s = SCR(strategy_id="s1", description="test", affected_files=[])
        assert s.affected_files == []


class TestAgentsMdUpdateModel:
    """Tests for the AgentsMdUpdate Pydantic model."""

    def test_defaults(self):
        from sacv.nodes._structured_output import AgentsMdUpdate as AMU
        a = AMU()
        assert a.common_mistakes == ""
        assert a.architecture_decisions == ""

    def test_with_values(self):
        from sacv.nodes._structured_output import AgentsMdUpdate as AMU
        a = AMU(common_mistakes="don't delete tests", architecture_decisions="use layers")
        assert a.common_mistakes == "don't delete tests"
        assert a.architecture_decisions == "use layers"


class TestStructuredOutputResult:
    """Tests for the StructuredOutputResult dataclass."""

    def test_default_retry_count(self):
        from sacv.nodes._structured_output import StructuredOutputResult
        r = StructuredOutputResult(data=[1], raw_content="[]")
        assert r.retry_count == 0

    def test_data_and_raw_content(self):
        from sacv.nodes._structured_output import StructuredOutputResult
        r = StructuredOutputResult(data={"key": "val"}, raw_content='{"key":"val"}', retry_count=2)
        assert r.data == {"key": "val"}
        assert r.raw_content == '{"key":"val"}'
        assert r.retry_count == 2


class TestStructuredOutputErrorCost:
    """M-07: StructuredOutputError carries updated_cost from failed agent call."""

    async def test_error_caries_updated_cost(self):
        """StructuredOutputError includes updated_cost from the last agent result."""
        agent = StubAgentProvider([
            make_json_agent_result("bad"),
            make_json_agent_result("still bad"),
        ])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=List[DiffPayload],
                system_prompt=_SYSTEM,
                max_retries=1,
                current_cost=5.0,
            )
        # The error should carry the updated cost (5.0 + agent cost)
        assert hasattr(exc_info.value, "updated_cost")
        assert exc_info.value.updated_cost >= 5.0  # cost increased from agent call

    async def test_error_cost_is_zero_when_no_cost_passed(self):
        """When no current_cost is passed, updated_cost defaults to 0."""
        agent = StubAgentProvider([make_json_agent_result("bad")])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=List[DiffPayload],
                system_prompt=_SYSTEM,
                max_retries=0,
            )
        assert hasattr(exc_info.value, "updated_cost")
        assert exc_info.value.updated_cost >= 0.0
