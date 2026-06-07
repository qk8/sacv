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
