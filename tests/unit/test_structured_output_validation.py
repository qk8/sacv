"""
tests/unit/test_structured_output_validation.py
=================================================
Unit tests for _structured_output.py edge cases not covered by existing tests.

Tests cover:
1. BaseModel with non-dict JSON input (e.g., list or string) → validation error
2. workflow_config is None → no cost tracking, still returns result
3. _validate with list parsed for BaseModel → raises ValidationError
4. _validate with string parsed for BaseModel → raises ValidationError
5. _validate with int parsed for BaseModel → raises ValidationError
"""
from __future__ import annotations

import pytest

from pydantic import BaseModel

from sacv.interfaces.agent_provider import AgentConfig, AgentResult
from sacv.testing.stub_providers import StubAgentProvider, make_json_agent_result
from sacv.nodes._structured_output import (
    extract_structured,
    _validate,
    StructuredOutputError,
    DiffPayload,
)


# ── _validate edge cases for BaseModel with non-dict input ────────────────────

class TestValidateBaseModelNonDictInput:
    """Tests for _validate when BaseModel is expected but non-dict JSON is provided."""

    def test_list_parsed_for_base_model_raises(self):
        """JSON array passed when BaseModel (not List[Model]) expected → raises."""
        content = '[{"file_path":"X.java","diff_content":"+fix"}]'
        with pytest.raises(Exception) as exc_info:
            _validate(content, DiffPayload)
        # Should be a ValidationError-like exception
        assert "DiffPayload" in str(exc_info.value) or "dict_type" in str(exc_info.value)

    def test_string_parsed_for_base_model_raises(self):
        """JSON string passed when BaseModel expected → raises."""
        content = '"just a string"'
        with pytest.raises(Exception) as exc_info:
            _validate(content, DiffPayload)
        assert "DiffPayload" in str(exc_info.value) or "dict_type" in str(exc_info.value)

    def test_int_parsed_for_base_model_raises(self):
        """JSON integer passed when BaseModel expected → raises."""
        content = '42'
        with pytest.raises(Exception) as exc_info:
            _validate(content, DiffPayload)
        assert "DiffPayload" in str(exc_info.value) or "dict_type" in str(exc_info.value)

    def test_null_parsed_for_base_model_raises(self):
        """JSON null passed when BaseModel expected → raises."""
        content = 'null'
        with pytest.raises(Exception) as exc_info:
            _validate(content, DiffPayload)
        assert "DiffPayload" in str(exc_info.value) or "dict_type" in str(exc_info.value)

    def test_boolean_parsed_for_base_model_raises(self):
        """JSON boolean passed when BaseModel expected → raises."""
        content = 'true'
        with pytest.raises(Exception) as exc_info:
            _validate(content, DiffPayload)
        assert "DiffPayload" in str(exc_info.value) or "dict_type" in str(exc_info.value)

    def test_nested_base_model_with_list_raises(self):
        """Nested BaseModel with list input → raises."""
        class Inner(BaseModel):
            name: str
        class Outer(BaseModel):
            inner: Inner

        content = '[{"name":"test"}]'
        with pytest.raises(Exception) as exc_info:
            _validate(content, Outer)
        assert "Outer" in str(exc_info.value) or "dict_type" in str(exc_info.value)


# ── workflow_config is None path ──────────────────────────────────────────────

@pytest.mark.asyncio
class TestExtractStructuredNoConfig:
    """Tests for extract_structured when workflow_config is None (no cost tracking)."""

    async def test_no_config_skips_cost_tracking(self):
        """workflow_config=None → cost tracking is skipped, result still returned."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "X.java",
                "diff_content": "+fix",
            }])
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=list[DiffPayload],
            system_prompt="You are a diff agent.",
            max_retries=3,
            # workflow_config not passed → defaults to None
        )
        assert len(result.data) == 1
        assert result.data[0].file_path == "X.java"

    async def test_no_config_with_current_cost_preserves_cost(self):
        """workflow_config=None + current_cost=5.0 → updated_cost equals current_cost."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "X.java",
                "diff_content": "+fix",
            }])
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=list[DiffPayload],
            system_prompt="You are a diff agent.",
            current_cost=5.0,
            # workflow_config=None (default)
        )
        # Without workflow_config, add_agent_cost is not called, so running_cost
        # stays at current_cost (5.0)
        assert result.updated_cost == 5.0

    async def test_no_config_retry_still_works(self):
        """workflow_config=None → retry loop still works on validation errors."""
        agent = StubAgentProvider([
            make_json_agent_result("not json"),
            make_json_agent_result([{
                "file_path": "X.java",
                "diff_content": "+fix",
            }]),
        ])
        result = await extract_structured(
            agent=agent,
            prompt="Create diff",
            response_model=list[DiffPayload],
            system_prompt="You are a diff agent.",
            max_retries=3,
        )
        assert len(result.data) == 1
        assert result.retry_count == 1

    async def test_no_config_all_retries_fail(self):
        """workflow_config=None → StructuredOutputError raised when all retries fail."""
        agent = StubAgentProvider([
            make_json_agent_result("bad 1"),
            make_json_agent_result("bad 2"),
            make_json_agent_result("bad 3"),
            make_json_agent_result("bad 4"),
        ])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=list[DiffPayload],
                system_prompt="You are a diff agent.",
                max_retries=3,
            )
        assert "List[DiffPayload]" in str(exc_info.value) or "list[DiffPayload]" in str(exc_info.value)
        # updated_cost should reflect the initial current_cost (0.0) since
        # add_agent_cost was never called
        assert exc_info.value.updated_cost == 0.0

    async def test_no_config_with_current_cost_and_failures(self):
        """workflow_config=None + current_cost=3.0 + all failures → error.updated_cost=3.0."""
        agent = StubAgentProvider([
            make_json_agent_result("bad"),
            make_json_agent_result("bad"),
            make_json_agent_result("bad"),
            make_json_agent_result("bad"),
        ])
        with pytest.raises(StructuredOutputError) as exc_info:
            await extract_structured(
                agent=agent,
                prompt="Create diff",
                response_model=list[DiffPayload],
                system_prompt="You are a diff agent.",
                max_retries=3,
                current_cost=3.0,
            )
        assert exc_info.value.updated_cost == 3.0


# ── _validate bare type edge cases ────────────────────────────────────────────

class TestValidateBareTypes:

    def test_dict_model_with_list_raises(self):
        """list JSON passed when dict expected → raises."""
        content = '["a", "b"]'
        with pytest.raises(Exception):
            _validate(content, dict)

    def test_list_model_with_dict_raises(self):
        """dict JSON passed when list expected → raises."""
        content = '{"key": "value"}'
        with pytest.raises(Exception):
            _validate(content, list)

    def test_str_model_with_list_raises(self):
        """list JSON passed when str expected → raises."""
        content = '["a"]'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_str_model_with_dict_raises(self):
        """dict JSON passed when str expected → raises."""
        content = '{"key": "value"}'
        with pytest.raises(Exception):
            _validate(content, str)

    def test_str_model_with_list_json_raises(self):
        """[1,2,3] JSON passed when str expected → raises."""
        content = '[1, 2, 3]'
        with pytest.raises(Exception):
            _validate(content, str)


# ── StructuredOutputError field tests ─────────────────────────────────────────

class TestStructuredOutputErrorFields:

    def test_last_raw_content_stored(self):
        """StructuredOutputError stores last_raw_content."""
        from sacv.nodes._structured_output import StructuredOutputError
        err = StructuredOutputError("fail", last_raw_content="bad output", updated_cost=1.5)
        assert err.last_raw_content == "bad output"
        assert err.updated_cost == 1.5

    def test_empty_raw_content_defaults_to_empty(self):
        """StructuredOutputError defaults last_raw_content to empty string."""
        from sacv.nodes._structured_output import StructuredOutputError
        err = StructuredOutputError("fail")
        assert err.last_raw_content == ""
        assert err.updated_cost == 0.0

    def test_message_contains_description(self):
        """StructuredOutputError message contains the error description."""
        from sacv.nodes._structured_output import StructuredOutputError
        err = StructuredOutputError("custom error message")
        assert "custom error message" in str(err)


# ── StructuredOutputResult field tests ────────────────────────────────────────

class TestStructuredOutputResultFields:

    def test_updated_cost_default(self):
        """StructuredOutputResult defaults updated_cost to 0.0."""
        from sacv.nodes._structured_output import StructuredOutputResult
        r = StructuredOutputResult(data=[1], raw_content="[]")
        assert r.updated_cost == 0.0

    def test_agent_result_stored(self):
        """StructuredOutputResult stores the raw agent_result."""
        from sacv.nodes._structured_output import StructuredOutputResult
        agent_result = AgentResult(
            content="test", tool_calls=[], finish_reason="stop",
            input_tokens=10, output_tokens=20,
        )
        r = StructuredOutputResult(data=[1], raw_content="[]", agent_result=agent_result)
        assert r.agent_result is agent_result
        assert r.agent_result.content == "test"
