"""
nodes/_structured_output.py
============================
Instructor-inspired structured output extraction with automatic retry.

Wraps AgentProvider.run_task() with Pydantic validation and error-feedback
retry — the same pattern instructor uses internally, but adapted for the
Claude Agent SDK's async generator pattern (query()) without adding a
dependency.

Usage:
    result = await extract_structured(
        agent=deps.agent,
        prompt="Create a diff for X",
        response_model=List[DiffPayload],
        system_prompt=_ACTOR_SYSTEM,
        max_retries=3,
        allowed_tools=["Read", "Bash"],
    )
    diffs = result.data  # List[DiffPayload] — type-safe

On validation failure, the Pydantic ValidationError is fed back to the LLM
as a re-prompt, guiding it toward correct output. After max_retries
exhausted, raises StructuredOutputError.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails, InitErrorDetails

from sacv.interfaces.agent_provider import AgentProvider, AgentConfig

T = TypeVar("T", bound=BaseModel | dict[str, object] | str | list[object])


class DiffPayload(BaseModel):
    """Pydantic model for unified diff payload from LLM output."""
    file_path: str
    diff_content: str
    operation: Literal["modify", "create", "delete"] = "modify"
    language: Literal["java", "typescript", "sql", "yaml", "other"] = "other"


class CriticFindingPayload(BaseModel):
    """Pydantic model for critic finding from LLM output."""
    critic: str
    severity: Literal["critical", "warning", "info"]
    file: str
    line: int | None = None
    rule_id: str = "UNKNOWN"
    message: str
    resolution_hint: str


class StrategyCandidateRaw(BaseModel):
    """Pydantic model for strategy candidate from LLM output."""
    strategy_id: str
    description: str
    affected_files: list[str]


class TestFile(BaseModel):
    """Pydantic model for test file from LLM output."""
    file_path: str = ""
    content: str = ""


class AgentsMdUpdate(BaseModel):
    """Pydantic model for AGENTS.md update from LLM output."""
    common_mistakes: str = ""
    architecture_decisions: str = ""


class StructuredOutputError(Exception):
    """Raised when all retries fail to produce valid structured output."""
    pass


@dataclass
class StructuredOutputResult(Generic[T]):
    """Wraps validated LLM output with metadata."""
    data: T
    raw_content: str
    retry_count: int = 0


async def extract_structured(
    agent: AgentProvider,
    prompt: str,
    response_model: type[T],
    system_prompt: str,
    context: dict[str, object] | None = None,
    max_retries: int = 3,
    allowed_tools: list[str] | None = None,
) -> StructuredOutputResult[T]:
    """
    Extract structured data from LLM output with automatic retry.

    On validation failure, feeds the Pydantic ValidationError back to the LLM
    as context for the next attempt — the same mechanism instructor uses
    internally.

    Args:
        agent: The AgentProvider to call.
        prompt: The task prompt for the LLM.
        response_model: A Pydantic BaseModel type, dict, str, or list.
        system_prompt: System prompt for the agent.
        context: Optional context dict passed to the agent.
        max_retries: Maximum retry attempts on validation failure.
        allowed_tools: Tool names the agent may use.

    Returns:
        StructuredOutputResult with validated data and retry metadata.

    Raises:
        StructuredOutputError: All retries exhausted without valid output.
    """
    last_errors: list[str] = []
    accumulated_context: list[str] = []

    for attempt in range(max_retries + 1):
        # Build prompt with accumulated validation errors for retry attempts
        full_prompt = prompt
        if accumulated_context:
            full_prompt = (
                "Previous attempt failed validation. Please fix:\n\n"
                + "\n\n".join(accumulated_context) + "\n\n"
                + f"Produce valid output:\n\n{prompt}"
            )

        config = AgentConfig(
            role="structured_output",
            system_prompt=system_prompt,
            max_turns=1,
            allowed_tools=allowed_tools or [],
        )

        result = await agent.run_task(
            prompt=full_prompt,
            context=context or {},
            config=config,
        )

        # Try to parse and validate
        try:
            parsed = _validate(result.content, response_model)
            return StructuredOutputResult(
                data=parsed,
                raw_content=result.content,
                retry_count=attempt,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_errors.append(f"Attempt {attempt + 1}: {exc}")
            accumulated_context.append(str(exc))

    raise StructuredOutputError(
        f"Failed to extract valid {response_model!r} after "
        f"{max_retries} retries. Errors:\n" + "\n".join(last_errors)
    )


def _validate(content: str, response_model: type[T]) -> T:
    """Parse JSON and validate against the response model."""
    parsed = json.loads(content)
    origin = get_origin(response_model)

    # Handle bare types: str, dict, list
    if response_model is str:
        if not isinstance(parsed, str):
            raise ValidationError.from_exception_data(
                title="validation",
                line_errors=[
                    InitErrorDetails(
                        type="string_type",
                        loc=(),
                        input=parsed,
                    )
                ],
            )
        return parsed  # type: ignore[return-value]

    if response_model is dict:
        if not isinstance(parsed, dict):
            raise ValidationError.from_exception_data(
                title="validation",
                line_errors=[
                    InitErrorDetails(
                        type="dict_type",
                        loc=(),
                        input=parsed,
                    )
                ],
            )
        return parsed  # type: ignore[return-value]

    if response_model is list:
        if not isinstance(parsed, list):
            raise ValidationError.from_exception_data(
                title="validation",
                line_errors=[
                    InitErrorDetails(
                        type="list_type",
                        loc=(),
                        input=parsed,
                    )
                ],
            )
        return parsed  # type: ignore[return-value]

    # Handle generic types like List[X], Dict[K, V] via TypeAdapter
    if origin is not None:
        adapter = TypeAdapter(response_model)
        return adapter.validate_python(parsed)

    # Pydantic BaseModel subclass — use model_validate
    if isinstance(parsed, dict):
        return response_model.model_validate(parsed)  # type: ignore[return-value, attr-defined]

    raise ValidationError.from_exception_data(
        title="validation",
        line_errors=[
            InitErrorDetails(
                type="dict_type",
                loc=(),
                input=parsed,
            )
        ],
    )
