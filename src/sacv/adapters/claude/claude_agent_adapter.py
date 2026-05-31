"""
adapters/claude/claude_agent_adapter.py
=======================================
Concrete implementation of AgentProvider using the Claude Agent SDK
(package: claude-code-sdk, formerly claude-code-sdk, being renamed to
claude-agent-sdk).

Key design decisions:
- Each ``run_task`` call creates a fresh agent context (no cross-call leakage).
- Subagents are created with role-specific system prompts and restricted tool sets.
- Retry logic (tenacity) handles transient API errors without polluting node logic.
- Token usage is tracked and surfaced in AgentResult for budget monitoring.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Claude Agent SDK — package name: claude-code-sdk
# Import path may change as the SDK is renamed to claude-agent-sdk.
# Adjust the import below if the package name changes.
try:
    from claude_code_sdk import (
        query,
        ClaudeCodeOptions,
        AssistantMessage,
        TextBlock,
        ToolUseBlock,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    # Provide stubs so the module is importable even without the SDK installed
    # (e.g. in environments that only run tests via VCR stubs).
    class query:  # type: ignore[no-redef]
        pass
    class ClaudeCodeOptions:  # type: ignore[no-redef]
        pass

from sacv.interfaces.agent_provider import AgentProvider, AgentConfig, AgentResult

log = structlog.get_logger(__name__)

# Default model — pinned to avoid unexpected behaviour on model upgrades.
# Update intentionally when the team decides to upgrade.
_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ClaudeAgentAdapter(AgentProvider):
    """
    Routes ``run_task`` calls through the Claude Agent SDK.

    The SDK's ``query()`` function runs a full agentic loop (reading files,
    running bash commands, editing code) and streams back typed message objects.
    This adapter collects all text blocks and aggregates token usage.
    """

    def __init__(
        self,
        model:   str = _DEFAULT_MODEL,
        cwd:     str | None = None,
        timeout: int = 300,
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "claude-code-sdk is required. Install with: "
                "pip install claude-code-sdk"
            )
        self._model   = model
        self._cwd     = cwd
        self._timeout = timeout

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    async def run_task(
        self,
        prompt:  str,
        context: dict,
        config:  AgentConfig,
    ) -> AgentResult:
        """
        Run a task through the Claude Agent SDK.

        The ``context`` dict is serialised and prepended to the prompt so the
        agent has access to the ContextSkeleton, diff proposals, etc., without
        the graph needing to manage multi-turn history.
        """
        full_prompt = _build_prompt(prompt, context)

        text_parts:    list[str] = []
        tool_calls:    list[dict] = []
        input_tokens   = 0
        output_tokens  = 0

        options = ClaudeCodeOptions(
            model=self._model,
            system_prompt=config.system_prompt,
            max_turns=config.max_turns,
            allowed_tools=config.allowed_tools or [],
            cwd=self._cwd,
        )

        try:
            async with asyncio.timeout(self._timeout):
                async for message in query(
                    prompt=full_prompt,
                    options=options,
                ):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                            elif isinstance(block, ToolUseBlock):
                                tool_calls.append({
                                    "name":  block.name,
                                    "input": block.input,
                                })
                    # Accumulate token usage if available on message
                    usage = getattr(message, "usage", None)
                    if usage:
                        input_tokens  += getattr(usage, "input_tokens",  0)
                        output_tokens += getattr(usage, "output_tokens", 0)

        except asyncio.TimeoutError:
            log.error(
                "claude_adapter.timeout",
                role=config.role,
                timeout=self._timeout,
            )
            raise TimeoutError(
                f"Claude Agent SDK call timed out after {self._timeout}s "
                f"(role={config.role})"
            )

        content = "\n".join(text_parts).strip()

        log.debug(
            "claude_adapter.complete",
            role=config.role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            content_len=len(content),
        )

        return AgentResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def create_subagent(self, config: AgentConfig) -> "ClaudeAgentAdapter":
        """
        Creates a new ClaudeAgentAdapter with the subagent's config.
        The subagent shares the same model and cwd but uses a different
        system prompt and restricted tool set.
        """
        adapter = ClaudeAgentAdapter(
            model=self._model,
            cwd=self._cwd,
            timeout=self._timeout,
        )
        return adapter


def _build_prompt(prompt: str, context: dict) -> str:
    """
    Prepend a compact JSON context block to the prompt when context is non-empty.

    We truncate large context values (e.g. full call-graph) to avoid
    saturating the context window.  The Scout node already produces a
    trimmed skeleton, so this is a safety net.
    """
    if not context:
        return prompt

    safe_context = _truncate_context(context, max_chars=8_000)
    ctx_block    = json.dumps(safe_context, indent=2)
    return f"<context>\n{ctx_block}\n</context>\n\n{prompt}"


def _truncate_context(obj: object, max_chars: int) -> object:
    """Recursively truncate string values and list lengths."""
    if isinstance(obj, str):
        return obj[:max_chars] if len(obj) > max_chars else obj
    if isinstance(obj, list):
        return [_truncate_context(v, max_chars) for v in obj[:20]]
    if isinstance(obj, dict):
        return {k: _truncate_context(v, max_chars) for k, v in list(obj.items())[:30]}
    return obj
