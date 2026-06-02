"""
nodes/critics/base.py
=====================
Shared execution logic for the critic fan-out.

All three critics (Security, Style, Consistency) share:
- A common AgentConfig template
- The same output schema (list[CriticFinding])
- An asyncio.Semaphore to enforce the ≤2 concurrent executions resource limit

Each critic node runs concurrently via asyncio.gather (inside all_critics node
or _evaluate_branch in speculative_branch), and findings are merged by the
_merge_lists reducer in WorkflowState.
"""
from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from typing import Literal, TYPE_CHECKING

import structlog

from sacv.orchestration.state import CriticFinding, WorkflowPhase
from sacv.interfaces.agent_provider import AgentConfig

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_CRITIC_BASE_SYSTEM = """\
You are a {role} code reviewer. Analyse only the diff provided.
Output ONLY a JSON array of finding objects. Each object must have:
  "critic":          "{critic_name}"
  "severity":        "critical" | "warning" | "info"
  "file":            file path (string)
  "line":            line number (int or null)
  "rule_id":         short rule identifier (string)
  "message":         clear description of the issue
  "resolution_hint": concrete fix instruction

If no issues found, output an empty array: []
No explanation. No markdown. Only the JSON array.
"""


async def _run_critic(
    role:        str,
    critic_name: Literal["security", "style", "consistency"],
    extra_rules: str,
    state:       "WorkflowState",
    deps:        "NodeDeps",
) -> list[CriticFinding]:
    """Shared critic execution logic — acquires semaphore before running."""
    proposal = state.get("diff_proposal")
    if not proposal:
        return []

    diff_text = "\n\n".join(
        f"--- {d['file_path']} ({d['operation']}) ---\n{d['diff_content']}"
        for d in proposal["diffs"]
    )

    prompt = (
        f"Review the following diff:\n\n{diff_text}\n\n"
        f"Additional context — module type: {state['module_type']}, "
        f"mode: {state['project_mode']}."
    )

    async with deps.critic_semaphore:
        result = await deps.agent.run_task(
            prompt=prompt,
            context={"proposal": proposal},
            config=AgentConfig(
                role=critic_name,
                system_prompt=_CRITIC_BASE_SYSTEM.format(
                    role=role, critic_name=critic_name
                ) + f"\n\nAdditional rules:\n{extra_rules}",
                max_turns=1,
                allowed_tools=[],
            ),
        )

    try:
        raw: list[dict] = json.loads(result.content)
    except (json.JSONDecodeError, ValueError):
        log.error(f"{critic_name}.parse_error", content=result.content[:200])
        return []

    findings: list[CriticFinding] = [
        CriticFinding(
            critic=critic_name,
            severity=r.get("severity", "warning"),
            file=r.get("file", "unknown"),
            line=r.get("line"),
            rule_id=r.get("rule_id", "UNKNOWN"),
            message=r.get("message", ""),
            resolution_hint=r.get("resolution_hint", ""),
        )
        for r in raw
        if isinstance(r, dict)
    ]

    log.info(
        f"{critic_name}.complete",
        findings=len(findings),
        critical=sum(1 for f in findings if f["severity"] == "critical"),
    )
    return findings


