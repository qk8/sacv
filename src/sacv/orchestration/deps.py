"""
orchestration/deps.py
=====================
Shared NodeDeps dataclass for LangGraph node dependency injection.

Extracted from graph.py to eliminate circular import risk:
graph.py imports from speculative_branch, and speculative_branch's
_evaluate_branch imported NodeDeps from graph.py inside the function body.
Moving NodeDeps here breaks the cycle — both modules import from deps.py
independently with no cross-dependency.

See BUG-013 for full root-cause analysis.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sacv.interfaces.agent_provider      import AgentProvider
from sacv.interfaces.memory_provider     import MemoryProvider
from sacv.interfaces.code_graph_provider import CodeGraphProvider
from sacv.interfaces.cross_domain_provider import CrossDomainProvider
from sacv.interfaces.git_provider        import GitProvider
from sacv.interfaces.sandbox_provider    import SandboxProvider
from sacv.interfaces.diff_provider       import DiffProvider
from sacv.orchestration.config           import WorkflowConfig


@dataclass
class NodeDeps:
    agent:        AgentProvider
    memory:       MemoryProvider
    code_graph:   CodeGraphProvider
    cross_domain: CrossDomainProvider
    git:          GitProvider
    sandbox:      SandboxProvider
    diff:         DiffProvider
    config:       WorkflowConfig = field(default_factory=WorkflowConfig)
    # Per-instance semaphore — prevents module-level sharing across
    # parallel graph invocations and pytest workers (BUG-012 fix).
    critic_semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.critic_semaphore = asyncio.Semaphore(
            self.config.max_parallel_critics,
        )
