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

from pathlib import Path

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
    # When cloning for speculative branches, the child shares the parent's
    # semaphore to enforce a global LLM call throttle (MED-007).
    critic_semaphore: asyncio.Semaphore = field(init=False)
    _semaphore_injected: bool = field(default=False, init=False)

    @property
    def repo_root(self) -> Path:
        """Canonical repo root. All workflow file I/O must be relative to this."""
        return self.git.repo_root

    def __post_init__(self) -> None:
        if not self._semaphore_injected:
            self.critic_semaphore = asyncio.Semaphore(
                self.config.max_parallel_critics,
            )

    def with_git_and_sandbox(
        self,
        git:       GitProvider,
        sandbox:   SandboxProvider,
        diff:      DiffProvider,
    ) -> "NodeDeps":
        """
        Create a branch-scoped NodeDeps sharing the parent's critic_semaphore.

        This ensures speculative branches collectively respect the global
        LLM call throttle rather than each branch getting its own semaphore.
        """
        child = NodeDeps(
            agent=self.agent,
            memory=self.memory,
            code_graph=self.code_graph,
            cross_domain=self.cross_domain,
            git=git,
            sandbox=sandbox,
            diff=diff,
            config=self.config,
        )
        child.critic_semaphore = self.critic_semaphore
        child._semaphore_injected = True
        return child
