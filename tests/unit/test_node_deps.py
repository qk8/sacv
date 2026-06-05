"""
tests/unit/test_node_deps.py
=============================
Unit tests for NodeDeps — dependency injection dataclass.

Tests cover:
1. Default construction creates semaphore
2. with_git_and_sandbox creates child with shared semaphore
3. Child has replaced git/sandbox/diff providers
4. Child preserves config from parent
5. Child preserves other providers from parent
6. _semaphore_injected flag set correctly
"""
import asyncio
import pytest
from unittest.mock import MagicMock

from sacv.orchestration.deps import NodeDeps
from sacv.orchestration.config import WorkflowConfig
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult


def _base_deps():
    agent = StubAgentProvider([make_json_agent_result("ok")])
    memory = StubMemoryProvider()
    code_graph = StubCodeGraphProvider()
    cross_domain = StubCrossDomainProvider()
    git = StubGitProvider()
    sandbox = StubSandboxProvider()
    diff = StubDiffProvider()
    config = WorkflowConfig()

    return NodeDeps(
        agent=agent, memory=memory, code_graph=code_graph,
        cross_domain=cross_domain, git=git, sandbox=sandbox,
        diff=diff, config=config,
    )


class TestNodeDepsConstruction:

    def test_default_construction_creates_semaphore(self):
        deps = _base_deps()
        assert isinstance(deps.critic_semaphore, asyncio.Semaphore)

    def test_semaphore_value_from_config(self):
        cfg = WorkflowConfig(max_parallel_critics=3)
        from sacv.orchestration.deps import NodeDeps
        from sacv.testing.stub_providers import (
            StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
            StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
            StubSandboxProvider,
        )
        agent = StubAgentProvider([make_json_agent_result("ok")])
        deps = NodeDeps(
            agent=agent, memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(), sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(), config=cfg,
        )
        # Semaphore starts at max_parallel_critics, can be acquired that many times
        async def drain():
            for _ in range(3):
                await deps.critic_semaphore.acquire()
            return deps.critic_semaphore._value
        value = asyncio.run(drain())
        assert value == 0  # all 3 acquired

    def test_semaphore_not_injected_initially_false(self):
        deps = _base_deps()
        assert deps._semaphore_injected is False

    def test_semaphore_injected_false_after_normal_init(self):
        """Normal construction leaves _semaphore_injected=False.

        __post_init__ creates the semaphore but does NOT set the flag.
        Only with_git_and_sandbox sets _semaphore_injected=True.
        """
        deps = _base_deps()
        assert deps._semaphore_injected is False


class TestWithGitAndSandbox:

    def test_child_has_replaced_git(self):
        deps = _base_deps()
        new_git = StubGitProvider(current_branch_name="feature-branch")
        child = deps.with_git_and_sandbox(
            git=new_git,
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.git is new_git
        assert child.git is not deps.git

    def test_child_has_replaced_sandbox(self):
        deps = _base_deps()
        new_sandbox = StubSandboxProvider()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=new_sandbox,
            diff=StubDiffProvider(),
        )
        assert child.sandbox is new_sandbox
        assert child.sandbox is not deps.sandbox

    def test_child_has_replaced_diff(self):
        deps = _base_deps()
        new_diff = StubDiffProvider()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=new_diff,
        )
        assert child.diff is new_diff
        assert child.diff is not deps.diff

    def test_child_shares_semaphore(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.critic_semaphore is deps.critic_semaphore

    def test_child_semaphore_injected(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child._semaphore_injected is True

    def test_child_preserves_config(self):
        cfg = WorkflowConfig(max_blast_files=100, min_strategy_score=0.7)
        from sacv.orchestration.deps import NodeDeps
        deps = NodeDeps(
            agent=StubAgentProvider([make_json_agent_result("ok")]),
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
            config=cfg,
        )
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.config.max_blast_files == 100
        assert child.config.min_strategy_score == 0.7

    def test_child_preserves_agent(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.agent is deps.agent

    def test_child_preserves_memory(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.memory is deps.memory

    def test_child_preserves_code_graph(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.code_graph is deps.code_graph

    def test_child_preserves_cross_domain(self):
        deps = _base_deps()
        child = deps.with_git_and_sandbox(
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        assert child.cross_domain is deps.cross_domain

    def test_repo_root_delegates_to_git(self):
        deps = _base_deps()
        # repo_root property delegates to git.repo_root
        assert deps.repo_root is not None

    def test_nested_branch_scoped_deps(self):
        """Chaining with_git_and_sandbox preserves shared semaphore."""
        deps = _base_deps()
        child1 = deps.with_git_and_sandbox(
            git=StubGitProvider(current_branch_name="branch-1"),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        child2 = child1.with_git_and_sandbox(
            git=StubGitProvider(current_branch_name="branch-2"),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
        )
        # All share the same semaphore
        assert child1.critic_semaphore is deps.critic_semaphore
        assert child2.critic_semaphore is deps.critic_semaphore
        # Each has its own git
        assert child1.git is not child2.git
        assert child1.git.calls is not child2.git.calls
