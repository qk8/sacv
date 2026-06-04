"""
testing/stub_providers.py
=========================
Stub implementations of every ABC interface.
Used in all unit and integration tests — zero infrastructure required.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from sacv.interfaces.agent_provider      import AgentProvider, AgentConfig, AgentResult
from sacv.interfaces.memory_provider     import (
    MemoryProvider, EpisodicEvent, ProceduralConstraint,
)
from sacv.orchestration.state import LessonLearned
from sacv.interfaces.code_graph_provider import (
    CodeGraphProvider, BlastRadiusMap, CallGraph,
)
from sacv.interfaces.cross_domain_provider import CrossDomainProvider
from sacv.interfaces.diff_provider       import (
    DiffProvider, UnifiedDiff, DiffResult, DiffValidationError,
)
from sacv.interfaces.git_provider        import GitProvider
from sacv.interfaces.sandbox_provider    import SandboxProvider, SandboxHandle, ExecResult


# ── Agent ─────────────────────────────────────────────────────────────────────

class StubAgentProvider(AgentProvider):
    """Pops pre-configured AgentResult objects in FIFO order."""

    def __init__(self, responses: list[AgentResult] | None = None) -> None:
        self._queue: deque[AgentResult] = deque(responses or [])
        self.calls:  list[tuple[str, str]] = []   # (role, prompt[:80])

    def enqueue(self, result: AgentResult) -> None:
        self._queue.append(result)

    async def run_task(
        self, prompt: str, context: dict, config: AgentConfig
    ) -> AgentResult:
        self.calls.append((config.role, prompt[:80]))
        if not self._queue:
            raise AssertionError(
                f"StubAgentProvider exhausted. "
                f"Role={config.role!r}, prompt={prompt[:60]!r}"
            )
        return self._queue.popleft()

    async def create_subagent(self, config: AgentConfig) -> "StubAgentProvider":
        return StubAgentProvider(list(self._queue))


def make_json_agent_result(content: object, tokens: int = 10) -> AgentResult:
    """Helper: wrap a JSON-serialisable object as an AgentResult."""
    import json
    return AgentResult(
        content=json.dumps(content),
        tool_calls=[],
        finish_reason="stop",
        input_tokens=tokens,
        output_tokens=tokens,
    )


# ── Memory ────────────────────────────────────────────────────────────────────

class StubMemoryProvider(MemoryProvider):
    """In-memory store with full introspection for assertions."""

    def __init__(
        self,
        procedural: list[ProceduralConstraint] | None = None,
    ) -> None:
        self._constraints         = procedural or []
        self.stored_events:        list[EpisodicEvent]         = []
        self.consolidated_sessions: list[str]                  = []
        self.purged_sessions:       list[str]                  = []

    async def store_episodic(self, event: EpisodicEvent) -> None:
        self.stored_events.append(event)

    async def retrieve_procedural(
        self, context_tags: list[str]
    ) -> list[ProceduralConstraint]:
        return self._constraints

    async def purge_noise(self, session_id: str) -> None:
        self.purged_sessions.append(session_id)


# ── CodeGraph ─────────────────────────────────────────────────────────────────

class StubCodeGraphProvider(CodeGraphProvider):
    def __init__(
        self,
        blast:    BlastRadiusMap | None = None,
        graph:    CallGraph      | None = None,
        subgraph: dict           | None = None,
    ) -> None:
        self._blast   = blast    or BlastRadiusMap([], [], 0, [], [], 0.0)
        self._graph   = graph    or CallGraph(".", [], [])
        self._subgraph = subgraph or {}

    async def get_blast_radius(self, file_paths: list[str]) -> BlastRadiusMap:
        return self._blast

    async def get_call_graph(self, entry_points: list[str]) -> CallGraph:
        return self._graph

    async def get_dependency_subgraph(self, scope: list[str]) -> dict:
        return self._subgraph


# ── CrossDomain ───────────────────────────────────────────────────────────────

class StubCrossDomainProvider(CrossDomainProvider):
    async def map_code_to_schema(self, entity_names: list[str]) -> dict:
        return {"entities": entity_names}

    async def get_arch_alignment(self, module_paths: list[str]) -> dict:
        return {"aligned": True}

    async def get_sql_impact(self, changed_files: list[str]) -> dict:
        return {"affected_tables": []}


# ── Diff ──────────────────────────────────────────────────────────────────────

class StubDiffProvider(DiffProvider):
    """Accepts all diffs; optionally pre-configured to return validation errors."""

    def __init__(
        self,
        validation_errors: list[DiffValidationError] | None = None,
        apply_success:     bool = True,
    ) -> None:
        self._errors  = validation_errors or []
        self._success = apply_success
        self.applied: list[list[UnifiedDiff]] = []

    async def validate_no_full_overwrite(
        self, diffs: list[UnifiedDiff]
    ) -> list[DiffValidationError]:
        return self._errors

    async def apply_diffs(self, diffs: list[UnifiedDiff]) -> DiffResult:
        if self._success:
            self.applied.append(list(diffs))
        return DiffResult(
            success=self._success,
            applied_files=[d.file_path for d in diffs] if self._success else [],
            conflicts=[] if self._success else [{"file": d.file_path} for d in diffs],
            validation_errors=[],
        )

    async def generate_ast_diff(
        self, original: str, modified: str, language: str
    ) -> UnifiedDiff:
        return UnifiedDiff(
            file_path="stub.java",
            diff_content=f"--- a\n+++ b\n@@ -1 +1 @@\n-{original[:20]}\n+{modified[:20]}",
            operation="modify",
            language=language,
        )


# ── Git ───────────────────────────────────────────────────────────────────────

class StubGitProvider(GitProvider):
    """Records all calls; never touches the filesystem."""

    def __init__(
        self,
        current_branch_name: str = "main",
        green_sha:           str = "abc1234deadbeef",
    ) -> None:
        self._branch = current_branch_name
        self._green  = green_sha
        self._branches: set[str] = {current_branch_name}
        self.calls:  list[tuple[str, ...]] = []

    @property
    def repo_root(self) -> Path:
        return Path.cwd()

    def _rec(self, *args: str) -> None:
        self.calls.append(args)

    def create_branch(self, name: str, from_ref: str = "HEAD") -> str:
        self._rec("create_branch", name, from_ref)
        self._branch = name
        self._branches.add(name)
        return name

    def commit(self, message: str, add_all: bool = True) -> str:
        self._rec("commit", message)
        return "deadbeef00000000"

    def checkout(self, branch_name: str) -> None:
        self._rec("checkout", branch_name)
        self._branch = branch_name
        self._branches.add(branch_name)

    def stash(self, message: str) -> str:
        self._rec("stash", message)
        return "stash@{0}"

    def stash_pop(self, ref: str) -> None:
        self._rec("stash_pop", ref)

    def stash_drop(self, ref: str) -> None:
        self._rec("stash_drop", ref)

    def reset_hard(self, ref: str) -> None:
        self._rec("reset_hard", ref)

    def get_last_green_commit(self) -> str:
        return self._green

    def record_green_commit(self, sha: str) -> None:
        self._rec("record_green", sha)
        self._green = sha

    def current_branch(self) -> str:
        return self._branch

    def uncommitted_files(self) -> list[str]:
        return []

    def create_worktree(self, branch_name: str, worktree_path: Path) -> Path:
        self._rec("create_worktree", branch_name, str(worktree_path))
        return Path(worktree_path)

    def remove_worktree(self, worktree_path: Path) -> None:
        self._rec("remove_worktree", str(worktree_path))

    def stage_file(self, path: str) -> None:
        self._rec("stage_file", path)

    def head_sha(self) -> str:
        return self._green

    def delete_branch(self, name: str, force: bool = False) -> None:
        self._rec("delete_branch", name, str(force))
        self._branches.discard(name)

    def list_branches(self, pattern: str = "agent-*") -> list[str]:
        self._rec("list_branches", pattern)
        return list(self._branches)


# ── Sandbox ───────────────────────────────────────────────────────────────────

class StubSandboxProvider(SandboxProvider):
    """
    Returns pre-configured ExecResult objects per command pattern.
    Falls back to a default result (exit_code=0) if no match.
    """

    def __init__(
        self,
        default_exit_code: int  = 0,
        default_stdout:    str  = "",
        default_stderr:    str  = "",
    ) -> None:
        self._default = ExecResult(default_exit_code, default_stdout, default_stderr, 10)
        self._overrides: dict[str, ExecResult] = {}
        self.exec_calls: list[str] = []

    def register(self, command_fragment: str, result: ExecResult) -> None:
        """Register a result for any command containing ``command_fragment``."""
        self._overrides[command_fragment] = result

    async def warm_container(self) -> SandboxHandle:
        return SandboxHandle("stub-container-id", "/workspace", warm=True)

    async def exec_in_container(
        self,
        handle:  SandboxHandle,
        command: str,
        env:     dict[str, str] | None = None,
        timeout: int = 120,
    ) -> ExecResult:
        self.exec_calls.append(command)
        for fragment, result in self._overrides.items():
            if fragment in command:
                return result
        return self._default

    async def destroy_container(self, handle: SandboxHandle) -> None:
        pass

    def create_isolated_instance(self, host_mount: str) -> "StubSandboxProvider":
        return StubSandboxProvider(
            default_exit_code=self._default.exit_code,
            default_stdout=self._default.stdout,
            default_stderr=self._default.stderr,
        )
