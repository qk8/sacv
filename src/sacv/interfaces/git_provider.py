from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

class GitProvider(ABC):
    @property
    @abstractmethod
    def repo_root(self) -> Path:
        """Return the absolute root of the git repository."""
        ...

    @abstractmethod
    def create_branch(self, name: str, from_ref: str = "HEAD") -> str: ...

    @abstractmethod
    def commit(self, message: str, add_all: bool = True) -> str:
        """Stage all changes (if add_all) and create a commit. Returns the new SHA."""
        ...

    @abstractmethod
    def checkout(self, branch_name: str) -> None: ...
    @abstractmethod
    def stash(self, message: str) -> str: ...
    @abstractmethod
    def stash_pop(self, ref: str) -> None: ...

    @abstractmethod
    def stash_drop(self, ref: str) -> None: ...
    @abstractmethod
    def reset_hard(self, ref: str) -> None: ...
    @abstractmethod
    def get_last_green_commit(self) -> str: ...
    @abstractmethod
    def record_green_commit(self, sha: str) -> None: ...
    @abstractmethod
    def current_branch(self) -> str: ...
    @abstractmethod
    def uncommitted_files(self) -> list[str]: ...

    # ── Speculative branch isolation ──────────────────────────────────────

    @abstractmethod
    def create_worktree(self, branch_name: str, worktree_path: Path) -> Path: ...

    @abstractmethod
    def remove_worktree(self, worktree_path: Path) -> None: ...

    # ── Selective staging & SHA queries ────────────────────────────────────

    @abstractmethod
    def stage_file(self, path: str) -> None: ...

    @abstractmethod
    def head_sha(self) -> str: ...

    @abstractmethod
    def delete_branch(self, name: str, force: bool = False) -> None:
        """Delete a local git branch. Raises RuntimeError if branch does not exist
        and force=False."""
        ...

    @abstractmethod
    def list_branches(self, pattern: str = "agent-*") -> list[str]:
        """Return branch names matching the given glob pattern."""
        ...
