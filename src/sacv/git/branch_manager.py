"""
git/branch_manager.py
=====================
Production implementation of GitProvider using GitPython.

Replaces the following shell scripts entirely:
- branch-manager.sh
- rollback.sh  (merged into ``reset_hard`` / ``restore_last_green``)
- git-operations.sh

All methods are synchronous — git operations are deterministic subprocess
calls with no async benefit.  They are called from async node code via
``asyncio.to_thread`` when needed (the interface is sync by design to
keep the contract simple and testable without an event loop).

The "last green commit" is persisted to ``.workflow/green-sha`` — a plain
text file managed by this class.  This gives the HITL escalation node a
reliable hard-reset target even after multiple speculative branches.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.interfaces.git_provider import GitProvider

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


def sanitize_branch_name(name: str) -> str:
    """Sanitize a string for safe use as a git branch name.

    Replaces characters forbidden or problematic in git branch names
    (per git-check-ref-format) with dashes: / \\ ~ ^ ? * [ ] spaces
    consecutive dots, leading/trailing dots.
    """
    sanitized = re.sub(r"[^\w.\-]", "-", name)
    # Collapse multiple dashes
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    # Strip leading/trailing dashes
    sanitized = sanitized.strip("-")
    # Default to safe fallback
    return sanitized or "branch"



class BranchManager(GitProvider):
    """
    Thin, type-safe wrapper around git CLI subprocess calls.

    Uses ``subprocess.run`` rather than GitPython to avoid the library's
    mutable state model and keep the interface deterministic.
    """

    def __init__(self, repo_root: str | Path = ".") -> None:
        self._root = Path(repo_root).resolve()

    @property
    def repo_root(self) -> Path:
        return self._root

    # ── GitProvider interface ─────────────────────────────────────────────

    def create_branch(self, name: str, from_ref: str = "HEAD") -> str:
        self._run(["git", "checkout", "-b", name, from_ref])
        log.info("git.branch_created", name=name, from_ref=from_ref)
        return name

    def checkout(self, branch_name: str) -> None:
        self._run(["git", "checkout", branch_name])
        log.debug("git.checkout", branch=branch_name)

    def stash(self, message: str) -> str:
        """
        Stash working changes and return the stash SHA (stable ref).
        Returns empty string if there is nothing to stash.
        """
        # Check if there is anything to stash first
        status = self._run(["git", "status", "--porcelain"])
        if not status.stdout.strip():
            log.debug("git.stash_nothing_to_stash")
            return ""  # nothing to stash — return sentinel

        self._run(["git", "stash", "push", "-u", "-m", message])

        # Use the stash SHA for a stable reference that survives subsequent
        # stash operations. 'git rev-parse stash@{0}' returns the object SHA
        # immediately after the push.
        result = self._run(["git", "rev-parse", "stash@{0}"])
        ref = result.stdout.strip()
        log.info("git.stash", message=message, ref=ref[:12])
        return ref

    def stash_pop(self, ref: str) -> None:
        """
        Pop a specific stash entry by its SHA.

        Resolves the SHA to a positional stash@{N} ref before popping,
        because ``git stash pop <SHA>`` is not valid git syntax.
        """
        if not ref:
            log.debug("git.stash_pop_skipped", reason="empty ref")
            return
        positional = self._sha_to_stash_ref(ref)
        if positional:
            self._run(["git", "stash", "pop", positional])

    def stash_drop(self, ref: str) -> None:
        """Drop a specific stash entry by its SHA."""
        if not ref:
            log.debug("git.stash_drop_skipped", reason="empty ref")
            return
        positional = self._sha_to_stash_ref(ref)
        if positional:
            self._run(["git", "stash", "drop", positional])
            log.info("git.stash_pop", ref=positional)
        else:
            log.warning("git.stash_pop_not_found", sha=ref[:12])

    def _sha_to_stash_ref(self, sha: str) -> str | None:
        """
        Look up the stash@{N} positional reference corresponding to a SHA.

        Returns ``None`` if the SHA is not found in the stash list.
        """
        try:
            # git stash list --format='%H %gd' emits lines like:
            # abc1234... stash@{0}
            result = self._run(["git", "stash", "list", "--format=%H %gd"])
            for line in result.stdout.splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2 and parts[0].startswith(sha[:12]):
                    return parts[1]  # e.g. "stash@{2}"
        except RuntimeError:
            pass
        return None

    def reset_hard(self, ref: str) -> None:
        """
        Hard-reset the working tree to ``ref``.

        This is the emergency rollback used by the HITL escalation node.
        All uncommitted changes are discarded.
        """
        self._run(["git", "reset", "--hard", ref])
        self._run(["git", "clean", "-fd"])   # remove untracked files
        log.warning("git.reset_hard", ref=ref)

    def get_last_green_commit(self) -> str:
        """
        Returns the last commit SHA recorded by ``record_green_commit``.
        Falls back to HEAD if no record exists.
        """
        green_sha_file = self._root / ".workflow" / "green-sha"
        if green_sha_file.exists():
            sha = green_sha_file.read_text().strip()
            if sha:
                return sha
        # Fallback: use HEAD
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()

    def record_green_commit(self, sha: str) -> None:
        green_sha_file = self._root / ".workflow" / "green-sha"
        green_sha_file.parent.mkdir(parents=True, exist_ok=True)
        green_sha_file.write_text(sha.strip())
        log.info("git.green_commit_recorded", sha=sha[:12])

    def current_branch(self) -> str:
        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def uncommitted_files(self) -> list[str]:
        result = self._run(["git", "status", "--porcelain"])
        lines  = [l for l in result.stdout.splitlines() if l.strip()]
        # Porcelain format: XX FILENAME or XX FILENAME <tab> RENAMED_FILE
        # First 2 chars are status, char 3 is space or tab
        return [l[3:].split("\t")[0] for l in lines if len(l) >= 3]

    # ── Speculative branch isolation ──────────────────────────────────────

    def create_worktree(self, branch_name: str, worktree_path: Path) -> Path:
        """Create an isolated worktree for speculative evaluation.

        If the branch already exists (e.g., from a prior crashed run),
        reuse it without creating a new branch.
        """
        worktree_path.mkdir(parents=True, exist_ok=True)
        # Check if branch already exists
        check = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=str(self._root), capture_output=True, text=True, timeout=10,
        )
        branch_exists = bool(check.stdout.strip())

        if branch_exists:
            # Reuse existing branch — worktree add without -b
            self._run(["git", "worktree", "add", str(worktree_path), branch_name])
        else:
            # Create new branch
            self._run(["git", "worktree", "add", str(worktree_path), "-b", branch_name])

        log.info("git.worktree_created", branch=branch_name, path=str(worktree_path))
        return worktree_path

    def remove_worktree(self, worktree_path: Path) -> None:
        """Remove an isolated worktree after evaluation."""
        self._run(["git", "worktree", "remove", "--force", str(worktree_path)])
        log.info("git.worktree_removed", path=str(worktree_path))

    def prune_worktrees(self) -> None:
        """Prune stale worktree registry entries left by removed worktrees."""
        self._run(["git", "worktree", "prune"])
        log.debug("git.worktrees_pruned")

    def stage_file(self, path: str) -> None:
        """Stage a single file for commit (partial staging)."""
        self._run(["git", "add", path])
        log.debug("git.stage_file", path=path)

    def head_sha(self) -> str:
        """Return the current HEAD commit SHA."""
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()

    # ── Additional utility methods ────────────────────────────────────────

    def list_branches(self, pattern: str = "agent-*") -> list[str]:
        result = self._run(["git", "branch", "--list", pattern])
        return [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]

    def delete_branch(self, name: str, force: bool = False) -> None:
        flag = "-D" if force else "-d"
        self._run(["git", "branch", flag, name])
        log.info("git.branch_deleted", name=name, force=force)

    def commit(self, message: str, add_all: bool = True) -> str:
        if add_all:
            self._run(["git", "add", "-A"])
            # Un-stage workflow internals — these contain internal state,
            # not application code. They should never enter the repo.
            self._run(["git", "reset", "--", ".workflow/"])
        result = self._run(["git", "commit", "-m", message])
        sha = self._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        log.info("git.commit", sha=sha[:12], message=message[:60])
        return sha

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.error(
                "git.command_failed",
                cmd=" ".join(cmd),
                stderr=result.stderr[:300],
                rc=result.returncode,
            )
            raise RuntimeError(
                f"git command failed (rc={result.returncode}): "
                f"{' '.join(cmd)}\n{result.stderr[:300]}"
            )
        return result
