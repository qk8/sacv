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

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.interfaces.git_provider import GitProvider

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_GREEN_SHA_FILE = Path(".workflow/green-sha")


class BranchManager(GitProvider):
    """
    Thin, type-safe wrapper around git CLI subprocess calls.

    Uses ``subprocess.run`` rather than GitPython to avoid the library's
    mutable state model and keep the interface deterministic.
    """

    def __init__(self, repo_root: str | Path = ".") -> None:
        self._root = Path(repo_root).resolve()

    # ── GitProvider interface ─────────────────────────────────────────────

    def create_branch(self, name: str, from_ref: str = "HEAD") -> str:
        self._run(["git", "checkout", "-b", name, from_ref])
        log.info("git.branch_created", name=name, from_ref=from_ref)
        return name

    def checkout(self, branch_name: str) -> None:
        self._run(["git", "checkout", branch_name])
        log.debug("git.checkout", branch=branch_name)

    def stash(self, message: str) -> str:
        result = self._run(["git", "stash", "push", "-m", message])
        ref = "stash@{0}"   # git always pushes to the top of the stash
        log.info("git.stash", message=message, ref=ref)
        return ref

    def stash_pop(self, ref: str) -> None:
        self._run(["git", "stash", "pop", ref])
        log.info("git.stash_pop", ref=ref)

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
        if _GREEN_SHA_FILE.exists():
            sha = _GREEN_SHA_FILE.read_text().strip()
            if sha:
                return sha
        # Fallback: use HEAD
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()

    def record_green_commit(self, sha: str) -> None:
        _GREEN_SHA_FILE.parent.mkdir(parents=True, exist_ok=True)
        _GREEN_SHA_FILE.write_text(sha.strip())
        log.info("git.green_commit_recorded", sha=sha[:12])

    def current_branch(self) -> str:
        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def uncommitted_files(self) -> list[str]:
        result = self._run(["git", "status", "--porcelain"])
        lines  = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return [l[3:] for l in lines]   # strip status prefix (e.g. "M  ")

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
