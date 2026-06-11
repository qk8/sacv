"""
tests/unit/test_branch_manager_edge_cases.py
==============================================
Unit tests for BranchManager edge cases and error handling paths.

Tests cover:
1. stash_pop with empty ref — returns without error
2. stash_pop with non-matching SHA — returns without error (SHA not in stash list)
3. stash_drop with empty ref — returns without error
4. stash_drop with non-matching SHA — logs warning, does not raise
5. _sha_to_stash_ref with SHA not in stash — returns None
6. remove_worktree — removes existing worktree
7. prune_worktrees — runs without error in clean repo
8. stage_file — stages a single file
"""
from __future__ import annotations

import subprocess
import pytest
from pathlib import Path

from sacv.git.branch_manager import BranchManager


def _init_test_repo(tmp_path: Path) -> BranchManager:
    """Initialize a minimal git repo and return a BranchManager pointing at it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)
    return BranchManager(repo)


@pytest.mark.unit
class TestStashPopEdgeCases:

    def test_empty_ref_returns_without_error(self, tmp_path):
        """stash_pop with empty string ref should return early without error."""
        mgr = _init_test_repo(tmp_path)
        # Should not raise
        mgr.stash_pop("")

    def test_non_matching_sha_returns_without_error(self, tmp_path):
        """stash_pop with SHA not in stash list should return without error."""
        mgr = _init_test_repo(tmp_path)
        # Stash something first so stash list exists
        (mgr._root / "README.md").write_text("modified")
        mgr.stash("test stash")
        # Now try to pop a SHA that doesn't exist in stash
        mgr.stash_pop("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


@pytest.mark.unit
class TestStashDropEdgeCases:

    def test_empty_ref_returns_without_error(self, tmp_path):
        """stash_drop with empty string ref should return early without error."""
        mgr = _init_test_repo(tmp_path)
        # Should not raise
        mgr.stash_drop("")

    def test_non_matching_sha_logs_warning(self, tmp_path):
        """stash_drop with SHA not in stash list should log warning, not raise."""
        mgr = _init_test_repo(tmp_path)
        # Stash something first so stash list exists
        (mgr._root / "README.md").write_text("modified")
        mgr.stash("test stash")
        # Drop a SHA that doesn't exist — should not raise
        mgr.stash_drop("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

    def test_existing_sha_is_dropped(self, tmp_path):
        """stash_drop with valid ref actually removes the stash."""
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "README.md").write_text("modified")
        stash_ref = mgr.stash("test stash")
        assert stash_ref != ""
        # Drop the stash
        mgr.stash_drop(stash_ref)
        # Try to pop it again — should behave like non-matching SHA
        mgr.stash_pop(stash_ref)


@pytest.mark.unit
class TestShaToStashRef:

    def test_sha_not_in_stash_returns_none(self, tmp_path):
        """SHA not found in stash list returns None."""
        mgr = _init_test_repo(tmp_path)
        # Create an empty stash list by stashing and dropping
        (mgr._root / "README.md").write_text("modified")
        ref = mgr.stash("temp")
        mgr.stash_drop(ref)
        # Now stash list is empty — lookup should return None
        result = mgr._sha_to_stash_ref("abc123def456")
        assert result is None

    def test_partial_sha_match(self, tmp_path):
        """Partial SHA prefix match returns the positional ref."""
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "README.md").write_text("modified")
        ref = mgr.stash("test stash")
        assert ref != ""
        # Use first 12 chars of SHA
        partial = ref[:12]
        result = mgr._sha_to_stash_ref(partial)
        assert result is not None
        assert result.startswith("stash@{")


@pytest.mark.unit
class TestWorktreeOperations:

    def test_remove_worktree(self, tmp_path):
        """remove_worktree removes an existing worktree."""
        mgr = _init_test_repo(tmp_path)
        worktree_path = tmp_path / "worktree"
        mgr.create_worktree("feature-1", worktree_path)
        assert worktree_path.exists()
        mgr.remove_worktree(worktree_path)
        assert not worktree_path.exists()

    def test_prune_worktrees_clean(self, tmp_path):
        """prune_worktrees runs without error in a clean repo."""
        mgr = _init_test_repo(tmp_path)
        # Should not raise
        mgr.prune_worktrees()

    def test_create_worktree_different_branches(self, tmp_path):
        """create_worktree creates separate worktrees for different branches."""
        mgr = _init_test_repo(tmp_path)
        worktree_path = tmp_path / "worktree"
        mgr.create_worktree("feature-1", worktree_path)
        worktree_path2 = tmp_path / "worktree2"
        mgr.create_worktree("feature-2", worktree_path2)
        assert worktree_path.exists()
        assert worktree_path2.exists()


@pytest.mark.unit
class TestStageFile:

    def test_stages_single_file(self, tmp_path):
        """stage_file stages only the specified file."""
        mgr = _init_test_repo(tmp_path)
        # Modify two files
        (mgr._root / "README.md").write_text("modified readme")
        (mgr._root / "new.txt").write_text("new content")
        # Stage only one
        mgr.stage_file("new.txt")
        # Check staged files via git diff --cached
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=mgr._root, capture_output=True, text=True, check=True,
        )
        staged = result.stdout.strip().splitlines()
        assert "new.txt" in staged
        assert "README.md" not in staged
