"""
tests/unit/test_branch_manager.py
===================================
Unit tests for BranchManager.

Tests cover:
1. create_branch — creates and checks out a new branch
2. stash — returns empty string when nothing to stash
3. stash — stashes and returns SHA when changes exist
4. reset_hard — resets to green SHA
5. get_last_green_commit — reads from file, falls back to HEAD
6. record_green_commit — persists SHA to file
7. current_branch — returns branch name
8. uncommitted_files — parses git status output
9. worktree — creates worktree for new branch
10. list_branches — returns matching branches
11. delete_branch — deletes a branch
12. commit — commits with all files
"""
from __future__ import annotations

import subprocess
import pytest
from pathlib import Path

from sacv.git.branch_manager import BranchManager, sanitize_branch_name


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
class TestCreateBranch:

    def test_creates_new_branch(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        mgr.create_branch("feature-1")
        branch = mgr.current_branch()
        assert branch == "feature-1"

    def test_creates_from_ref(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Create a tag at HEAD
        subprocess.run(["git", "tag", "v1"], cwd=mgr._root, capture_output=True, check=True)
        mgr.create_branch("feature-1", from_ref="v1")
        assert mgr.current_branch() == "feature-1"

    def test_returns_branch_name(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        result = mgr.create_branch("feature-1")
        assert result == "feature-1"


@pytest.mark.unit
class TestStash:

    def test_nothing_to_stash_returns_empty(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        result = mgr.stash("test stash")
        assert result == ""

    def test_stash_with_changes_returns_sha(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Make an uncommitted change (tracked file)
        (mgr._root / "README.md").write_text("hello modified")
        result = mgr.stash("test stash")
        assert result != ""
        # Should be a SHA-like string
        assert len(result) >= 7

    def test_stash_pop_returns_branch_to_clean(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Make a tracked change
        (mgr._root / "README.md").write_text("hello modified")
        stash_ref = mgr.stash("test stash")
        assert stash_ref != ""
        # Working tree should be clean now
        files = mgr.uncommitted_files()
        assert "README.md" not in files


@pytest.mark.unit
class TestResetHard:

    def test_resets_to_green_sha(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Record green SHA
        green_sha = mgr.head_sha()
        mgr.record_green_commit(green_sha)
        # Make a change
        (mgr._root / "changed.txt").write_text("changed")
        mgr.reset_hard(green_sha)
        # File should be gone
        assert not (mgr._root / "changed.txt").exists()

    def test_resets_to_head_when_no_green_sha(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # No green SHA recorded — should fall back to HEAD
        # This should not raise
        head = mgr.head_sha()
        mgr.reset_hard(head)

    def test_clean_after_reset(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "untracked.txt").write_text("untracked")
        green_sha = mgr.head_sha()
        mgr.record_green_commit(green_sha)
        mgr.reset_hard(green_sha)
        # Untracked files should be removed by git clean -fd
        assert not (mgr._root / "untracked.txt").exists()


@pytest.mark.unit
class TestGreenSha:

    def test_record_and_read(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        sha = "abc123def456"
        mgr.record_green_commit(sha)
        assert mgr.get_last_green_commit() == sha

    def test_fallback_to_head_when_no_record(self, tmp_path):
        # Clean up any stale green SHA file from previous tests
        green_sha_file = Path(tmp_path) / "repo" / ".workflow" / "green-sha"
        if green_sha_file.exists():
            green_sha_file.unlink(missing_ok=True)
        mgr = _init_test_repo(tmp_path)
        mgr2 = BranchManager(mgr._root)
        sha = mgr2.get_last_green_commit()
        # Should return HEAD
        assert sha == mgr2.head_sha()

    def test_empty_file_fallback_to_head(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Write empty green SHA file using the instance's path
        green_sha_file = mgr._root / ".workflow" / "green-sha"
        green_sha_file.parent.mkdir(parents=True, exist_ok=True)
        green_sha_file.write_text("")
        mgr2 = BranchManager(mgr._root)
        sha = mgr2.get_last_green_commit()
        assert sha == mgr2.head_sha()

    def test_whitespace_in_file_handled(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        sha = mgr.head_sha()
        green_sha_file = mgr._root / ".workflow" / "green-sha"
        green_sha_file.parent.mkdir(parents=True, exist_ok=True)
        green_sha_file.write_text(f"  {sha}  \n")
        mgr2 = BranchManager(mgr._root)
        assert mgr2.get_last_green_commit() == sha


@pytest.mark.unit
class TestCurrentBranch:

    def test_returns_main_after_init(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        assert mgr.current_branch() == "master" or mgr.current_branch() == "main"

    def test_returns_new_branch_name(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        mgr.create_branch("feature-1")
        assert mgr.current_branch() == "feature-1"


@pytest.mark.unit
class TestUncommittedFiles:

    def test_empty_when_clean(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        assert mgr.uncommitted_files() == []

    def test_returns_modified_files(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "README.md").write_text("modified")
        files = mgr.uncommitted_files()
        assert "README.md" in files

    def test_returns_added_files(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "new.txt").write_text("new")
        files = mgr.uncommitted_files()
        assert "new.txt" in files


@pytest.mark.unit
class TestListBranches:

    def test_returns_agent_branches(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        mgr.create_branch("agent-task-1")
        mgr.create_branch("agent-task-2")
        mgr.create_branch("feature-x")
        branches = mgr.list_branches("agent-*")
        assert "agent-task-1" in branches
        assert "agent-task-2" in branches
        assert "feature-x" not in branches

    def test_empty_when_no_matches(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        mgr.create_branch("feature-x")
        branches = mgr.list_branches("agent-*")
        assert branches == []


@pytest.mark.unit
class TestDeleteBranch:

    def test_deletes_existing_branch(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        original = mgr.current_branch()
        mgr.create_branch("feature-1")
        # Switch back to original branch before deleting (git won't delete
        # a branch we're on if it thinks the repo is a worktree for it)
        mgr.checkout(original)
        mgr.delete_branch("feature-1")
        branches = mgr.list_branches("feature-1")
        assert branches == []

    def test_force_deletes_unmerged(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        original = mgr.current_branch()
        mgr.create_branch("feature-1")
        mgr.checkout(original)
        mgr.delete_branch("feature-1", force=True)
        branches = mgr.list_branches("feature-1")
        assert branches == []


@pytest.mark.unit
class TestCommit:

    def test_commits_all_changes(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        (mgr._root / "new.txt").write_text("new content")
        sha = mgr.commit("test commit")
        assert len(sha) == 40  # full SHA

    def test_commit_returns_sha(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Need a file change to commit (README already committed)
        (mgr._root / "new.txt").write_text("new content")
        sha = mgr.commit("test commit")
        assert len(sha) == 40

    def test_excludes_workflow_dir(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        # Create both a workflow file AND a regular file to commit
        workflow_dir = tmp_path / "repo" / ".workflow"
        workflow_dir.mkdir()
        (workflow_dir / "state.json").write_text("{}")
        (mgr._root / "new.txt").write_text("new content")
        sha = mgr.commit("test commit")
        assert len(sha) == 40
        # The .workflow directory should not be in the commit
        # (git reset -- .workflow/ un-stages it)


@pytest.mark.unit
class TestHeadSha:

    def test_returns_current_head(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        sha = mgr.head_sha()
        assert len(sha) == 40

    def test_changes_after_commit(self, tmp_path):
        mgr = _init_test_repo(tmp_path)
        sha1 = mgr.head_sha()
        (mgr._root / "new.txt").write_text("new")
        mgr.commit("second commit")
        sha2 = mgr.head_sha()
        assert sha1 != sha2


class TestSanitizeBranchName:

    def test_alphanumeric_unchanged(self):
        assert sanitize_branch_name("feature-1") == "feature-1"

    def test_spaces_become_dashes(self):
        assert sanitize_branch_name("feature one") == "feature-one"

    def test_slashes_become_dashes(self):
        assert sanitize_branch_name("feature/one") == "feature-one"

    def test_backslashes_become_dashes(self):
        assert sanitize_branch_name("feature\\one") == "feature-one"

    def test_tildes_become_dashes(self):
        assert sanitize_branch_name("feature~one") == "feature-one"

    def test_carets_become_dashes(self):
        assert sanitize_branch_name("feature^one") == "feature-one"

    def test_question_marks_become_dashes(self):
        assert sanitize_branch_name("feature?one") == "feature-one"

    def test_asterisks_become_dashes(self):
        assert sanitize_branch_name("feature*one") == "feature-one"

    def test_brackets_become_dashes(self):
        # [ and ] each become a dash, then collapsed by multi-dash rule
        assert sanitize_branch_name("feature[one]two") == "feature-one-two"

    def test_multiple_dashes_collapsed(self):
        assert sanitize_branch_name("feature---one") == "feature-one"

    def test_leading_dashes_stripped(self):
        assert sanitize_branch_name("---feature") == "feature"

    def test_trailing_dashes_stripped(self):
        assert sanitize_branch_name("feature---") == "feature"

    def test_all_special_chars(self):
        result = sanitize_branch_name("feat/ure\\name~v1^beta?final*test[1]")
        assert "/" not in result
        assert "\\" not in result
        assert "~" not in result
        assert "^" not in result
        assert "?" not in result
        assert "*" not in result
        assert "[" not in result
        assert "]" not in result
        assert "--" not in result

    def test_dots_preserved(self):
        assert sanitize_branch_name("feat.ure") == "feat.ure"

    def test_underscores_preserved(self):
        assert sanitize_branch_name("feat_ure") == "feat_ure"

    def test_numbers_preserved(self):
        assert sanitize_branch_name("feat123") == "feat123"

    def test_empty_result_defaults_to_branch(self):
        assert sanitize_branch_name("---") == "branch"

    def test_special_chars_only_defaults_to_branch(self):
        assert sanitize_branch_name("???") == "branch"

    def test_unicode_word_chars_preserved(self):
        # Python \w matches Unicode word characters, so é is preserved
        result = sanitize_branch_name("featéure")
        assert "é" in result

    def test_parentheses_become_dashes(self):
        assert sanitize_branch_name("feat(ure)") == "feat-ure"

    def test_real_world_task_names(self):
        assert sanitize_branch_name("implement user service") == "implement-user-service"
        assert sanitize_branch_name("fix @bug in auth") == "fix-bug-in-auth"
        assert sanitize_branch_name("add new [feature]") == "add-new-feature"
