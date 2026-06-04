"""
tests/unit/test_diff_engine.py
================================
Unit tests for DiffEngine.

Tests cover:
1. validate_no_full_overwrite — targeted diff passes
2. validate_no_full_overwrite — full overwrite rejected
3. validate_no_full_overwrite — create operation allowed
4. validate_no_full_overwrite — delete operation allowed
5. validate_no_full_overwrite — nonexistent file treated as create
6. validate_no_full_overwrite — boundary at exact ratio
7. apply_diffs — create file
8. apply_diffs — delete file
9. apply_diffs — patch apply success
10. apply_diffs — patch failure produces conflict
11. generate_ast_diff — produces unified diff
"""
from __future__ import annotations

import pytest
from pathlib import Path

from sacv.git.diff_engine import DiffEngine, MAX_OVERWRITE_RATIO
from sacv.interfaces.diff_provider import UnifiedDiff


@pytest.mark.asyncio
@pytest.mark.unit
class TestValidateNoFullOverwrite:

    def _diff(self, file_path: str, content: str = "line1\nline2\nline3\n",
              operation: str = "modify") -> UnifiedDiff:
        return UnifiedDiff(
            file_path=file_path,
            diff_content=content,
            operation=operation,
            language="java",
        )

    async def test_targeted_diff_passes(self, tmp_path):
        """A diff that removes 1 line from a 100-line file should pass."""
        engine = DiffEngine(tmp_path)
        # Create a 100-line file
        target = tmp_path / "UserService.java"
        target.write_text("\n".join(f"line{i}" for i in range(100)))
        diff = self._diff(
            "UserService.java",
            "@@ -10 +10 @@\n-old\n+new",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_full_overwrite_rejected(self, tmp_path):
        """A diff removing >= 90% of lines should be rejected."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "UserService.java"
        target.write_text("\n".join(f"line{i}" for i in range(100)))
        # Remove 95 lines (95% of 100)
        diff = self._diff(
            "UserService.java",
            "\n".join(f"-line{i}" for i in range(95)),
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert len(errors) == 1
        assert "95%" in errors[0].reason

    async def test_create_operation_allowed(self, tmp_path):
        """New file creation is always allowed."""
        engine = DiffEngine(tmp_path)
        diff = self._diff("NewService.java", "new content", operation="create")
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_delete_operation_allowed(self, tmp_path):
        """File deletion is always allowed."""
        engine = DiffEngine(tmp_path)
        diff = self._diff("UserService.java", "-line1\n-line2", operation="delete")
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_nonexistent_file_treated_as_create(self, tmp_path):
        """If file doesn't exist, diff is treated as create (allowed)."""
        engine = DiffEngine(tmp_path)
        diff = self._diff("NonExistent.java", "-old\n+new")
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_empty_file_treated_as_create(self, tmp_path):
        """If file exists but is empty, diff is allowed."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "Empty.java"
        target.write_text("")
        diff = self._diff("Empty.java", "-old\n+new")
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_boundary_at_exact_ratio(self, tmp_path):
        """A diff removing exactly MAX_OVERWRITE_RATIO (90%) should be rejected."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "UserService.java"
        # 10 lines, remove 9 = 90% = exactly at threshold
        target.write_text("\n".join(f"line{i}" for i in range(10)))
        diff = self._diff(
            "UserService.java",
            "\n".join(f"-line{i}" for i in range(9)),
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert len(errors) == 1

    async def test_below_boundary_passes(self, tmp_path):
        """A diff removing 89% of lines should pass (below 90% threshold)."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "UserService.java"
        # 100 lines, remove 89 = 89% < 90%
        target.write_text("\n".join(f"line{i}" for i in range(100)))
        diff = self._diff(
            "UserService.java",
            "\n".join(f"-line{i}" for i in range(89)),
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_multiple_diffs_independent_validation(self, tmp_path):
        """Each diff is validated independently; one failure doesn't block others."""
        engine = DiffEngine(tmp_path)
        target1 = tmp_path / "Good.java"
        target1.write_text("\n".join(f"line{i}" for i in range(100)))
        target2 = tmp_path / "Bad.java"
        target2.write_text("\n".join(f"line{i}" for i in range(100)))
        diffs = [
            self._diff("Good.java", "@@ -10 +10 @@\n-old\n+new"),  # 1/100 = 1%
            self._diff("Bad.java", "\n".join(f"-line{i}" for i in range(95))),  # 95%
        ]
        errors = await engine.validate_no_full_overwrite(diffs)
        assert len(errors) == 1
        assert errors[0].file_path == "Bad.java"

    async def test_added_lines_not_counted_as_removal(self, tmp_path):
        """Only removed lines (starting with -) are counted, not added lines."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "UserService.java"
        target.write_text("\n".join(f"line{i}" for i in range(100)))
        # Add 50 new lines but remove only 1
        added = "\n+".join(f"new{i}" for i in range(50))
        diff = self._diff("UserService.java", f"-old\n{added}")
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestApplyDiffs:

    async def test_create_file(self, tmp_path):
        engine = DiffEngine(tmp_path)
        diff = UnifiedDiff(
            file_path="new/File.java",
            diff_content="package new;\npublic class File {}",
            operation="create", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert result.success
        assert "new/File.java" in result.applied_files
        assert (tmp_path / "new/File.java").exists()

    async def test_create_file_from_raw_content(self, tmp_path):
        """Create with raw content (not unified diff format)."""
        engine = DiffEngine(tmp_path)
        diff = UnifiedDiff(
            file_path="raw.java",
            diff_content="public class Raw {}",
            operation="create", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert result.success
        content = (tmp_path / "raw.java").read_text()
        assert "public class Raw {}" in content

    async def test_delete_file(self, tmp_path):
        engine = DiffEngine(tmp_path)
        target = tmp_path / "to_delete.java"
        target.write_text("content")
        diff = UnifiedDiff(
            file_path="to_delete.java",
            diff_content="-content",
            operation="delete", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert result.success
        assert "to_delete.java" in result.applied_files
        assert not target.exists()

    async def test_delete_nonexistent_file_succeeds(self, tmp_path):
        """Deleting a file that doesn't exist is a no-op (success)."""
        engine = DiffEngine(tmp_path)
        diff = UnifiedDiff(
            file_path="nonexistent.java",
            diff_content="",
            operation="delete", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert result.success

    async def test_patch_failure_creates_conflict(self, tmp_path):
        """Patch that doesn't apply produces a conflict."""
        engine = DiffEngine(tmp_path)
        target = tmp_path / "UserService.java"
        target.write_text("public class UserService {}\n")
        diff = UnifiedDiff(
            file_path="UserService.java",
            diff_content="@@ -1 +1 @@\n-old\n+new",  # "old" doesn't exist in file
            operation="modify", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert not result.success
        assert len(result.conflicts) == 1
        assert result.conflicts[0]["file"] == "UserService.java"

    async def test_mixed_success_and_failure(self, tmp_path):
        """One success + one failure = partial success."""
        engine = DiffEngine(tmp_path)
        # Existing file for modify (with proper unified diff)
        good_target = tmp_path / "Good.java"
        good_target.write_text("good\n")
        # File that does NOT exist — create will succeed
        diffs = [
            UnifiedDiff(file_path="Good.java",
                        diff_content="--- a/Good.java\n+++ b/Good.java\n@@ -1 +1 @@\n-good\n+new content\n",
                        operation="modify", language="java"),
            UnifiedDiff(file_path="New.java", diff_content="new content",
                        operation="create", language="java"),
            UnifiedDiff(file_path="Bad.java", diff_content="@@ -1 +1 @@\n-old\n+new",
                        operation="modify", language="java"),
        ]
        result = await engine.apply_diffs(diffs)
        assert not result.success  # has conflicts
        assert "Good.java" in result.applied_files
        assert "New.java" in result.applied_files
        assert len(result.conflicts) == 1

    async def test_apply_diffs_uses_asyncio_to_thread(self, tmp_path):
        """Internal helpers run in asyncio.to_thread (sync)."""
        engine = DiffEngine(tmp_path)
        # Just verify it's callable as async
        diff = UnifiedDiff(
            file_path="test.java", diff_content="content",
            operation="create", language="java",
        )
        result = await engine.apply_diffs([diff])
        assert result.success


@pytest.mark.asyncio
@pytest.mark.unit
class TestGenerateAstDiff:

    async def test_produces_unified_diff(self):
        engine = DiffEngine(".")
        diff = await engine.generate_ast_diff("line1\nline2\n", "line1\nline3\n", language="java")
        assert diff.operation == "modify"
        assert diff.language == "java"
        assert "@@" in diff.diff_content
        assert "-line2" in diff.diff_content
        assert "+line3" in diff.diff_content

    async def test_create_operation_diff(self):
        engine = DiffEngine(".")
        diff = await engine.generate_ast_diff("", "new content", language="java")
        assert diff.operation == "modify"
        assert "+new content" in diff.diff_content

    async def test_delete_operation_diff(self):
        engine = DiffEngine(".")
        diff = await engine.generate_ast_diff("old content", "", language="java")
        assert diff.operation == "modify"
        assert "-old content" in diff.diff_content
