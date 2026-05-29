"""
tests/unit/test_diff_validation.py
====================================
Unit tests for the full-overwrite guard in DiffEngine.
"""

import pytest
from sacv.git.diff_engine import DiffEngine, MAX_OVERWRITE_RATIO
from sacv.interfaces.diff_provider import UnifiedDiff
import tempfile
from pathlib import Path


@pytest.mark.asyncio
class TestDiffValidation:

    async def test_targeted_diff_passes_validation(self, tmp_path):
        target = tmp_path / "Service.java"
        # 50 lines of content
        target.write_text("\n".join([f"line {i}" for i in range(50)]))

        engine = DiffEngine(repo_root=tmp_path)
        diff   = UnifiedDiff(
            file_path="Service.java",
            # Only removes 2 lines out of 50
            diff_content="@@ -10,2 +10,2 @@\n-line 10\n-line 11\n+updated 10\n+updated 11",
            operation="modify",
            language="java",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_full_overwrite_diff_fails_validation(self, tmp_path):
        target = tmp_path / "Service.java"
        target.write_text("\n".join([f"line {i}" for i in range(50)]))

        engine = DiffEngine(repo_root=tmp_path)
        # Remove 48 of 50 lines = 96% — above MAX_OVERWRITE_RATIO
        removed_lines = "\n".join([f"-line {i}" for i in range(48)])
        diff = UnifiedDiff(
            file_path="Service.java",
            diff_content=f"@@ -1,48 +1,0 @@\n{removed_lines}",
            operation="modify",
            language="java",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert len(errors) == 1
        assert "Service.java" in errors[0].file_path
        assert "threshold" in errors[0].reason

    async def test_create_operation_always_passes(self, tmp_path):
        engine = DiffEngine(repo_root=tmp_path)
        diff   = UnifiedDiff(
            file_path="NewService.java",
            diff_content="\n".join([f"+line {i}" for i in range(200)]),
            operation="create",
            language="java",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_delete_operation_always_passes(self, tmp_path):
        target = tmp_path / "OldService.java"
        target.write_text("content")
        engine = DiffEngine(repo_root=tmp_path)
        diff   = UnifiedDiff(
            file_path="OldService.java",
            diff_content="-content",
            operation="delete",
            language="java",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []

    async def test_nonexistent_file_skip_validation(self, tmp_path):
        engine = DiffEngine(repo_root=tmp_path)
        diff   = UnifiedDiff(
            file_path="does_not_exist.java",
            diff_content="-line\n+other",
            operation="modify",
            language="java",
        )
        errors = await engine.validate_no_full_overwrite([diff])
        assert errors == []
