"""
git/diff_engine.py
==================
Concrete implementation of DiffProvider.

Enforces the "diff-only, no full-file overwrites" constraint by analysing
the unified diff hunk statistics before applying.  A diff that replaces
more than ``MAX_OVERWRITE_RATIO`` of a file's lines is rejected.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

import structlog

from sacv.interfaces.diff_provider import (
    DiffProvider,
    UnifiedDiff,
    DiffResult,
    DiffValidationError,
)

log = structlog.get_logger(__name__)

# A diff that modifies ≥90% of a file's lines is treated as a full overwrite.
MAX_OVERWRITE_RATIO = 0.90

# Regex to count hunk lines
_ADDED_RE   = re.compile(r"^\+(?!\+\+)", re.MULTILINE)
_REMOVED_RE = re.compile(r"^-(?!--)", re.MULTILINE)


class DiffEngine(DiffProvider):

    def __init__(self, repo_root: str | Path = ".") -> None:
        self._root = Path(repo_root).resolve()

    async def validate_no_full_overwrite(
        self, diffs: list[UnifiedDiff]
    ) -> list[DiffValidationError]:
        errors: list[DiffValidationError] = []
        for diff in diffs:
            if diff.operation == "create":
                # Validate that the target does NOT already exist
                if (self._root / diff.file_path).exists():
                    errors.append(DiffValidationError(
                        file_path=diff.file_path,
                        reason="'create' operation targets an existing file. Use 'modify'.",
                    ))
                continue
            if diff.operation == "delete":
                continue   # deletion is allowed (no overwrite concern)

            file_path = self._root / diff.file_path
            if not file_path.exists():
                continue   # file doesn't exist yet — treat as create

            existing_lines = len(file_path.read_text().splitlines())
            if existing_lines == 0:
                continue

            removed = len(_REMOVED_RE.findall(diff.diff_content))
            ratio   = removed / existing_lines

            if ratio >= MAX_OVERWRITE_RATIO:
                errors.append(DiffValidationError(
                    file_path=diff.file_path,
                    reason=(
                        f"Diff removes {removed}/{existing_lines} lines "
                        f"({ratio:.0%} ≥ {MAX_OVERWRITE_RATIO:.0%} threshold). "
                        "Produce a targeted patch, not a full-file rewrite."
                    ),
                ))
        return errors

    async def apply_diffs(self, diffs: list[UnifiedDiff]) -> DiffResult:
        applied:   list[str] = []
        conflicts: list[dict] = []

        for diff in diffs:
            try:
                if diff.operation == "create":
                    await asyncio.to_thread(self._create_file, diff)
                elif diff.operation == "delete":
                    await asyncio.to_thread(self._delete_file, diff)
                else:
                    await asyncio.to_thread(self._apply_patch, diff)
                applied.append(diff.file_path)
            except (RuntimeError, FileNotFoundError, OSError) as exc:
                conflicts.append({"file": diff.file_path, "error": str(exc)})

        success = len(conflicts) == 0
        log.info(
            "diff_engine.apply",
            applied=len(applied),
            conflicts=len(conflicts),
        )
        return DiffResult(
            success=success,
            applied_files=applied,
            conflicts=conflicts,
            validation_errors=[],
        )

    async def generate_ast_diff(
        self, original: str, modified: str, language: str
    ) -> UnifiedDiff:
        """
        Produces a unified diff between ``original`` and ``modified`` strings.
        Uses Python's difflib — no subprocess needed.
        """
        import difflib
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            original_lines, modified_lines, lineterm=""
        ))
        return UnifiedDiff(
            file_path="<generated>",
            diff_content="\n".join(diff_lines),
            operation="modify",
            language=language,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _apply_patch(self, diff: UnifiedDiff) -> None:
        proc = subprocess.run(
            ["patch", "-p1", "--no-backup-if-mismatch"],
            input=diff.diff_content,
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"patch failed for {diff.file_path}: {proc.stderr[:300]}"
            )
        # Verify the patch actually took effect. The `patch` command can
        # silently apply fuzzy matches when multi-line context doesn't match
        # exactly, corrupting the file. We verify by checking that at least
        # one added line is present in the resulting file.
        target = self._root / diff.file_path
        if not target.exists():
            raise RuntimeError(f"patch target missing after apply: {diff.file_path}")
        patched = target.read_text()
        added_lines = [
            line[1:]
            for line in diff.diff_content.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        for line in added_lines:
            if line and line in patched:
                break
        else:
            raise RuntimeError(
                f"patch applied but verification failed for {diff.file_path}: "
                "added lines not found in patched file — possible fuzzy match"
            )

    def _create_file(self, diff: UnifiedDiff) -> None:
        target = self._root / diff.file_path
        if target.exists():
            raise RuntimeError(
                f"Cannot create '{diff.file_path}': file already exists. "
                "Use operation='modify' with a diff patch instead."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # For create operations, diff_content may be raw file content
        # or a unified diff starting with "+++" — handle both.
        if diff.diff_content.startswith("+++"):
            added = [
                line[1:]
                for line in diff.diff_content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            target.write_text("\n".join(added))
        else:
            target.write_text(diff.diff_content)

    def _delete_file(self, diff: UnifiedDiff) -> None:
        target = self._root / diff.file_path
        if target.exists():
            target.unlink()
