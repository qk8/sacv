from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

@dataclass
class UnifiedDiff:
    file_path: str
    diff_content: str
    operation: Literal["modify", "create", "delete"]
    language: str

@dataclass
class DiffValidationError:
    file_path: str
    reason: str

@dataclass
class DiffResult:
    success: bool
    applied_files: list[str]
    conflicts: list[dict]
    validation_errors: list[DiffValidationError]

class DiffProvider(ABC):
    @abstractmethod
    async def apply_diffs(self, diffs: list[UnifiedDiff]) -> DiffResult: ...
    @abstractmethod
    async def validate_no_full_overwrite(self, diffs: list[UnifiedDiff]) -> list[DiffValidationError]: ...
    @abstractmethod
    async def generate_ast_diff(self, original: str, modified: str, language: str) -> UnifiedDiff: ...
