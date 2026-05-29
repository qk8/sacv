from __future__ import annotations
from abc import ABC, abstractmethod

class CrossDomainProvider(ABC):
    @abstractmethod
    async def map_code_to_schema(self, entity_names: list[str]) -> dict: ...
    @abstractmethod
    async def get_arch_alignment(self, module_paths: list[str]) -> dict: ...
    @abstractmethod
    async def get_sql_impact(self, changed_files: list[str]) -> dict: ...
