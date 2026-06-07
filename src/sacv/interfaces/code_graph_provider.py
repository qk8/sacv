from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BlastRadiusMap:
    entry_files: list[str]
    affected_files: list[str]
    dependency_depth: int
    cross_service_impact: list[str]
    schema_impact: list[str]
    risk_score: float

@dataclass
class CallGraph:
    entry_point: str
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]

class CodeGraphProvider(ABC):
    @abstractmethod
    async def get_blast_radius(self, file_paths: list[str]) -> BlastRadiusMap: ...
    @abstractmethod
    async def get_call_graph(self, entry_points: list[str]) -> CallGraph: ...
    @abstractmethod
    async def get_dependency_subgraph(self, scope: list[str]) -> dict[str, object]: ...
