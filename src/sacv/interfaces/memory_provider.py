from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EpisodicEvent:
    session_id: str
    event_type: str
    payload: dict
    timestamp: str

@dataclass
class ProceduralConstraint:
    constraint_id: str
    category: str
    description: str
    weight: float

@dataclass
class LessonLearned:
    task_id: str
    pattern_discovered: str
    negative_constraints: list[ProceduralConstraint]
    blast_radius_learned: dict
    correction_type: str

class MemoryProvider(ABC):
    @abstractmethod
    async def store_episodic(self, event: EpisodicEvent) -> None: ...
    @abstractmethod
    async def retrieve_procedural(self, context_tags: list[str]) -> list[ProceduralConstraint]: ...
    @abstractmethod
    async def consolidate_session(self, session_id: str) -> LessonLearned: ...
    @abstractmethod
    async def purge_noise(self, session_id: str) -> None: ...
