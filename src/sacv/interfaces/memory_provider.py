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

class MemoryProvider(ABC):
    @abstractmethod
    async def store_episodic(self, event: EpisodicEvent) -> None: ...
    @abstractmethod
    async def retrieve_procedural(self, context_tags: list[str]) -> list[ProceduralConstraint]: ...
    # NOTE: consolidate_session was removed — the real LessonLearned is
    # written via store_episodic by the memory_consolidation node.
    @abstractmethod
    async def purge_noise(self, session_id: str) -> None: ...
