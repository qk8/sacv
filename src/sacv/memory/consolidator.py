"""Session consolidation and sleep-cycle compression."""
from __future__ import annotations
from sacv.interfaces.memory_provider import MemoryProvider

class SessionConsolidator:
    def __init__(self, memory: MemoryProvider) -> None:
        self._memory = memory

    async def run(self, session_id: str) -> None:
        await self._memory.purge_noise(session_id)
        await self._memory.consolidate_session(session_id)
