"""
Shadow mode: idle-period background fuzzing inside Docker sandbox.
Discovers edge cases autonomously; writes findings to AgentMemory.
Scheduled non-blockingly after memory_consolidation completes.
"""
from __future__ import annotations
import asyncio
import structlog
from sacv.interfaces.sandbox_provider import SandboxProvider
from sacv.interfaces.memory_provider import MemoryProvider, EpisodicEvent
from datetime import datetime, timezone

log = structlog.get_logger(__name__)

async def run_shadow_mode(
    sandbox: SandboxProvider,
    memory:  MemoryProvider,
    session_id: str,
    fuzz_commands: list[str] | None = None,
) -> None:
    """
    Run in the background (asyncio.create_task) after task completion.
    Executes lightweight fuzz/lint commands inside the warm sandbox and
    stores any new findings as episodic events.
    """
    commands = fuzz_commands or [
        "mvn test -Dtest=*FuzzTest -q 2>&1 | tail -20",
        "npm run lint:strict 2>&1 | tail -20",
    ]
    try:
        handle = await sandbox.warm_container()
        for cmd in commands:
            result = await sandbox.exec_in_container(handle, cmd, timeout=60)
            if result.exit_code != 0 and result.stdout:
                await memory.store_episodic(EpisodicEvent(
                    session_id=session_id,
                    event_type="shadow_finding",
                    payload={"command": cmd, "output": result.stdout[:500]},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                log.info("shadow_mode.finding", cmd=cmd[:40])
    except Exception as exc:
        log.warning("shadow_mode.error", error=str(exc))
