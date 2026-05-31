"""
adapters/memory/agentmemory_adapter.py
=======================================
Concrete implementation of MemoryProvider backed by the agentmemory MCP server
(https://github.com/rohitg00/agentmemory).

The MCP server exposes three memory stores:
- episodic:   timestamped event log (what happened)
- semantic:   indexed knowledge (codebase patterns, architecture facts)
- procedural: learned constraints (negative prompts, correction rules)

This adapter communicates with the MCP server via stdio subprocess, following
the standard MCP client protocol.

If the MCP server is unavailable (e.g. in offline tests), the adapter
degrades gracefully to a no-op in-memory fallback.
"""
from __future__ import annotations

import json
import subprocess
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import structlog

from sacv.interfaces.memory_provider import (
    MemoryProvider,
    EpisodicEvent,
    ProceduralConstraint,
    LessonLearned,
)

log = structlog.get_logger(__name__)

# MCP server command — assumes agentmemory is installed and on PATH
_SERVER_CMD  = ["agentmemory", "serve", "--transport", "stdio"]
_TIMEOUT_SEC = 10


class AgentMemoryAdapter(MemoryProvider):
    """
    Routes memory operations to the agentmemory MCP server.

    Lifecycle:
        adapter = AgentMemoryAdapter()
        await adapter.start()
        ...
        await adapter.stop()

    Or use as an async context manager:
        async with AgentMemoryAdapter() as mem:
            ...
    """

    def __init__(self, server_cmd: list[str] = _SERVER_CMD) -> None:
        self._server_cmd = server_cmd
        self._proc:  asyncio.subprocess.Process | None = None
        self._lock   = asyncio.Lock()
        self._req_id = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._server_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info("agentmemory.started", pid=self._proc.pid)

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()
            log.info("agentmemory.stopped")

    async def __aenter__(self) -> "AgentMemoryAdapter":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── MemoryProvider interface ───────────────────────────────────────────

    async def store_episodic(self, event: EpisodicEvent) -> None:
        await self._call_tool("memory_store", {
            "type":      "episodic",
            "content":   json.dumps(asdict(event)),
            "metadata":  {
                "session_id":  event.session_id,
                "event_type":  event.event_type,
                "timestamp":   event.timestamp,
            },
        })

    async def retrieve_procedural(
        self, context_tags: list[str]
    ) -> list[ProceduralConstraint]:
        result = await self._call_tool("memory_search", {
            "type":  "procedural",
            "query": " ".join(context_tags),
            "limit": 20,
        })

        constraints: list[ProceduralConstraint] = []
        for item in (result or []):
            try:
                data = json.loads(item.get("content", "{}"))
                constraints.append(ProceduralConstraint(
                    constraint_id=data.get("constraint_id", item.get("id", "?")),
                    category=data.get("category", "general"),
                    description=data.get("description", item.get("content", "")),
                    weight=float(data.get("weight", 1.0)),
                ))
            except Exception:
                pass

        log.debug("agentmemory.retrieved_procedural", count=len(constraints))
        return constraints

    async def consolidate_session(self, session_id: str) -> LessonLearned:
        """
        Compress episodic events for this session into a summary.
        Delegates the summarisation to the MCP server's built-in
        compression endpoint.
        """
        result = await self._call_tool("memory_compress", {
            "session_id": session_id,
            "keep_types": ["lesson_learned"],
        })
        # Return a minimal LessonLearned — the real one was written by the node
        return LessonLearned(
            task_id=session_id,
            pattern_discovered=str(result or ""),
            negative_constraints=[],
            blast_radius_learned={},
            correction_type="consolidated",
            session_duration_ms=0,    # MCP server doesn't report duration; node computes it
        )

    async def purge_noise(self, session_id: str) -> None:
        """Delete intermediate failed-attempt events to prevent memory pollution."""
        await self._call_tool("memory_delete", {
            "filter": {
                "session_id": session_id,
                "event_type": {"$in": ["actor_attempt", "critic_finding_temp"]},
            }
        })
        log.debug("agentmemory.noise_purged", session_id=session_id)

    # ── MCP JSON-RPC transport ────────────────────────────────────────────

    async def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Sends a ``tools/call`` JSON-RPC request to the MCP server
        and returns the parsed result.

        Falls back to None on any error (degraded mode — no crash).
        """
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            log.warning("agentmemory.not_started", tool=tool_name)
            return None

        async with self._lock:
            self._req_id += 1
            request = {
                "jsonrpc": "2.0",
                "id":      self._req_id,
                "method":  "tools/call",
                "params":  {
                    "name":      tool_name,
                    "arguments": arguments,
                },
            }

            try:
                payload = json.dumps(request) + "\n"
                self._proc.stdin.write(payload.encode())
                await asyncio.wait_for(
                    self._proc.stdin.drain(), timeout=_TIMEOUT_SEC
                )

                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=_TIMEOUT_SEC
                )
                response = json.loads(raw.decode())

                if "error" in response:
                    log.error(
                        "agentmemory.rpc_error",
                        tool=tool_name,
                        error=response["error"],
                    )
                    return None

                # MCP tool result is in result.content[0].text (JSON string)
                content = response.get("result", {}).get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                return None

            except (asyncio.TimeoutError, json.JSONDecodeError, OSError) as exc:
                log.error("agentmemory.transport_error", tool=tool_name, error=str(exc))
                return None
