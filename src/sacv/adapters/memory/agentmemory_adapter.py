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
        await self._initialize()

    async def _initialize(self) -> None:
        """Perform the MCP initialize / initialized lifecycle handshake."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return
        # Step 1: send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sacv", "version": "0.1.0"},
            },
        }
        self._proc.stdin.write((json.dumps(init_request) + "\n").encode())
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=_TIMEOUT_SEC)
        # Step 2: read server's initialize response
        await asyncio.wait_for(self._proc.stdout.readline(), timeout=_TIMEOUT_SEC)
        # Step 3: send initialized notification (no response expected)
        initialized_notif = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        self._proc.stdin.write((json.dumps(initialized_notif) + "\n").encode())
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=_TIMEOUT_SEC)
        log.info("agentmemory.initialized")

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
        await self._call_tool("create_memory", {
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
        result = await self._call_tool("search_memory", {
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

    async def purge_noise(self, session_id: str) -> None:
        """Delete intermediate failed-attempt events to prevent memory pollution."""
        await self._call_tool("delete_memory", {
            "filter": {
                "session_id": session_id,
                "event_type": {"$in": ["actor_attempt", "critic_finding_temp"]},
            }
        })
        log.debug("agentmemory.noise_purged", session_id=session_id)

    # ── MCP JSON-RPC transport ────────────────────────────────────────────

    async def _ensure_running(self) -> bool:
        """Return True if the subprocess is alive; attempt one reconnect if not."""
        if self._proc and self._proc.returncode is None:
            return True
        log.error("agentmemory.process_dead",
                  returncode=self._proc.returncode if self._proc else None)
        try:
            await self.start()
            log.info("agentmemory.reconnected")
            return True
        except Exception as exc:
            log.error("agentmemory.reconnect_failed", error=str(exc))
            return False

    async def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Sends a ``tools/call`` JSON-RPC request to the MCP server
        and returns the parsed result.

        Attempts one reconnect if the subprocess is dead. Falls back to None
        on any error (degraded mode — no crash).
        """
        alive = await self._ensure_running()
        if not alive:
            log.error("agentmemory.degraded_mode", tool=tool_name,
                      impact="memory operations are no-ops this session")
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
