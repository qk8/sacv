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
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from sacv.adapters.mcp_transport import McpStdioTransport
from sacv.interfaces.memory_provider import (
    MemoryProvider,
    EpisodicEvent,
    ProceduralConstraint,
)

_SERVER_CMD = ["agentmemory", "serve", "--transport", "stdio"]


class AgentMemoryAdapter(McpStdioTransport, MemoryProvider):

    _log_prefix = "agentmemory"
    _default_cmd = _SERVER_CMD
    _TIMEOUT_SEC = 10

    def _on_failure(self) -> None:
        return None

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

        return constraints

    async def purge_noise(self, session_id: str) -> None:
        """Delete intermediate failed-attempt events to prevent memory pollution."""
        for event_type in ("actor_attempt", "critic_finding_temp"):
            await self._call_tool("delete_memory", {
                "filter": {
                    "session_id": session_id,
                    "event_type": event_type,
                }
            })

    # ── Custom _call_tool: AgentMemory returns raw text for some tools ────

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Wrapper around ``_call`` that logs the tool name in degraded mode.

        The base ``_call`` returns ``None`` on failure (from ``_on_failure``);
        domain methods handle ``None`` gracefully.
        """
        return await self._call(tool_name, arguments)
