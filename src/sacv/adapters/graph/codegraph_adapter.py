"""CodeGraph MCP adapter — connects to colbymchenry/codegraph MCP server."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from sacv.interfaces.code_graph_provider import (
    CodeGraphProvider, BlastRadiusMap, CallGraph,
)

log = structlog.get_logger(__name__)

_DEFAULT_CMD  = ["codegraph", "serve"]
_TIMEOUT_SEC  = 15


class CodeGraphAdapter(CodeGraphProvider):

    def __init__(self, server_cmd: list[str] = _DEFAULT_CMD) -> None:
        self._cmd  = server_cmd
        self._proc: asyncio.subprocess.Process | None = None
        self._lock  = asyncio.Lock()
        self._req_id = 0

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info("codegraph.started", pid=self._proc.pid)
        await self._initialize()

    async def _initialize(self) -> None:
        """Perform the MCP initialize / initialized lifecycle handshake."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return
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
        await asyncio.wait_for(self._proc.stdout.readline(), timeout=_TIMEOUT_SEC)
        initialized_notif = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        self._proc.stdin.write((json.dumps(initialized_notif) + "\n").encode())
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=_TIMEOUT_SEC)
        log.info("codegraph.initialized")

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def __aenter__(self) -> "CodeGraphAdapter":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def _call(self, tool: str, args: dict) -> dict:
        """MCP JSON-RPC stdio transport — mirrors AgentMemoryAdapter._call_tool."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            log.warning("codegraph.not_started", tool=tool)
            return {}
        async with self._lock:
            self._req_id += 1
            request = {
                "jsonrpc": "2.0",
                "id":      self._req_id,
                "method":  "tools/call",
                "params":  {"name": tool, "arguments": args},
            }
            try:
                payload = json.dumps(request) + "\n"
                self._proc.stdin.write(payload.encode())
                await asyncio.wait_for(self._proc.stdin.drain(), timeout=_TIMEOUT_SEC)
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=_TIMEOUT_SEC
                )
                response = json.loads(raw.decode())
                if "error" in response:
                    log.error("codegraph.rpc_error", tool=tool, error=response["error"])
                    return {}
                content = response.get("result", {}).get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {}
                return {}
            except (asyncio.TimeoutError, json.JSONDecodeError, OSError) as exc:
                log.error("codegraph.transport_error", tool=tool, error=str(exc))
                return {}

    async def get_blast_radius(self, file_paths: list[str]) -> BlastRadiusMap:
        raw = await self._call("blast_radius", {"files": file_paths})
        return BlastRadiusMap(
            entry_files=raw.get("entry_files", file_paths),
            affected_files=raw.get("affected_files", []),
            dependency_depth=raw.get("depth", 0),
            cross_service_impact=raw.get("cross_service", []),
            schema_impact=raw.get("schema_impact", []),
            risk_score=float(raw.get("risk_score", 0.0)),
        )

    async def get_call_graph(self, entry_points: list[str]) -> CallGraph:
        raw = await self._call("call_graph", {"entry_points": entry_points})
        return CallGraph(
            entry_point=entry_points[0] if entry_points else ".",
            nodes=raw.get("nodes", []),
            edges=raw.get("edges", []),
        )

    async def get_dependency_subgraph(self, scope: list[str]) -> dict:
        return await self._call("dependency_subgraph", {"scope": scope})
