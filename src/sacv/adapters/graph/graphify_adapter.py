"""Graphify MCP adapter — connects to safishamsi/graphify MCP server."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from sacv.interfaces.cross_domain_provider import CrossDomainProvider

log = structlog.get_logger(__name__)

_DEFAULT_CMD  = ["graphify", "serve"]
_TIMEOUT_SEC  = 15


class GraphifyAdapter(CrossDomainProvider):

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
        log.info("graphify.started", pid=self._proc.pid)
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
        log.info("graphify.initialized")

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def __aenter__(self) -> "GraphifyAdapter":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def _call(self, tool: str, args: dict) -> dict:
        """MCP JSON-RPC stdio transport — mirrors AgentMemoryAdapter._call_tool."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            log.warning("graphify.not_started", tool=tool)
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
                    log.error("graphify.rpc_error", tool=tool, error=response["error"])
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
                log.error("graphify.transport_error", tool=tool, error=str(exc))
                return {}

    async def map_code_to_schema(self, entity_names: list[str]) -> dict:
        return await self._call("map_code_to_schema", {"entities": entity_names})

    async def get_arch_alignment(self, module_paths: list[str]) -> dict:
        return await self._call("arch_alignment", {"modules": module_paths})

    async def get_sql_impact(self, changed_files: list[str]) -> dict:
        return await self._call("sql_impact", {"files": changed_files})
