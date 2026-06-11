"""
adapters/mcp_transport.py
=========================
Shared MCP (Model Context Protocol) JSON-RPC over stdio subprocess transport.

All three MCP adapters (AgentMemoryAdapter, CodeGraphAdapter, GraphifyAdapter)
use identical subprocess lifecycle, MCP handshake, reconnect logic, and
JSON-RPC request/response transport. This base class eliminates ~100 lines
of duplication across those three files.

Subclasses must define:
  - _log_prefix: str  (e.g. "agentmemory", "codegraph", "graphify")
  - _default_cmd: list[str]  (server binary + args)
  - _TIMEOUT_SEC: int  (per-operation timeout)

Usage:
    class MyAdapter(McpStdioTransport):
        _log_prefix = "myserver"
        _default_cmd = ["myserver", "serve"]
        _TIMEOUT_SEC = 10

        def my_tool(self, arg: str) -> dict:
            raw = await self._call("my_tool", {"arg": arg})
            return self._parse(raw)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class McpStdioTransport:
    """Async context manager for MCP JSON-RPC over stdio subprocess."""

    _log_prefix: str = "mcp"
    _default_cmd: list[str] = []
    _TIMEOUT_SEC: int = 15

    def __init__(self, server_cmd: list[str] | None = None) -> None:
        self._cmd = server_cmd or list(self._default_cmd)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._req_id = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info(f"{self._log_prefix}.started", pid=self._proc.pid)
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
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=self._TIMEOUT_SEC)
        raw = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=self._TIMEOUT_SEC,
        )
        try:
            response = json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self._log_prefix}: MCP initialize response was not JSON: "
                f"{raw[:200]!r}"
            ) from exc
        if "error" in response:
            raise RuntimeError(
                f"{self._log_prefix}: MCP initialize failed: {response['error']}"
            )
        if "result" not in response:
            raise RuntimeError(
                f"{self._log_prefix}: MCP initialize response missing 'result': "
                f"{response}"
            )
        server_info = response["result"].get("serverInfo", {})
        log.info(
            f"{self._log_prefix}.initialized",
            server_name=server_info.get("name", "unknown"),
            server_version=server_info.get("version", "unknown"),
        )
        initialized_notif = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        self._proc.stdin.write((json.dumps(initialized_notif) + "\n").encode())
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=self._TIMEOUT_SEC)

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def validate(self) -> None:
        """
        Check that the server binary is available and the MCP handshake succeeds.
        Raises RuntimeError with a clear message if anything is wrong.
        Call this once at startup before invoking any tool.
        """
        import shutil
        if self._cmd:
            binary = self._cmd[0]
            if not shutil.which(binary):
                raise RuntimeError(
                    f"{self._log_prefix}: server binary '{binary}' not found on PATH. "
                    f"Install it or set the correct path in your configuration."
                )
        # Start the subprocess and perform the handshake
        await self.start()
        log.info(f"{self._log_prefix}.validated")

    async def __aenter__(self) -> "McpStdioTransport":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── Reconnect ──────────────────────────────────────────────────────────

    async def _ensure_running(self) -> bool:
        """Return True if the subprocess is alive; attempt one reconnect if not."""
        if self._proc and self._proc.returncode is None:
            return True

        returncode = self._proc.returncode if self._proc else None

        # ── Capture crash log from stderr before discarding the process ───────
        crash_log = ""
        if self._proc and self._proc.stderr:
            try:
                crash_bytes = await asyncio.wait_for(
                    self._proc.stderr.read(4096), timeout=1.0
                )
                crash_log = crash_bytes.decode("utf-8", errors="replace").strip()
            except (asyncio.TimeoutError, Exception):
                pass

        log.error(f"{self._log_prefix}.process_dead",
                  returncode=returncode,
                  crash_log=crash_log[:500] if crash_log else "(empty)")
        try:
            await self.start()
            log.info(f"{self._log_prefix}.reconnected")
            return True
        except Exception as exc:
            log.error(f"{self._log_prefix}.reconnect_failed", error=str(exc))
            return False

    # ── JSON-RPC transport ────────────────────────────────────────────────

    async def _call(self, tool: str, args: dict[str, object]) -> Any:
        """
        Send a ``tools/call`` JSON-RPC request and return the parsed result.

        Attempts one reconnect if the subprocess is dead. Falls back to
        ``self._on_failure()`` on any error.
        """
        async with self._lock:
            # Check liveness inside the lock to prevent TOCTOU race
            # where subprocess dies between check and write
            alive = await self._ensure_running()
            if not alive:
                log.error(f"{self._log_prefix}.degraded_mode", tool=tool,
                          impact=f"{self._log_prefix} operations are no-ops this session")
                return self._on_failure()

            self._req_id += 1
            params: dict[str, object] = {"name": tool, "arguments": args}
            # OTEL-002: inject W3C traceparent for distributed tracing
            try:
                from sacv.tracing import get_traceparent
                _tp = get_traceparent()
                if _tp:
                    params["_meta"] = {"traceparent": _tp}
            except ImportError:
                pass
            request = {
                "jsonrpc": "2.0",
                "id":      self._req_id,
                "method":  "tools/call",
                "params":  params,
            }

            # _ensure_running guarantees _proc and its streams are not None
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            try:
                payload = json.dumps(request) + "\n"
                self._proc.stdin.write(payload.encode())
                await asyncio.wait_for(
                    self._proc.stdin.drain(), timeout=self._TIMEOUT_SEC
                )

                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=self._TIMEOUT_SEC
                )
                response = json.loads(raw.decode())

                if "error" in response:
                    log.error(f"{self._log_prefix}.rpc_error",
                              tool=tool, error=response["error"])
                    return self._on_failure()

                # MCP tool result is in result.content[0].text (JSON string)
                content = response.get("result", {}).get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                return self._on_failure()

            except OSError as exc:
                log.error(f"{self._log_prefix}.transport_error",
                          tool=tool, error=str(exc))
                # Subprocess may have died — attempt reconnect and retry
                reconnected = await self._ensure_running()
                if reconnected:
                    assert self._proc is not None
                    assert self._proc.stdin is not None
                    assert self._proc.stdout is not None
                    try:
                        payload = json.dumps(request) + "\n"
                        self._proc.stdin.write(payload.encode())
                        await asyncio.wait_for(
                            self._proc.stdin.drain(), timeout=self._TIMEOUT_SEC
                        )
                        raw = await asyncio.wait_for(
                            self._proc.stdout.readline(), timeout=self._TIMEOUT_SEC
                        )
                        response = json.loads(raw.decode())
                        if "error" in response:
                            return self._on_failure()
                        content = response.get("result", {}).get("content", [])
                        if content and isinstance(content, list):
                            text = content[0].get("text", "")
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                return text
                        return self._on_failure()
                    except OSError:
                        log.error(f"{self._log_prefix}.reconnect_retry_failed",
                                  tool=tool, error=str(exc))
                return self._on_failure()

            except (asyncio.TimeoutError, json.JSONDecodeError) as exc:
                log.error(f"{self._log_prefix}.transport_error",
                          tool=tool, error=str(exc))
                return self._on_failure()

    # ── Subclass hooks ─────────────────────────────────────────────────────

    def _on_failure(self) -> Any:
        """Return value when transport fails. Override in subclasses."""
        return None
