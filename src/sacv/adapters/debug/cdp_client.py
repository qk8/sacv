"""
adapters/debug/cdp_client.py
=============================
Node.js / TypeScript debugger using Chrome DevTools Protocol (CDP).

Node.js has a built-in V8 Inspector that speaks CDP over a WebSocket.
Start Node with:
    node --inspect-brk=0.0.0.0:9229 dist/app.js

This client connects to that WebSocket and sends CDP JSON commands.

CDP reference: https://chromedevtools.github.io/devtools-protocol/

No external dependencies beyond 'websockets' (already in requirements).

Usage:
    async with CdpClient("localhost", 9229) as cdp:
        await cdp.enable_debugger()
        bp = await cdp.set_breakpoint_by_url("src/services/UserService.ts", 42)
        await cdp.resume()
        hit = await cdp.wait_for_paused()
        scope = await cdp.get_scope_variables(hit.call_frame_id)
        stack = hit.call_frames
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import structlog

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

log = structlog.get_logger(__name__)


@dataclass
class CallFrame:
    frame_id:    str
    function:    str
    url:         str
    line:        int
    column:      int
    scope_chain: list[dict] = field(default_factory=list)


@dataclass
class PausedEvent:
    reason:      str
    call_frames: list[CallFrame]
    hit_breakpoints: list[str] = field(default_factory=list)

    @property
    def call_frame_id(self) -> str:
        return self.call_frames[0].frame_id if self.call_frames else ""


class CdpClient:
    """
    Async CDP client for TypeScript/Node.js debugging.

    Connects to the V8 Inspector WebSocket exposed by Node.js.
    """

    def __init__(self, host: str = "localhost", port: int = 9229) -> None:
        self._host = host
        self._port = port
        self._ws   = None
        self._id   = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._paused_future: asyncio.Future | None = None
        self._recv_task: asyncio.Task | None = None

    async def __aenter__(self) -> "CdpClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        if not _WS_AVAILABLE:
            raise ImportError("websockets library required: pip install websockets")

        ws_url = await self._discover_ws_url()
        self._ws = await websockets.connect(ws_url, max_size=10 * 1024 * 1024)
        self._recv_task = asyncio.create_task(self._receive_loop())
        log.info("cdp.connected", host=self._host, port=self._port)

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()

    # ── Debugger lifecycle ────────────────────────────────────────────────────

    async def enable_debugger(self) -> None:
        await self._send("Debugger.enable")
        await self._send("Runtime.enable")

    async def disable_debugger(self) -> None:
        await self._send("Debugger.disable")

    # ── Breakpoints ───────────────────────────────────────────────────────────

    async def set_breakpoint_by_url(
        self, url_or_file: str, line: int, column: int = 0
    ) -> str:
        """
        Set a breakpoint at (url_or_file, line).
        url_or_file: can be a file path like 'src/services/UserService.ts'
        Returns breakpoint ID.
        """
        result = await self._send("Debugger.setBreakpointByUrl", {
            "lineNumber": line - 1,  # CDP is 0-indexed
            "url":        url_or_file,
            "columnNumber": column,
        })
        bp_id = result.get("breakpointId", "")
        log.debug("cdp.breakpoint_set", url=url_or_file, line=line, id=bp_id)
        return bp_id

    async def remove_breakpoint(self, bp_id: str) -> None:
        await self._send("Debugger.removeBreakpoint", {"breakpointId": bp_id})

    # ── Execution control ─────────────────────────────────────────────────────

    async def resume(self) -> None:
        await self._send("Debugger.resume")

    async def step_over(self) -> PausedEvent | None:
        await self._send("Debugger.stepOver")
        return await self.wait_for_paused()

    async def step_into(self) -> PausedEvent | None:
        await self._send("Debugger.stepInto")
        return await self.wait_for_paused()

    async def step_out(self) -> PausedEvent | None:
        await self._send("Debugger.stepOut")
        return await self.wait_for_paused()

    async def wait_for_paused(self, timeout: float = 30.0) -> PausedEvent | None:
        loop = asyncio.get_running_loop()
        self._paused_future = loop.create_future()
        try:
            result = await asyncio.wait_for(
                asyncio.shield(self._paused_future), timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            log.warning("cdp.wait_for_paused_timeout")
            return None
        finally:
            self._paused_future = None

    # ── Variable inspection ───────────────────────────────────────────────────

    async def get_scope_variables(
        self, call_frame_id: str
    ) -> dict[str, Any]:
        """
        Get all variables in scope for a given call frame.
        Returns {name: {value, type, description}} dict.
        """
        frame_result = await self._send("Runtime.getProperties", {
            "objectId":      call_frame_id,
            "ownProperties": True,
        })
        variables = {}
        for scope in frame_result.get("result", []):
            name = scope.get("name", "?")
            val  = scope.get("value", {})
            variables[name] = {
                "value":       val.get("value", val.get("description", "?")),
                "type":        val.get("type", "unknown"),
                "description": val.get("description", ""),
            }
        return variables

    async def get_scope_variables_from_paused(
        self, paused: PausedEvent
    ) -> dict[str, Any]:
        """Convenience: get variables for the top frame of a PausedEvent."""
        if not paused.call_frames:
            return {}
        frame  = paused.call_frames[0]
        result = {}
        for scope in frame.scope_chain:
            obj_id = scope.get("object", {}).get("objectId")
            if not obj_id:
                continue
            props = await self._send("Runtime.getProperties", {
                "objectId": obj_id,
                "ownProperties": True,
            })
            for prop in props.get("result", []):
                n = prop.get("name", "?")
                v = prop.get("value", {})
                result[n] = {
                    "value":       v.get("value", v.get("description", "?")),
                    "type":        v.get("type", "unknown"),
                    "description": v.get("description", ""),
                }
        return result

    async def evaluate_in_frame(
        self, expression: str, call_frame_id: str
    ) -> Any:
        """Evaluate a JavaScript expression in the context of a call frame."""
        result = await self._send("Debugger.evaluateOnCallFrame", {
            "callFrameId": call_frame_id,
            "expression":  expression,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value")

    async def evaluate(self, expression: str) -> Any:
        """Evaluate a JavaScript expression in the global context."""
        result = await self._send("Runtime.evaluate", {
            "expression":    expression,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        msg_id  = self._id
        message = {"id": msg_id, "method": method, "params": params or {}}
        loop    = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[msg_id] = future

        assert self._ws is not None
        await self._ws.send(json.dumps(message))

        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=10.0)
            return result
        except asyncio.TimeoutError:
            log.warning("cdp.send_timeout", method=method)
            return {}

    async def _receive_loop(self) -> None:
        """Background task that dispatches incoming CDP messages."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                # Response to a command
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(str(msg["error"])))
                        else:
                            fut.set_result(msg.get("result", {}))
                # Event
                elif msg.get("method") == "Debugger.paused":
                    event = _parse_paused(msg.get("params", {}))
                    if self._paused_future and not self._paused_future.done():
                        self._paused_future.set_result(event)
        except Exception as exc:
            log.debug("cdp.receive_loop_ended", reason=str(exc))

    async def _discover_ws_url(self) -> str:
        """Query Node.js /json endpoint to get the WebSocket URL."""
        import urllib.request as _urllib

        def _blocking_get() -> str:
            try:
                url  = f"http://{self._host}:{self._port}/json"
                resp = _urllib.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                if data:
                    return data[0]["webSocketDebuggerUrl"]
            except Exception:
                pass
            return f"ws://{self._host}:{self._port}"

        return await asyncio.to_thread(_blocking_get)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_paused(params: dict) -> PausedEvent:
    frames = []
    for f in params.get("callFrames", []):
        loc = f.get("location", {})
        frames.append(CallFrame(
            frame_id=f.get("callFrameId", ""),
            function=f.get("functionName", "<anonymous>"),
            url=f.get("url", ""),
            line=loc.get("lineNumber", 0) + 1,   # back to 1-indexed
            column=loc.get("columnNumber", 0),
            scope_chain=f.get("scopeChain", []),
        ))
    return PausedEvent(
        reason=params.get("reason", "breakpoint"),
        call_frames=frames,
        hit_breakpoints=params.get("hitBreakpoints", []),
    )
