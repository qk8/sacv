"""
adapters/debug/jdwp_client.py
==============================
Java debugger client using JDB (Java Debugger — built into JDK).

JDB is the command-line interface to JDWP. It's included in every JDK
installation, requires zero extra dependencies, and is immediately available
inside our Docker sandbox.

Protocol: subprocess stdin/stdout — JDB reads commands line by line and
prints results as text. This is intentionally simpler than implementing
raw JDWP (which would require 2 000+ lines of protocol code).

For teams that need lower-latency debug sessions, the same interface
contract can be satisfied by a DAP-over-JDWP adapter (java-debug) — the
IntelligentDebuggerNode only calls JdwpClient methods, not JDB directly,
so the underlying implementation is swappable.

Usage:
    async with JdwpClient("localhost", 5005) as jdb:
        await jdb.set_breakpoint("UserService", 42)
        await jdb.run()
        hit = await jdb.wait_for_breakpoint_hit()
        vars = await jdb.get_local_variables()
        stack = await jdb.get_call_stack()
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_PROMPT = ">"          # JDB command prompt
_TIMEOUT = 30.0        # seconds to wait for breakpoint hit
_READ_TIMEOUT = 2.0    # seconds to wait for command output


@dataclass
class BreakpointHitInfo:
    class_name:  str
    method_name: str
    file:        str
    line:        int
    thread_name: str = "main"


@dataclass
class LocalVariable:
    name:  str
    value: str
    type:  str


class JdwpClient:
    """
    Async JDB wrapper for Java step-through debugging inside Docker sandbox.

    The sandbox JVM must be started with JDWP suspended:
        java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:5005
    """

    def __init__(self, host: str = "localhost", port: int = 5005) -> None:
        self._host    = host
        self._port    = port
        self._process: asyncio.subprocess.Process | None = None

    async def __aenter__(self) -> "JdwpClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Start JDB and attach to the remote JDWP process."""
        self._process = await asyncio.create_subprocess_exec(
            "jdb", "-attach", f"{self._host}:{self._port}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Consume JDB startup banner
        await self._read_until_prompt(timeout=10.0)
        log.info("jdwp.connected", host=self._host, port=self._port)

    async def close(self) -> None:
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(b"quit\n")
                await self._process.stdin.drain()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except Exception:
                self._process.kill()

    # ── Breakpoint control ────────────────────────────────────────────────────

    async def set_breakpoint_at_line(
        self, class_name: str, line: int
    ) -> bool:
        """
        Set a breakpoint at a specific line.
        class_name: fully qualified or simple class name (e.g. 'UserService')
        """
        output = await self._cmd(f"stop at {class_name}:{line}")
        success = "Deferring breakpoint" in output or "Set breakpoint" in output
        log.debug("jdwp.breakpoint_set", class_name=class_name, line=line, ok=success)
        return success

    async def set_breakpoint_at_method(
        self, class_name: str, method_name: str
    ) -> bool:
        output = await self._cmd(f"stop in {class_name}.{method_name}")
        return "Set breakpoint" in output or "Deferring" in output

    # ── Execution control ─────────────────────────────────────────────────────

    async def run(self) -> str:
        """Resume execution (equivalent to IDE 'Resume')."""
        return await self._cmd("cont", timeout=_TIMEOUT)

    async def step_over(self) -> BreakpointHitInfo | None:
        """Execute the current line and stop at the next line."""
        output = await self._cmd("next", timeout=_TIMEOUT)
        return _parse_hit(output)

    async def step_into(self) -> BreakpointHitInfo | None:
        """Step into the method call on the current line."""
        output = await self._cmd("step", timeout=_TIMEOUT)
        return _parse_hit(output)

    async def step_out(self) -> BreakpointHitInfo | None:
        """Step out of the current method."""
        output = await self._cmd("step up", timeout=_TIMEOUT)
        return _parse_hit(output)

    async def wait_for_breakpoint_hit(
        self, timeout: float = _TIMEOUT
    ) -> BreakpointHitInfo | None:
        """Wait for a breakpoint to be hit after resuming."""
        try:
            output = await self._read_until_prompt(timeout=timeout)
            return _parse_hit(output)
        except asyncio.TimeoutError:
            log.warning("jdwp.breakpoint_timeout", timeout=timeout)
            return None

    # ── Inspection ────────────────────────────────────────────────────────────

    async def get_local_variables(self) -> list[LocalVariable]:
        """Return all local variables in the current frame."""
        output = await self._cmd("locals")
        return _parse_locals(output)

    async def get_call_stack(self) -> list[str]:
        """Return the current call stack as a list of strings."""
        output = await self._cmd("where")
        return _parse_where(output)

    async def evaluate(self, expression: str) -> str:
        """Evaluate a Java expression in the current scope."""
        output = await self._cmd(f"print {expression}")
        # JDB returns: "expression = value\n> " — strip trailing prompt
        output = re.sub(r"\s*>\s*$", "", output)
        parts = output.split(" = ", 1)
        return parts[1].strip() if len(parts) == 2 else output.strip()

    async def get_thread_list(self) -> list[str]:
        output = await self._cmd("threads")
        # Filter out JDB prompt lines (just ">")
        return [
            line.strip()
            for line in output.splitlines()
            if line.strip() and not re.match(r"^>\s*$", line.strip())
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _cmd(
        self, command: str, timeout: float = _READ_TIMEOUT
    ) -> str:
        assert self._process and self._process.stdin
        self._process.stdin.write(f"{command}\n".encode())
        await self._process.stdin.drain()
        return await self._read_until_prompt(timeout=timeout)

    async def _read_until_prompt(self, timeout: float = _READ_TIMEOUT) -> str:
        assert self._process and self._process.stdout
        buf = b""
        try:
            async with asyncio.timeout(timeout):
                while True:
                    chunk = await self._process.stdout.read(256)
                    if not chunk:
                        break
                    buf += chunk
                    if buf.rstrip().endswith(b">"):
                        break
        except asyncio.TimeoutError:
            pass
        return buf.decode(errors="replace")


# ── Output parsers (pure functions) ──────────────────────────────────────────

def _parse_hit(output: str) -> BreakpointHitInfo | None:
    """Parse JDB output to find a breakpoint/step hit location."""
    # Pattern: "Breakpoint hit: thread=main(t1), UserService.findById(), line=42 bci=..."
    # Thread name may be quoted or unquoted; captures up to comma/whitespace/paren
    m = re.search(
        r'thread="?([^",\s(]+)"?.*?([\w.$]+)\.([\w<>]+)\(\).*?line=(\d+)',
        output, re.DOTALL
    )
    if not m:
        return None
    class_part = m.group(2).split('.')[-1]
    # Strip inner class suffix ($Builder → outer class file)
    class_part = class_part.split('$')[0]
    return BreakpointHitInfo(
        thread_name=m.group(1),
        class_name=m.group(2),
        method_name=m.group(3),
        file=f"{class_part}.java",
        line=int(m.group(4)),
    )


def _parse_locals(output: str) -> list[LocalVariable]:
    """Parse JDB 'locals' output."""
    variables: list[LocalVariable] = []
    for line in output.splitlines():
        # Pattern: "  variableName = value (type)"
        m = re.match(r"\s+(\w+) = (.+?)(?:\s+\((\w+)\))?$", line)
        if m:
            variables.append(LocalVariable(
                name=m.group(1),
                value=m.group(2).strip(),
                type=m.group(3) or "unknown",
            ))
    return variables


def _parse_where(output: str) -> list[str]:
    """Parse JDB 'where' (stack trace) output."""
    frames = []
    for line in output.splitlines():
        # Pattern: "  [1] ClassName.method(ClassName.java:42)"
        m = re.match(r"\s+\[\d+\] (.+?)\((.+):(\d+)\)", line)
        if m:
            frames.append(f"{m.group(1)}({m.group(2)}:{m.group(3)})")
    return frames
