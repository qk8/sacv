"""
nodes/_log_parser.py
====================
Stack trace pruning and semantic mapping (approach 5 from debugging session).

Problem: Java stack traces are 150+ lines, 90% Spring/Hibernate internals.
TypeScript errors reference bundle paths, not source files.

Solution: Filter framework noise → keep user-package lines → return structured dicts.

Pure functions — no I/O, no LLM calls. Fully unit-testable.
Used by both the Verifier node and the IntelligentDebuggerNode.
"""
from __future__ import annotations

from typing import Any
import re
from dataclasses import dataclass

# ── Java framework packages to filter out ────────────────────────────────────
_JAVA_NOISE_PREFIXES = (
    "org.springframework.",
    "org.hibernate.",
    "com.sun.",
    "sun.",
    "jdk.",
    "java.lang.reflect.",
    "java.lang.Thread",
    "org.junit.",
    "org.mockito.",
    "net.bytebuddy.",
    "com.zaxxer.",           # HikariCP
    "org.apache.tomcat.",
    "org.apache.catalina.",
    "io.micrometer.",
    "ch.qos.logback.",
)

# Java "at" line pattern
_JAVA_AT_RE = re.compile(
    r"\s+at ([\w.$]+)\.([\w$<>]+)\((\w+\.java):(\d+)\)"
)
# Java exception header
_JAVA_EXCEPTION_RE = re.compile(
    r"^([\w.]+Exception|[\w.]+Error): (.+)$", re.MULTILINE
)
# TypeScript/Node error line (source-mapped or raw)
_TS_AT_RE = re.compile(
    r"\s+at (?:[\w.<>]+\s+)?\(?((?:src|components|pages|features|lib|app)/[^\s:)]+):(\d+):(\d+)\)?"
)
# Node bundle path (to detect unmapped lines)


@dataclass
class ParsedFrame:
    file:    str
    line:    int
    method:  str
    message: str = ""


def prune_java_stack(
    raw_output:   str,
    user_package: str = "com.example",
) -> list[ParsedFrame]:
    """
    Pure function. Filters a Java stack trace to user-package frames only.

    Returns structured ParsedFrame list sorted from innermost to outermost
    user-code frame.
    """
    frames: list[ParsedFrame] = []
    current_message = ""

    # Extract exception message
    m = _JAVA_EXCEPTION_RE.search(raw_output)
    if m:
        current_message = f"{m.group(1)}: {m.group(2)}"

    for match in _JAVA_AT_RE.finditer(raw_output):
        qualified_class = match.group(1)
        method          = match.group(2)
        filename        = match.group(3)
        line_no         = int(match.group(4))

        # Skip framework noise
        if any(qualified_class.startswith(p) for p in _JAVA_NOISE_PREFIXES):
            continue
        # Keep only user package
        if not qualified_class.startswith(user_package):
            continue

        frames.append(ParsedFrame(
            file=filename,
            line=line_no,
            method=f"{qualified_class}.{method}",
            message=current_message,
        ))

    return frames


def prune_typescript_stack(
    raw_output:   str,
    src_root:     str = "src",
) -> list[ParsedFrame]:
    """
    Pure function. Filters a TypeScript/Node.js stack trace to source files.

    Keeps only frames that reference the project's source root.
    Skips node_modules and bundled dist/ files.
    """
    frames: list[ParsedFrame] = []
    current_message = _extract_ts_message(raw_output)

    for match in _TS_AT_RE.finditer(raw_output):
        filepath = match.group(1)
        line_no  = int(match.group(2))

        # Skip bundle/dist/node_modules
        if any(seg in filepath for seg in ("/dist/", "/.next/", "/node_modules/")):
            continue

        frames.append(ParsedFrame(
            file=filepath,
            line=line_no,
            method=_extract_ts_method_name(raw_output, filepath),
            message=current_message,
        ))

    return frames


def prune_stack(
    raw_output:   str,
    module_type:  str,
    user_package: str = "com.example",
    src_root:     str = "src",
) -> list[ParsedFrame]:
    """Dispatch to the correct pruner based on module_type."""
    if "frontend" in module_type:
        return prune_typescript_stack(raw_output, src_root)
    return prune_java_stack(raw_output, user_package)


def frames_to_dict(frames: list[ParsedFrame]) -> list[dict[str, Any]]:
    return [
        {"file": f.file, "line": f.line, "method": f.method, "message": f.message}
        for f in frames
    ]


def format_for_actor(frames: list[ParsedFrame]) -> str:
    """Format pruned frames as a compact, token-efficient string for the Actor prompt."""
    if not frames:
        return "(no user-code frames found in stack trace)"
    return "\n".join(
        f"  → {f.method}({f.file}:{f.line})"
        + (f" — {f.message}" if f.message and i == 0 else "")
        for i, f in enumerate(frames)
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_ts_message(raw: str) -> str:
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("at ") and not stripped.startswith("Error:"):
            if "Error" in stripped or "TypeError" in stripped or "ReferenceError" in stripped:
                return stripped[:200]
    return ""


def _extract_ts_method_name(raw: str, filepath: str) -> str:
    """Try to extract the function name from the line before the 'at' frame."""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if filepath in line and i > 0:
            prev = lines[i - 1].strip()
            if prev and "at " in prev:
                parts = prev.split("at ")
                if len(parts) > 1:
                    return parts[1].split("(")[0].strip()
    return "unknown"
