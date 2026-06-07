"""CodeGraph MCP adapter — connects to colbymchenry/codegraph MCP server."""
from __future__ import annotations

from typing import Any

from sacv.adapters.mcp_transport import McpStdioTransport
from sacv.interfaces.code_graph_provider import (
    CodeGraphProvider, BlastRadiusMap, CallGraph,
)


_DEFAULT_CMD = ["codegraph", "serve"]


class CodeGraphAdapter(McpStdioTransport, CodeGraphProvider):

    _log_prefix = "codegraph"
    _default_cmd = _DEFAULT_CMD
    _TIMEOUT_SEC = 15

    def _on_failure(self) -> dict[str, Any]:
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

    async def get_dependency_subgraph(self, scope: list[str]) -> dict[str, Any]:
        return await self._call("dependency_subgraph", {"scope": scope})  # type: ignore[no-any-return]
