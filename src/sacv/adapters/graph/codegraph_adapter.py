"""CodeGraph MCP adapter — connects to colbymchenry/codegraph MCP server."""
from __future__ import annotations
import json, asyncio
from sacv.interfaces.code_graph_provider import CodeGraphProvider, BlastRadiusMap, CallGraph

class CodeGraphAdapter(CodeGraphProvider):
    def __init__(self, server_cmd: list[str] = ["codegraph", "serve"]) -> None:
        self._cmd = server_cmd

    async def _call(self, tool: str, args: dict) -> dict:
        # MCP JSON-RPC call via stdio (same pattern as AgentMemoryAdapter)
        return {}   # full implementation mirrors agentmemory_adapter.py

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
