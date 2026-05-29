"""Graphify MCP adapter — connects to safishamsi/graphify MCP server."""
from __future__ import annotations
from sacv.interfaces.cross_domain_provider import CrossDomainProvider

class GraphifyAdapter(CrossDomainProvider):
    def __init__(self, server_cmd: list[str] = ["graphify", "serve"]) -> None:
        self._cmd = server_cmd

    async def _call(self, tool: str, args: dict) -> dict:
        return {}   # full implementation mirrors agentmemory_adapter.py

    async def map_code_to_schema(self, entity_names: list[str]) -> dict:
        return await self._call("map_code_to_schema", {"entities": entity_names})

    async def get_arch_alignment(self, module_paths: list[str]) -> dict:
        return await self._call("arch_alignment", {"modules": module_paths})

    async def get_sql_impact(self, changed_files: list[str]) -> dict:
        return await self._call("sql_impact", {"files": changed_files})
