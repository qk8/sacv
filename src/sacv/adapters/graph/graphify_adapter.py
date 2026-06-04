"""Graphify MCP adapter — connects to safishamsi/graphify MCP server."""
from __future__ import annotations

from sacv.adapters.mcp_transport import McpStdioTransport
from sacv.interfaces.cross_domain_provider import CrossDomainProvider


_DEFAULT_CMD = ["graphify", "serve"]


class GraphifyAdapter(McpStdioTransport, CrossDomainProvider):

    _log_prefix = "graphify"
    _default_cmd = _DEFAULT_CMD
    _TIMEOUT_SEC = 15

    def _on_failure(self) -> dict:
        return {}

    async def map_code_to_schema(self, entity_names: list[str]) -> dict:
        return await self._call("map_code_to_schema", {"entities": entity_names})

    async def get_arch_alignment(self, module_paths: list[str]) -> dict:
        return await self._call("arch_alignment", {"modules": module_paths})

    async def get_sql_impact(self, changed_files: list[str]) -> dict:
        return await self._call("sql_impact", {"files": changed_files})
