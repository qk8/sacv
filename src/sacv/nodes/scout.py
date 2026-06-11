"""
nodes/scout.py
==============
Builds a tight ContextSkeleton by querying CodeGraph + Graphify MCP.

Refactoring additions (approaches 3, 5, 11):
  - Reads AGENTS.md if it exists and injects it as agents_md_context.
    This gives Actor/Critics access to project-specific conventions and
    the accumulated "Common Mistakes" list without re-fetching AgentMemory.
  - Reads .dependency-cruiser.json / ArchUnit test count as arch rule signal.
  - Blast-radius pipeline routing: sets a derived field that tells the
    Verifier whether to trigger both frontend AND backend pipelines.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.orchestration.state import ProjectMode, WorkflowPhase
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.nodes._audit import make_audit_entry

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_FILE_PATTERN = re.compile(r"[\w/\-]+\.(?:tsx|ts|java|sql|yaml|yml|json|xml)")


def make_scout_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def scout_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "scout")
        async with node_timer("scout", state=state) as timing:
            task_id     = state["task_id"]
            mode        = state["project_mode"]
            description = state.get("task_description", "")

            log.info("scout.start", task_id=task_id, mode=mode)

            # ── 1. File hints from task description ───────────────────────────
            file_hints = list(set(_FILE_PATTERN.findall(description)))

            # ── 2. CodeGraph: call-graph + dependency sub-graph ───────────────
            call_graph = await deps.code_graph.get_call_graph(file_hints or ["."])
            subgraph   = await deps.code_graph.get_dependency_subgraph(file_hints or ["."])

            # ── 3. Graphify: cross-domain alignment ───────────────────────────
            entity_names = [Path(f).stem for f in file_hints]
            schema_map   = await deps.cross_domain.map_code_to_schema(entity_names)
            arch_align   = await deps.cross_domain.get_arch_alignment(file_hints or ["."])

            # ── 4. Build ContextSkeleton ──────────────────────────────────────
            context_skeleton = {
                "call_graph":   {
                    "entry": call_graph.entry_point,
                    "nodes": call_graph.nodes[:30],
                    "edges": call_graph.edges[:50],
                },
                "dependencies": subgraph,
                "schema_map":   schema_map,
                "arch_align":   arch_align,
            }

            # ── 5. Blast-radius (Brownfield) ──────────────────────────────────
            blast_radius_map: dict[str, Any] | None = None
            if mode == ProjectMode.BROWNFIELD.value and file_hints:
                blast = await deps.code_graph.get_blast_radius(file_hints)
                blast_radius_map = {
                    "entry_files":          blast.entry_files,
                    "affected_files":       blast.affected_files,
                    "dependency_depth":     blast.dependency_depth,
                    "cross_service_impact": blast.cross_service_impact,
                    "schema_impact":        blast.schema_impact,
                    "risk_score":           blast.risk_score,
                }
                log.info(
                    "scout.blast_radius",
                    affected=len(blast.affected_files),
                    risk=blast.risk_score,
                    schema_impact=len(blast.schema_impact),
                )

            # ── 6. Read AGENTS.md (approach 3, 11) ────────────────────────────
            agents_md_context: str | None = None
            agents_md_path = deps.repo_root / "AGENTS.md"
            if agents_md_path.exists():
                raw = agents_md_path.read_text(encoding="utf-8")
                max_chars = deps.config.agents_md_prompt_chars
                agents_md_context = raw[:max_chars]
                if len(raw) > max_chars:
                    agents_md_context += "\n\n[...truncated — see AGENTS.md for full content]"
                log.info("scout.agents_md_loaded", chars=len(agents_md_context))

            timing["skeleton_nodes"] = len(call_graph.nodes)
            log.info(
                "scout.complete",
                skeleton_nodes=len(call_graph.nodes),
                has_blast_radius=blast_radius_map is not None,
                has_agents_md=agents_md_context is not None,
            )

            return {
                "current_phase":    WorkflowPhase.VALUE_NODE.value,
                "context_skeleton": context_skeleton,
                "blast_radius_map": blast_radius_map,
                "agents_md_context": agents_md_context,
                "workflow_audit_trail": [make_audit_entry(
                    "scout",
                    "context_built",
                    {
                        "file_hints":           file_hints,
                        "call_graph_nodes":     len(call_graph.nodes),
                        "blast_risk":           (blast_radius_map or {}).get("risk_score", 0.0),
                        "agents_md_loaded":     agents_md_context is not None,
                    },
                )],
            }

    return scout_node
