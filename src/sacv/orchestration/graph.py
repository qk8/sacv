"""
orchestration/graph.py
======================
Assembles the SACV LangGraph StateGraph.

Refactoring additions (debugging session):
  - IntelligentDebuggerNode added between Verifier and Actor.
    Triggered when diagnostic == AMBIGUOUS.
    Always routes to Actor after debug session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowState
from sacv.orchestration.edges import (
    route_after_value_node,
    route_after_tdd_gate,
    route_after_verifier,
    route_after_speculative_branch,
    route_after_preflight,
    compute_confidence_score,
)
from sacv.interfaces.agent_provider      import AgentProvider
from sacv.interfaces.memory_provider     import MemoryProvider
from sacv.interfaces.code_graph_provider import CodeGraphProvider
from sacv.interfaces.cross_domain_provider import CrossDomainProvider
from sacv.interfaces.git_provider        import GitProvider
from sacv.interfaces.sandbox_provider    import SandboxProvider
from sacv.interfaces.diff_provider       import DiffProvider

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@dataclass
class NodeDeps:
    agent:        AgentProvider
    memory:       MemoryProvider
    code_graph:   CodeGraphProvider
    cross_domain: CrossDomainProvider
    git:          GitProvider
    sandbox:      SandboxProvider
    diff:         DiffProvider
    config:       WorkflowConfig = field(default_factory=WorkflowConfig)


def _inject_confidence(deps: NodeDeps):
    from sacv.nodes.verifier import make_verifier_node
    _inner = make_verifier_node(deps)

    async def verifier_with_confidence(state: WorkflowState) -> dict:
        out    = await _inner(state)
        merged = {**state, **out}
        score  = compute_confidence_score(merged, deps.config)
        return {**out, "confidence_score": score}

    return verifier_with_confidence


def build_graph(
    deps:         NodeDeps,
    checkpointer: object | None = None,
) -> "CompiledStateGraph":
    from sacv.nodes.bootstrap            import make_bootstrap_node
    from sacv.nodes.mode_router          import make_mode_router_node
    from sacv.nodes.scout                import make_scout_node
    from sacv.nodes.value_node           import make_value_node
    from sacv.nodes.tdd_gate             import make_tdd_gate_node
    from sacv.nodes.actor                import make_actor_node
    from sacv.nodes.preflight_node       import make_preflight_node
    from sacv.nodes.critics.security     import make_security_critic_node
    from sacv.nodes.critics.style        import make_style_critic_node
    from sacv.nodes.critics.consistency  import make_consistency_critic_node
    from sacv.nodes.critics.base         import make_aggregate_critics_node
    from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node  # NEW
    from sacv.nodes.replan               import make_replan_node
    from sacv.nodes.speculative_branch   import make_speculative_branch_node
    from sacv.nodes.hitl_escalation      import make_hitl_escalation_node
    from sacv.nodes.memory_consolidation import make_memory_consolidation_node

    cfg     = deps.config
    builder = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("bootstrap",            make_bootstrap_node(deps))
    builder.add_node("mode_router",          make_mode_router_node(deps))
    builder.add_node("scout",                make_scout_node(deps))
    builder.add_node("value_node",           make_value_node(deps))
    builder.add_node("tdd_gate",             make_tdd_gate_node(deps))
    builder.add_node("actor",                make_actor_node(deps))
    builder.add_node("preflight_node",       make_preflight_node(deps))
    builder.add_node("security_critic",      make_security_critic_node(deps))
    builder.add_node("style_critic",         make_style_critic_node(deps))
    builder.add_node("consistency_critic",   make_consistency_critic_node(deps))
    builder.add_node("aggregate_critics",    make_aggregate_critics_node(deps))
    builder.add_node("verifier",             _inject_confidence(deps))
    builder.add_node("intelligent_debugger", make_intelligent_debugger_node(deps))  # NEW
    builder.add_node("replan",               make_replan_node(deps))
    builder.add_node("speculative_branch",   make_speculative_branch_node(deps))
    builder.add_node("hitl_escalation",      make_hitl_escalation_node(deps))
    builder.add_node("memory_consolidation", make_memory_consolidation_node(deps))

    # ── Direct edges ──────────────────────────────────────────────────────
    builder.add_edge(START,                  "bootstrap")
    builder.add_edge("bootstrap",            "mode_router")
    builder.add_edge("mode_router",          "scout")
    builder.add_edge("scout",                "value_node")
    builder.add_edge("actor",                "preflight_node")
    builder.add_edge("security_critic",      "aggregate_critics")
    builder.add_edge("style_critic",         "aggregate_critics")
    builder.add_edge("consistency_critic",   "aggregate_critics")
    builder.add_edge("aggregate_critics",    "verifier")
    # Debugger always routes to actor (with structured observations attached)
    builder.add_edge("intelligent_debugger", "actor")              # NEW
    builder.add_edge("replan",               "value_node")
    builder.add_edge("memory_consolidation", END)
    builder.add_edge("hitl_escalation",      END)

    # ── Conditional edges ─────────────────────────────────────────────────
    builder.add_conditional_edges(
        "value_node",
        route_after_value_node,
        {"tdd_gate": "tdd_gate", "hitl_escalation": "hitl_escalation"},
    )
    builder.add_conditional_edges(
        "tdd_gate",
        route_after_tdd_gate,
        {"actor": "actor", "tdd_gate": "tdd_gate", "hitl_escalation": "hitl_escalation"},
    )
    builder.add_conditional_edges(
        "preflight_node",
        route_after_preflight,
        {
            "actor":              "actor",
            "security_critic":    "security_critic",
            "style_critic":       "style_critic",
            "consistency_critic": "consistency_critic",
        },
    )

    builder.add_conditional_edges(
        "verifier",
        lambda s: route_after_verifier(s, cfg),
        {
            "memory_consolidation": "memory_consolidation",
            "actor":                "actor",
            "intelligent_debugger": "intelligent_debugger",   # NEW
            "speculative_branch":   "speculative_branch",
            "hitl_escalation":      "hitl_escalation",
        },
    )
    builder.add_conditional_edges(
        "speculative_branch",
        lambda s: route_after_speculative_branch(s, cfg),
        {
            "memory_consolidation": "memory_consolidation",
            "replan":               "replan",
            "hitl_escalation":      "hitl_escalation",
        },
    )

    return builder.compile(checkpointer=checkpointer or MemorySaver())
