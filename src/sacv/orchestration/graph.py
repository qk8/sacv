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

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer

log = structlog.get_logger(__name__)

from langgraph.graph import StateGraph, START, END

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowState
from sacv.orchestration.edges import (
    route_after_value_node,
    route_after_tdd_gate,
    route_after_verifier,
    route_after_speculative_branch,
    route_after_preflight,
    route_after_actor,
    route_after_replan,
    compute_confidence_score,
)
from sacv.orchestration.deps import NodeDeps

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver


from sacv.orchestration.verifier_utils import (
    run_verifier_with_confidence as _run_verifier_with_confidence,
)


def _make_all_critics_node(deps: "NodeDeps") -> Any:
    """
    Single node that runs all 3 critics concurrently and returns merged findings.

    Replaces the broken Send()-based fan-out pattern where each critic had
    add_edge -> aggregate_critics -> verifier, causing verifier to run 3x.
    Now all critics run inside one node via asyncio.gather, fan-in here,
    and a single edge -> verifier ensures it runs exactly once.
    """
    from sacv.nodes.critics.security    import make_security_critic_node
    from sacv.nodes.critics.style       import make_style_critic_node
    from sacv.nodes.critics.consistency import make_consistency_critic_node
    from sacv.orchestration.state import WorkflowPhase

    async def all_critics_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "all_critics")
        async with node_timer("all_critics", state=state) as timing:
            sec_node = make_security_critic_node(deps)
            sty_node = make_style_critic_node(deps)
            con_node = make_consistency_critic_node(deps)

            results = await asyncio.gather(
                sec_node(state),
                sty_node(state),
                con_node(state),
                return_exceptions=True,
            )

            critic_errors: list[str] = []

            def _safe_out(result, name):
                if isinstance(result, Exception):
                    log.error(
                        "critic_node_exception",
                        critic=name,
                        error=str(result),
                        exc_info=True,
                    )
                    critic_errors.append(name)
                    return {"critic_findings": [], "cumulative_cost_dollars": state.get("cumulative_cost_dollars", 0.0)}
                return result

            sec_out = _safe_out(results[0], "security")
            sty_out = _safe_out(results[1], "style")
            con_out = _safe_out(results[2], "consistency")

            all_findings = (
                sec_out.get("critic_findings", [])
                + sty_out.get("critic_findings", [])
                + con_out.get("critic_findings", [])
            )
            # Each critic receives the same state snapshot; each returns
            # baseline + its own cost. Sum all three outputs and subtract 3x
            # the baseline to isolate the incremental cost. Add baseline back
            # so the cumulative total (including prior nodes' costs) is preserved.
            baseline = state.get("cumulative_cost_dollars", 0.0)
            final_cost = (
                baseline
                + sec_out.get("cumulative_cost_dollars", baseline)
                + sty_out.get("cumulative_cost_dollars", baseline)
                + con_out.get("cumulative_cost_dollars", baseline)
                - 3.0 * baseline
            )
            sec_n = len(sec_out.get("critic_findings", []))
            sty_n = len(sty_out.get("critic_findings", []))
            con_n = len(con_out.get("critic_findings", []))

            timing["findings"] = len(all_findings)
            timing["sec_n"] = sec_n
            timing["sty_n"] = sty_n
            timing["con_n"] = con_n
            timing["critic_errors"] = critic_errors

            audit_entries: list[dict[str, object]] | None = None
            if critic_errors:
                audit_entries = [{
                    "timestamp_ms": time.time() * 1000,
                    "node": "all_critics",
                    "decision": f"critic_exceptions: {', '.join(critic_errors)}",
                    "key_values": {"failed_critics": critic_errors, "findings_count": len(all_findings)},
                }]

            return {
                "current_phase":           WorkflowPhase.CRITICS.value,
                "critic_findings":         all_findings,
                "critic_errors":           critic_errors,
                "cumulative_cost_dollars": final_cost,
                "workflow_audit_trail":    audit_entries,
            }

    return all_critics_node


def _inject_confidence(deps: "NodeDeps") -> Any:
    async def verifier_with_confidence(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "verifier_with_confidence")
        async with node_timer("verifier_with_confidence", state=state) as timing:
            result = await _run_verifier_with_confidence(state, deps)
            timing["confidence"] = result.get("verifier_confidence")
            return {k: v for k, v in result.items()}

    return verifier_with_confidence


def build_branch_subgraph(deps: "NodeDeps") -> Any:
    """
    Build the mini-workflow used by speculative branches.

    Each speculative branch runs: actor -> preflight -> critics -> verifier.

    Returns an uncompiled StateGraph -- each branch compiles its own copy
    with its own checkpointer (WORKER-001).

    This function is shared between build_graph() (for the main graph's
    actor->preflight->critics->verifier path) and speculative_branch
    (_evaluate_branch), ensuring both stay in sync when nodes change.
    """
    from sacv.nodes.actor                import make_actor_node
    from sacv.nodes.preflight_node       import make_preflight_node
    from sacv.nodes.critics.security     import make_security_critic_node
    from sacv.nodes.critics.style        import make_style_critic_node
    from sacv.nodes.critics.consistency  import make_consistency_critic_node
    from sacv.orchestration.verifier_utils import (
        run_verifier_with_confidence as _run_verifier_with_confidence,
    )

    builder = StateGraph(WorkflowState)

    builder.add_node("actor",            make_actor_node(deps))
    builder.add_node("preflight_node",   make_preflight_node(deps))
    builder.add_node("all_critics",      _make_all_critics_node(deps))
    builder.add_node("verifier",         _inject_confidence(deps))

    # Direct edges: actor -> preflight -> critics -> verifier
    builder.add_edge("actor", "preflight_node")
    builder.add_edge("preflight_node", "all_critics")
    builder.add_edge("all_critics", "verifier")

    return builder


def build_graph(
    deps:         "NodeDeps",
    checkpointer: "BaseCheckpointSaver[Any] | None" = None,
) -> Any:
    from sacv.nodes.bootstrap            import make_bootstrap_node
    from sacv.nodes.mode_router          import make_mode_router_node
    from sacv.nodes.scout                import make_scout_node
    from sacv.nodes.value_node           import make_value_node
    from sacv.nodes.tdd_gate             import make_tdd_gate_node
    from sacv.nodes.actor                import make_actor_node
    from sacv.nodes.preflight_node       import make_preflight_node
    from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node  # NEW
    from sacv.nodes.replan               import make_replan_node
    from sacv.nodes.speculative_branch   import make_speculative_branch_node
    from sacv.nodes.hitl_escalation      import make_hitl_escalation_node
    from sacv.nodes.memory_consolidation import make_memory_consolidation_node

    cfg     = deps.config
    builder = StateGraph(WorkflowState)

    # -- Register nodes --
    builder.add_node("bootstrap",            make_bootstrap_node(deps))
    builder.add_node("mode_router",          make_mode_router_node(deps))
    builder.add_node("scout",                make_scout_node(deps))
    builder.add_node("value_node",           make_value_node(deps))
    builder.add_node("tdd_gate",             make_tdd_gate_node(deps))
    builder.add_node("actor",                make_actor_node(deps))
    builder.add_node("preflight_node",       make_preflight_node(deps))
    builder.add_node("all_critics",          _make_all_critics_node(deps))
    builder.add_node("verifier",             _inject_confidence(deps))
    builder.add_node("intelligent_debugger", make_intelligent_debugger_node(deps))  # NEW
    builder.add_node("replan",               make_replan_node(deps))
    builder.add_node("speculative_branch",   make_speculative_branch_node(deps))
    builder.add_node("hitl_escalation",      make_hitl_escalation_node(deps))
    builder.add_node("memory_consolidation", make_memory_consolidation_node(deps))

    # -- Direct edges --
    builder.add_edge(START,                  "bootstrap")
    builder.add_edge("bootstrap",            "mode_router")
    builder.add_edge("mode_router",          "scout")
    builder.add_edge("scout",                "value_node")
    builder.add_conditional_edges(
        "actor",
        lambda s: route_after_actor(s, cfg),
        {
            "actor":           "actor",           # self-loop for empty-diff retry
            "preflight_node":  "preflight_node",
            "hitl_escalation": "hitl_escalation",
        },
    )
     # Debugger always routes to actor (with structured observations attached)
    builder.add_edge("intelligent_debugger", "actor")              # NEW
    builder.add_edge("memory_consolidation", END)
    builder.add_edge("hitl_escalation",      "memory_consolidation")  # HIGH-002: persist failure lessons after HITL resume
    builder.add_edge("all_critics",          "verifier")

    # -- Conditional edges --
    builder.add_conditional_edges(
        "value_node",
        route_after_value_node,
        {"tdd_gate": "tdd_gate", "hitl_escalation": "hitl_escalation"},
    )
    builder.add_conditional_edges(
        "tdd_gate",
        lambda s: route_after_tdd_gate(s, cfg),
        {"actor": "actor", "tdd_gate": "tdd_gate", "hitl_escalation": "hitl_escalation"},
    )
    builder.add_conditional_edges(
        "preflight_node",
        route_after_preflight,
        {
            "actor":           "actor",
            "all_critics":     "all_critics",
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
    builder.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "tdd_gate":       "tdd_gate",
            "hitl_escalation": "hitl_escalation",
        },
    )

    if checkpointer is None:
        raise ValueError(
            "build_graph() requires an explicit checkpointer (got None). "
            "For production use: AsyncSqliteSaver.from_conn_string('.workflow/sacv.db') "
            "For testing use: MemorySaver() -- note that MemorySaver does not persist "
            "across process restarts and cannot resume after HITL interrupts."
        )
    return builder.compile(checkpointer=checkpointer)
