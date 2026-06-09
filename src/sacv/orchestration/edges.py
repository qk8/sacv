"""
orchestration/edges.py
======================
All conditional edge functions. Pure functions: (WorkflowState) -> str.
No I/O, no side effects. Fully unit-testable.

Refactoring additions (debugging session):
  - route_after_verifier: AMBIGUOUS diagnostic now routes to intelligent_debugger
    instead of sending Actor back for a blind retry.
"""
from __future__ import annotations

import structlog

from sacv.orchestration.state import WorkflowState
from sacv.orchestration.config import WorkflowConfig

log = structlog.get_logger(__name__)

# ── Preflight routing ─────────────────────────────────────────────────────────

def route_after_preflight(state: WorkflowState) -> str:
    result = state.get("preflight_result") or {}
    if (
        result.get("lsp_errors")
        or result.get("arch_violations")
        or result.get("cross_stack_errors")
        or result.get("blast_errors")
    ):
        return "actor"
    return "all_critics"


# ── Confidence score ──────────────────────────────────────────────────────────

def compute_confidence_score(
    state:  WorkflowState,
    config: WorkflowConfig,
) -> float:
    """Pure function. No I/O.

    Confidence decreases as cumulative cost approaches warning_dollar.
    The penalty scales linearly between warning_dollar and critical_dollar,
    reaching 1.0 (full penalty) at critical_dollar.
    """
    cfg        = config
    correction = state["correction_state"]
    attempt    = correction["attempt_count"]
    cost       = state.get("cumulative_cost_dollars", 0.0)

    attempt_penalty    = min(1.0, attempt / max(cfg.max_self_correction_cycles, 1))
    stagnation_penalty = 0.40 if correction.get("stagnation_pattern", "none") != "none" else 0.0
    blast              = state.get("blast_radius_map") or {}
    blast_penalty      = float(blast.get("risk_score", 0.0)) * 0.30
    findings           = state.get("critic_findings") or []
    critic_penalty     = min(0.30, sum(
        0.10 for f in findings if f.get("severity") == "critical"
    ))

    # Cost penalty: linear ramp from 0 at warning_dollar to 1.0 at critical_dollar
    warning = cfg.token_budget.warning_dollar
    critical = cfg.token_budget.critical_dollar
    if cost >= critical:
        cost_penalty = 1.0
    elif cost >= warning:
        cost_penalty = (cost - warning) / (critical - warning)
    else:
        cost_penalty = 0.0

    result = max(0.0, 1.0 - attempt_penalty - stagnation_penalty - blast_penalty - critic_penalty - cost_penalty)
    return float(result)


# ── Actor routing ─────────────────────────────────────────────────────────────

def route_after_actor(
    state:  WorkflowState,
    config: WorkflowConfig,
) -> str:
    """
    Route after Actor node.

    If stagnation was detected during actor execution, skip preflight/critics
    and go directly to HITL escalation to avoid wasting LLM calls and Docker
    cycles on a known-stagnated path.

    If actor produced no diff (overwrite rejected or apply failed), loop back
    to actor directly — skip the full pipeline to avoid wasting Docker/critic
    cycles and prevent silent completion with no code applied.

    Safety valve: prevent infinite empty-diff loops (MED-004).
    """
    cfg = config
    correction = state["correction_state"]
    if correction.get("stagnation_pattern", "none") != "none":
        return "hitl_escalation"
    # If actor produced no diff, retry without wasting Docker/critic cycles
    if state.get("diff_proposal") is None:
        if state.get("empty_diff_retries", 0) >= cfg.max_empty_diff_retries:
            return "hitl_escalation"
        return "actor"
    return "preflight_node"


# ── Main routing functions ────────────────────────────────────────────────────

def route_after_value_node(state: WorkflowState) -> str:
    return "hitl_escalation" if not state.get("strategy_candidates") else "tdd_gate"


def route_after_tdd_gate(
        state:  WorkflowState,
        config: WorkflowConfig,
    ) -> str:
        cfg = config
        if state.get("red_phase_evidence_path"):
            return "actor"
        if state.get("tdd_gate_attempts", 0) >= cfg.max_tdd_gate_attempts:
            return "hitl_escalation"
        return "tdd_gate"


def route_after_verifier(
    state:  WorkflowState,
    config: WorkflowConfig,
) -> str:
    """
    Core circuit-breaker routing.

    Priority order:
    1. PASS                   → memory_consolidation
    2. Low confidence         → hitl_escalation  (approach 4 — early exit)
    3. CRITIC block           → actor  (M-02: targeted critic-guided fix)
    4. MAX attempts           → hitl_escalation
    5. diagnostic == AMBIGUOUS → intelligent_debugger  (NEW — debugging session)
    6. attempt >= 2           → speculative_branch
    7. attempt < 2            → actor  (retry with critic feedback)
    """
    cfg     = config
    verdict = state.get("verifier_verdict")
    if verdict is None:
        log.error(
            "route_after_verifier.missing_verdict",
            task_id=state.get("task_id"),
            phase=state.get("current_phase"),
        )
        return "hitl_escalation"

    if verdict["test_result"] == "PASS":
        return "memory_consolidation"

    # Token budget circuit-breaker (BUG-008)
    cost = state.get("cumulative_cost_dollars", 0.0)
    if cost >= cfg.token_budget.critical_dollar:
        log.error("route_after_verifier.budget_exceeded",
                  cost=cost, threshold=cfg.token_budget.critical_dollar)
        return "hitl_escalation"
    if cost >= cfg.token_budget.warning_dollar:
        log.warning("route_after_verifier.budget_warning", cost=cost)

    # Use the value already computed and stored by _inject_confidence
    confidence = state.get("confidence_score", 1.0)
    if confidence < cfg.confidence_escalation_threshold:
        return "hitl_escalation"

    # M-02: When blocked by critical critic findings, route to actor
    # for a targeted critic-guided fix (bypass speculative branch)
    if verdict.get("blocked_by_critic"):
        log.info("route_after_verifier.critic_blocked",
                 task_id=state.get("task_id"))
        return "actor"

    attempt    = state["correction_state"]["attempt_count"]
    diagnostic = verdict.get("diagnostic", "UNKNOWN")

    if attempt >= cfg.max_self_correction_cycles:
        return "hitl_escalation"

    # Debug on first AMBIGUOUS encounter; speculate on repeat AMBIGUOUS
    # (prevents AMBIGUOUS from starving speculative_branch — ARCH-004)
    if diagnostic == "AMBIGUOUS" and attempt <= 1:
        return "intelligent_debugger"

    if attempt >= 2:
        return "speculative_branch"

    return "actor"


def route_after_speculative_branch(
    state:  WorkflowState,
    config: WorkflowConfig,
) -> str:
    cfg     = config
    verdict = state.get("verifier_verdict")

    if verdict and verdict["test_result"] == "PASS":
        return "memory_consolidation"

    replan_count = state.get("replan_count", 0)
    if replan_count < cfg.max_replan_attempts:
        return "replan"
    return "hitl_escalation"


def route_after_replan(state: WorkflowState) -> str:
    """After replan, go straight to TDD gate with the new candidates."""
    if not state.get("strategy_candidates"):
        return "hitl_escalation"
    return "tdd_gate"
