"""
orchestration/edges.py
======================
All conditional edge functions. Pure functions: (WorkflowState) -> str | list[Send].
No I/O, no side effects. Fully unit-testable.

Refactoring additions (debugging session):
  - route_after_verifier: AMBIGUOUS diagnostic now routes to intelligent_debugger
    instead of sending Actor back for a blind retry.
"""
from __future__ import annotations

from langgraph.types import Send

from sacv.orchestration.state import WorkflowState
from sacv.orchestration.config import WorkflowConfig

_SENTINEL = object()


def _cfg(config: object) -> "WorkflowConfig":
    if config is _SENTINEL:
        from sacv.orchestration.config import WorkflowConfig as _WC
        return _WC()
    return config  # type: ignore[return-value]


# ── Preflight routing ─────────────────────────────────────────────────────────

def route_after_preflight(state: WorkflowState) -> str | list[Send]:
    result = state.get("preflight_result") or {}
    if (
        result.get("lsp_errors")
        or result.get("arch_violations")
        or result.get("cross_stack_errors")
    ):
        return "actor"
    return [
        Send("security_critic",    state),
        Send("style_critic",       state),
        Send("consistency_critic", state),
    ]


# ── Confidence score ──────────────────────────────────────────────────────────

def compute_confidence_score(
    state:  WorkflowState,
    config: object = _SENTINEL,
) -> float:
    """Pure function. No I/O."""
    cfg        = _cfg(config)
    correction = state["correction_state"]
    attempt    = correction["attempt_count"]

    attempt_penalty    = min(1.0, attempt / max(cfg.max_self_correction_cycles, 1))
    stagnation_penalty = 0.40 if correction.get("stagnation_pattern", "none") != "none" else 0.0
    blast              = state.get("blast_radius_map") or {}
    blast_penalty      = float(blast.get("risk_score", 0.0)) * 0.30
    findings           = state.get("critic_findings") or []
    critic_penalty     = min(0.30, sum(
        0.10 for f in findings if f.get("severity") == "critical"
    ))
    return max(0.0, 1.0 - attempt_penalty - stagnation_penalty - blast_penalty - critic_penalty)


# ── Main routing functions ────────────────────────────────────────────────────

def route_after_value_node(state: WorkflowState) -> str:
    return "hitl_escalation" if not state.get("strategy_candidates") else "tdd_gate"


_TDD_GATE_MAX_ATTEMPTS = 3


def route_after_tdd_gate(state: WorkflowState) -> str:
    if state.get("red_phase_evidence_path"):
        return "actor"
    if state.get("tdd_gate_attempts", 0) >= _TDD_GATE_MAX_ATTEMPTS:
        return "hitl_escalation"
    return "tdd_gate"


def route_after_verifier(
    state:  WorkflowState,
    config: object = _SENTINEL,
) -> str:
    """
    Core circuit-breaker routing.

    Priority order:
    1. PASS                   → memory_consolidation
    2. Low confidence         → hitl_escalation  (approach 4 — early exit)
    3. MAX attempts           → hitl_escalation
    4. diagnostic == AMBIGUOUS → intelligent_debugger  (NEW — debugging session)
    5. attempt >= 2           → speculative_branch
    6. attempt < 2            → actor  (retry with critic feedback)
    """
    cfg     = _cfg(config)
    verdict = state.get("verifier_verdict")
    if verdict is None:
        raise ValueError("route_after_verifier: verifier_verdict must not be None")

    if verdict["test_result"] == "PASS":
        return "memory_consolidation"

    confidence = compute_confidence_score(state, cfg)
    if confidence < cfg.confidence_escalation_threshold:
        return "hitl_escalation"

    attempt    = state["correction_state"]["attempt_count"]
    diagnostic = verdict.get("diagnostic", "UNKNOWN")

    if attempt >= cfg.max_self_correction_cycles:
        return "hitl_escalation"

    # AMBIGUOUS: don't blindly retry — debug first to find the real cause
    if diagnostic == "AMBIGUOUS":
        return "intelligent_debugger"

    if attempt >= 2:
        return "speculative_branch"
    return "actor"


def route_after_speculative_branch(
    state:  WorkflowState,
    config: object = _SENTINEL,
) -> str:
    cfg     = _cfg(config)
    verdict = state.get("verifier_verdict")

    if verdict and verdict["test_result"] == "PASS":
        return "memory_consolidation"

    replan_count = state.get("replan_count", 0)
    if replan_count < cfg.max_replan_attempts:
        return "replan"
    return "hitl_escalation"
