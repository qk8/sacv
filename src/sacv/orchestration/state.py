"""
orchestration/state.py
======================
Single source of truth for the SACV graph state.

Refactoring additions (approaches 1, 4, 6-11):
  - preflight_result:       LSP + StructuralCheck output (approach 1, 9, 10)
  - test_inventory_paths:   permanent test files committed this session (approach 6-8)
  - replan_count:           how many replan attempts have been made (approach 2, 4)
  - confidence_score:       composite escalation signal (approach 4)
  - task_description:       promoted to formal state field
  - arch_rules_updated:     whether arch rule files were written this session (approach 11)
"""
from __future__ import annotations

from enum import Enum
from enum import StrEnum
from typing import Annotated, Literal, TypedDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class ProjectMode(str, Enum):
    GREENFIELD = "greenfield"
    BROWNFIELD = "brownfield"

class ModuleType(str, Enum):
    BACKEND_DOMAIN   = "backend-domain"
    BACKEND_API      = "backend-api"
    FRONTEND_FEATURE = "frontend-feature"
    FRONTEND_DATA    = "frontend-data"
    INFRASTRUCTURE   = "infrastructure"
    CROSS_CUTTING    = "cross-cutting"

class WorkflowPhase(str, Enum):
    BOOTSTRAP             = "bootstrap"
    MODE_ROUTER           = "mode_router"
    SCOUT                 = "scout"
    VALUE_NODE            = "value_node"
    TDD_GATE              = "tdd_gate"
    ACTOR                 = "actor"
    PREFLIGHT             = "preflight"        # LSP + StructuralCheck
    CRITICS               = "critics"
    VERIFIER              = "verifier"
    INTELLIGENT_DEBUGGER  = "intelligent_debugger"
    REPLAN                = "replan"           # replanning before HITL
    SPECULATIVE_BRANCH    = "speculative_branch"
    HITL_ESCALATION       = "hitl_escalation"
    MEMORY_CONSOLIDATION  = "memory_consolidation"
    COMPLETE              = "complete"

class DiagnosticVerdict(str, Enum):
    FIX_IMPL    = "FIX_IMPL"
    FIX_TEST    = "FIX_TEST"
    AMBIGUOUS   = "AMBIGUOUS"
    STAGNATION  = "STAGNATION"
    PASS        = "PASS"

class StagnationPattern(str, Enum):
    NONE      = "none"
    ITERATION = "iteration"
    SEMANTIC  = "semantic"


# ── State reducers ────────────────────────────────────────────────────────────

class _CriticReset(StrEnum):
    """
    Sentinel: when set as critic_findings, the reducer RESETS the list.

    Uses StrEnum so the sentinel is a plain string under the hood —
    LangGraph's MemorySaver serialises state to msgpack before the
    reducer runs, and msgpack can serialise strings but not arbitrary
    Python class instances.
    """

    RESET = "__CRITIC_RESET__"


CRITIC_RESET = _CriticReset.RESET


def _merge_correction_state(
    existing: CorrectionCycleState | None, new: CorrectionCycleState | None,
) -> CorrectionCycleState:
    """Reducer for correction_state — shallow-merges node updates into existing state."""
    if new is None:
        return existing or {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        }
    # Shallow merge: existing keys overwritten by new keys
    result: dict[str, object] = {}
    if existing:
        result.update(existing)
    result.update(new)
    return result  # type: ignore[return-value]


def _merge_branches(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer for active_branches and exhausted_branches.

    - ``new is None``   → node did not touch this field → return existing (default ``[]``)
    - ``new == []``     → node explicitly cleared → return ``[]``
    - ``new == [...]``  → node appended/overwrote → append to existing
    """
    if new is None:
        return existing or []
    if isinstance(new, list):
        if len(new) == 0:
            return []
        return (existing or []) + new
    return existing or []


def _append_audit(
    existing: list["AuditEntry"] | None,
    new:      list["AuditEntry"] | None,
) -> list["AuditEntry"]:
    """Reducer for workflow_audit_trail — appends new audit entries."""
    if new is None:
        return existing or []
    if isinstance(new, list) and len(new) == 0:
        return existing or []
    return (existing or []) + new  # type: ignore[return-value]


def _merge_lists(existing: list[CriticFinding] | None, new: list[CriticFinding] | _CriticReset | None) -> list[CriticFinding]:
    """
    Reducer for critic fan-in.

    Return value semantics for any node updating ``critic_findings``:
      - Return ``None``  → field unchanged (node did not touch it).
      - Return ``CRITIC_RESET`` → RESET: clears all accumulated findings.
        Use this in Actor/Bootstrap to wipe stale data.
      - Return ``[]`` → no-op: critic found nothing (preserves existing findings).
      - Return ``[...]`` → APPEND: adds findings to the existing list.
        Use this in individual critic nodes.

    RULE:
      - Critics:     return {"critic_findings": [new_finding_1, ...]}
      - Actor/Reset: return {"critic_findings": CRITIC_RESET}
      - Observers:   return {} or omit the key entirely  (None = no change)
    """
    if new is None:
        return existing or []
    # Explicit sentinel check — works for both StrEnum instance and
    # deserialized plain str from msgpack checkpoint serialization
    if isinstance(new, str):
        if new == CRITIC_RESET:
            return []
        # Non-CRITIC_RESET string is unexpected; log and ignore
        import structlog
        structlog.get_logger(__name__).warning(
            "_merge_lists.unexpected_string_value", value=new[:50]
        )
        return existing or []
    if isinstance(new, list):
        if len(new) == 0:
            return existing or []        # empty list = no findings = no change
        return (existing or []) + new
    return existing or []


def _add_costs(existing: float | None, new: float | None) -> float:
    """Reducer for cumulative_cost_dollars — overwrites with the node's
    cumulative total.  Nodes (via add_agent_cost) already return the
    full running total, so the reducer must not add it again."""
    if new is None:
        return existing or 0.0
    return float(new)


def _merge_strings(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer for list[str] fields — appends new strings to existing."""
    if new is None:
        return existing or []
    if isinstance(new, list):
        if len(new) == 0:
            return existing or []
        return (existing or []) + new
    return existing or []


# ── Audit trail ───────────────────────────────────────────────────────────────

class AuditEntry(TypedDict):
    """A single entry in the workflow audit trail.

    Appended by nodes and routing functions to record key decisions.
    """
    timestamp_ms: float                # epoch time in milliseconds
    node:         str                  # node or routing function name
    decision:     str                  # human-readable decision description
    key_values:   dict[str, object]    # most important state values at this point


# ── Sub-state TypedDicts ──────────────────────────────────────────────────────

class CorrectionCycleState(TypedDict):
    attempt_count:      int
    branch_name:        str | None
    last_error_hash:    str | None
    error_history:      list[str]
    stagnation_pattern: str

class StrategyCandidate(TypedDict):
    strategy_id:        str
    description:        str
    affected_files:     list[str]
    token_depth_score:  float
    collision_score:    float
    blast_radius_score: float
    composite_score:    float

class UnifiedDiffPayload(TypedDict):
    file_path:    str
    diff_content: str
    operation:    Literal["modify", "create", "delete"]
    language:     Literal["java", "typescript", "sql", "yaml", "other"]

class DiffProposal(TypedDict):
    strategy_id:    str
    diffs:          list[UnifiedDiffPayload]
    branch_name:    str
    commit_message: str

class CriticFinding(TypedDict):
    critic:          Literal["security", "style", "consistency"]
    severity:        Literal["critical", "warning", "info"]
    file:            str
    line:            int | None
    rule_id:         str
    message:         str
    resolution_hint: str

class LspError(TypedDict):
    file:     str
    line:     int
    code:     str
    message:  str


class ArchViolation(TypedDict):
    rule:         str
    message:      str
    source_file:  str | None
    target_file:  str | None


class CrossStackError(TypedDict):
    file:    str
    line:    int | None
    code:    str
    message: str
    source:  str


class BlastError(TypedDict):
    rule:    str
    message: str


class PreflightResult(TypedDict):
    """
    Output of the Pre-Critic Preflight node.

    lsp_errors:         type errors + missing imports (approach 1)
    arch_violations:    layer boundary violations (approaches 9-10)
    cross_stack_errors: Java DTO changed but TypeScript types stale (approach 3A)
    blast_errors:       blast-radius file count exceeded (approach 3B)
    repair_suggestions: structured repair guidance for the Actor (CONCERN-2)
    passed:             True only when all checks produce zero findings
    duration_ms:        total elapsed time (should be <5 000ms)
    """
    passed:             bool
    lsp_errors:         list[LspError]
    arch_violations:    list[ArchViolation]
    cross_stack_errors: list[CrossStackError]
    blast_errors:       list[BlastError]
    repair_suggestions: list[dict[str, str]]
    duration_ms:        int

class TestFailure(TypedDict):
    message:  str
    file:     str | None


class OtelTrace(TypedDict):
    trace_id: str
    spans:    list[dict[str, object]]


class ActuatorSnapshot(TypedDict):
    """Spring Actuator bean/env snapshot."""
    pass


class VerifierVerdict(TypedDict):
    test_result:          Literal["PASS", "FAIL"]
    diagnostic:           str
    # Two-Phase Guardrail (approach 8)
    phase1_passed:        bool         # existing test suite
    phase2_passed:        bool         # newly written tests
    test_failures:        list[TestFailure]
    performance_delta:    dict[str, object] | None
    visual_diff_result:   dict[str, object] | None
    docker_exit_code:     int
    # Optional debug artefacts — always present in the dict, None when not collected
    playwright_trace_path: str | None
    otel_trace:             OtelTrace | None
    actuator_snapshot:      ActuatorSnapshot | None
    # HIGH-003: True when verifier blocked on critical critic findings
    # without running Docker. Actor should use critic feedback directly
    # rather than attempting test-driven debugging.
    blocked_by_critic:      bool

class ResolutionHint(TypedDict):
    priority:  int
    category:  Literal["architectural", "test_oracle", "security", "blast_radius"]
    hint:      str
    automated: bool

class FailureSummary(TypedDict):
    total_attempts:       int
    branches_exhausted:   list[str]
    stagnation_pattern:   str
    last_verifier_output: dict[str, object] | None
    critic_findings:      list[CriticFinding]


class GitState(TypedDict):
    active_branch:       str | None
    stash_ref:           str | None
    last_green_commit:   str | None
    stashed_branches:    list[str]
    uncommitted_files:   list[str]
    git_reset_failed:    str | None
    stash_pop_command:   str | None
    stash_note:          str | None


class ResumeInstructions(TypedDict):
    command:      str
    state_file:   str
    note:         str


class EscalationPayload(TypedDict):
    escalation_id:       str
    timestamp:           str
    workflow_version:    str
    task_id:             str
    task_description:    str
    failure_summary:     FailureSummary
    git_state:           GitState
    resolution_hints:    list[ResolutionHint]
    resume_instructions: ResumeInstructions
    audit_trail:         list[AuditEntry]   # HIGH-04: decision history for human reviewer
    debug_observations:  DebugObservations | None   # AUD-003: last debug session for immediate inspection
    last_preflight:      PreflightResult | None     # AUD-003: last preflight result for preflight loop diagnosis
    last_test_failures:  list[TestFailure]          # AUD-003: raw test failures for immediate inspection

class LessonLearned(TypedDict):
    task_id:              str
    pattern_discovered:   str
    negative_constraints: list[str]
    blast_radius_learned: dict[str, object]
    correction_type:      str
    session_duration_ms:  int


class ContextSkeleton(TypedDict):
    call_graph:   dict[str, object]
    dependencies: list[str]
    schema_map:   dict[str, object]
    arch_align:   dict[str, object]


class BlastRadiusMap(TypedDict):
    entry_files:          list[str]
    affected_files:       list[str]
    dependency_depth:     int
    cross_service_impact: list[str]
    schema_impact:        list[str]
    risk_score:           float


# ── Debug observation schema (must precede WorkflowState) ─────────────────────

class VariableInfo(TypedDict):
    value: str
    type:  str | None


class BreakpointHit(TypedDict):
    file:        str
    line:        int
    variables:   dict[str, VariableInfo]
    call_stack:  list[str]
    thread_id:   str | None
    extra_evals: dict[str, str]


class PrunedFrame(TypedDict):
    file:     str
    line:     int
    method:   str
    message:  str


class DebugObservations(TypedDict):
    """
    Structured output from the IntelligentDebuggerNode.
    Passed directly to the Actor so it can make a precise fix.
    """
    error_type:       str
    root_cause:       str
    breakpoint_hits:  list[BreakpointHit]
    actuator_beans:   dict[str, object] | None
    actuator_env:     dict[str, object] | None
    minimal_payload:  dict[str, object] | None
    playwright_trace_path: str | None
    otel_trace:       OtelTrace | None
    pruned_stack:     list[PrunedFrame]


# ── Root graph state ──────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────
    session_id:       str
    task_id:          str
    task_description: str        # promoted to formal field (was ad-hoc)
    session_start_ms: float | None  # epoch ms; set by bootstrap for duration tracking

    # ── Immutable configuration ────────────────────────────────────────────
    project_mode:  str
    module_type:   str

    # ── Preflight check profile ────────────────────────────────────────────
    check_profile: str  # "standard" | "full" — controls which preflight checks run

    # ── Phase tracking ────────────────────────────────────────────────────
    current_phase: str

    # ── Scout outputs ─────────────────────────────────────────────────────
    context_skeleton:  ContextSkeleton | None
    blast_radius_map:  BlastRadiusMap | None
    agents_md_context: str | None  # AGENTS.md content (approach 3, 11)

    # ── Value Node outputs ────────────────────────────────────────────────
    strategy_candidates: list[StrategyCandidate]
    selected_strategy:   StrategyCandidate | None
    pruned_strategies:   list[StrategyCandidate]

    # ── TDD Gate output ───────────────────────────────────────────────────
    red_phase_evidence_path: str | None
    test_inventory_paths:    list[str]   # permanent test files (approaches 6-8)
    tdd_gate_attempts:       int         # escape hatch for TDD gate infinite loop
    skip_tdd_gate:           bool        # bypass TDD gate for test scenarios

    # ── Actor output ──────────────────────────────────────────────────────
    diff_proposal:      DiffProposal | None
    empty_diff_retries: int         # no-diff Actor loops (separate from attempt_count)

    # ── Preflight output (NEW — approaches 1, 9, 10) ──────────────────────
    preflight_result: PreflightResult | None

    # ── Critic outputs ────────────────────────────────────────────────────
    critic_findings: Annotated[list[CriticFinding], _merge_lists]
    critic_errors: Annotated[list[str], _merge_strings]   # names of critics that raised exceptions

    # ── Verifier output ───────────────────────────────────────────────────
    verifier_verdict: VerifierVerdict | None

    # ── Correction loop ───────────────────────────────────────────────────
    correction_state:  Annotated[CorrectionCycleState, _merge_correction_state]
    confidence_score:  float    # computed after each verifier run (approach 4)

    # ── Replan (NEW — approach 2, 4) ──────────────────────────────────────
    replan_count: int

    # ── Speculative branching ─────────────────────────────────────────────
    active_branches:        Annotated[list[str], _merge_branches]
    exhausted_branches:     Annotated[list[str], _merge_branches]
    speculative_stash_ref:  str | None  # stash ref for restoring pre-speculation state (BUG-013)

    # ── HITL ──────────────────────────────────────────────────────────────
    escalation_payload: EscalationPayload | None

    # ── Memory ────────────────────────────────────────────────────────────
    procedural_constraints: list[str]
    lesson_learned:         LessonLearned | None
    arch_rules_updated:     bool    # whether arch rule files were written (approach 11)

    # ── Debugger output (IntelligentDebuggerNode → ActorNode) ─────────────
    debug_observations:     DebugObservations | None

    # ── Token budget tracking (BUG-008) ──────────────────────────────────
    cumulative_cost_dollars: Annotated[float, _add_costs]

    # ── Structured audit trail (HIGH-04) ─────────────────────────────────
    workflow_audit_trail: Annotated[list[AuditEntry], _append_audit]
