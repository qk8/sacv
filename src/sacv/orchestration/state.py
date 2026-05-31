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
    PREFLIGHT             = "preflight"        # NEW: LSP + StructuralCheck
    CRITICS               = "critics"
    VERIFIER              = "verifier"
    REPLAN                = "replan"           # NEW: replanning before HITL
    SPECULATIVE_BRANCH    = "speculative_branch"
    HITL_ESCALATION       = "hitl_escalation"
    MEMORY_CONSOLIDATION  = "memory_consolidation"
    COMPLETE              = "complete"

class DiagnosticVerdict(str, Enum):
    FIX_IMPL  = "FIX_IMPL"
    FIX_TEST  = "FIX_TEST"
    AMBIGUOUS = "AMBIGUOUS"
    PASS      = "PASS"

class StagnationPattern(str, Enum):
    NONE      = "none"
    ITERATION = "iteration"
    SEMANTIC  = "semantic"


# ── State reducers ────────────────────────────────────────────────────────────

def _merge_lists(existing: list, new: list) -> list:
    """Reducer: append new items — used by critic fan-in."""
    return (existing or []) + (new or [])


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

class PreflightResult(TypedDict):
    """
    Output of the Pre-Critic Preflight node.

    lsp_errors:         type errors + missing imports (approach 1)
    arch_violations:    layer boundary violations (approaches 9-10)
    cross_stack_errors: Java DTO changed but TypeScript types stale (approach 3A)
    passed:             True only when all checks produce zero findings
    duration_ms:        total elapsed time (should be <5 000ms)
    """
    passed:             bool
    lsp_errors:         list[dict]   # {file, line, message, code}
    arch_violations:    list[dict]   # {rule, source_file, target_file, message}
    cross_stack_errors: list[dict]   # {file, line, code, message, source}
    duration_ms:        int

class VerifierVerdict(TypedDict):
    test_result:          Literal["PASS", "FAIL"]
    diagnostic:           str
    # Two-Phase Guardrail (approach 8)
    phase1_passed:        bool         # existing test suite
    phase2_passed:        bool         # newly written tests
    test_failures:        list[dict]
    performance_delta:    dict | None
    visual_diff_result:   dict | None
    critic_findings:      list[CriticFinding]
    docker_exit_code:     int
    # Optional debug artefacts — always present in the dict, None when not collected
    playwright_trace_path: str | None
    otel_trace:            dict | None
    actuator_snapshot:     dict | None

class ResolutionHint(TypedDict):
    priority:  int
    category:  Literal["architectural", "test_oracle", "security", "blast_radius"]
    hint:      str
    automated: bool

class EscalationPayload(TypedDict):
    escalation_id:       str
    timestamp:           str
    workflow_version:    str
    task_id:             str
    task_description:    str
    failure_summary:     dict
    git_state:           dict
    resolution_hints:    list[ResolutionHint]
    resume_instructions: dict

class LessonLearned(TypedDict):
    task_id:              str
    pattern_discovered:   str
    negative_constraints: list[str]
    blast_radius_learned: dict
    correction_type:      str
    session_duration_ms:  int


# ── Debug observation schema (must precede WorkflowState) ─────────────────────

class BreakpointHit(TypedDict):
    file:      str
    line:      int
    variables: dict        # {name: {"value": ..., "type": ...}}
    call_stack: list[str]  # ["ClassName.method(File.java:42)", ...]
    thread_id:  str | None


class DebugObservations(TypedDict):
    """
    Structured output from the IntelligentDebuggerNode.
    Passed directly to the Actor so it can make a precise fix.
    """
    error_type:       str               # e.g. "NULL_REFERENCE", "ASYNC_RACE_CONDITION"
    root_cause:       str               # human-readable hypothesis
    breakpoint_hits:  list[BreakpointHit]
    # Spring Actuator snapshot (Java only, on BeanCreationException)
    actuator_beans:   dict | None
    actuator_env:     dict | None
    # Delta debug result (validation failures)
    minimal_payload:  dict | None       # smallest input that reproduces the failure
    # Playwright trace (frontend only)
    playwright_trace_path: str | None
    # OTel trace (cross-stack)
    otel_trace:       dict | None       # {trace_id, spans: [...]}
    # Pruned stack trace (always present)
    pruned_stack:     list[dict]        # [{file, line, method, message}]


# ── Root graph state ──────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────
    session_id:       str
    task_id:          str
    task_description: str        # promoted to formal field (was ad-hoc)

    # ── Immutable configuration ────────────────────────────────────────────
    project_mode:  str
    module_type:   str

    # ── Phase tracking ────────────────────────────────────────────────────
    current_phase: str

    # ── Scout outputs ─────────────────────────────────────────────────────
    context_skeleton:  dict | None
    blast_radius_map:  dict | None
    agents_md_context: str | None  # AGENTS.md content (approach 3, 11)

    # ── Value Node outputs ────────────────────────────────────────────────
    strategy_candidates: list[StrategyCandidate]
    selected_strategy:   StrategyCandidate | None
    pruned_strategies:   list[StrategyCandidate]

    # ── TDD Gate output ───────────────────────────────────────────────────
    red_phase_evidence_path: str | None
    test_inventory_paths:    list[str]   # permanent test files (approaches 6-8)
    tdd_gate_attempts:       int         # escape hatch for TDD gate infinite loop

    # ── Actor output ──────────────────────────────────────────────────────
    diff_proposal: DiffProposal | None

    # ── Preflight output (NEW — approaches 1, 9, 10) ──────────────────────
    preflight_result: PreflightResult | None

    # ── Critic outputs ────────────────────────────────────────────────────
    critic_findings: Annotated[list[CriticFinding], _merge_lists]

    # ── Verifier output ───────────────────────────────────────────────────
    verifier_verdict: VerifierVerdict | None

    # ── Correction loop ───────────────────────────────────────────────────
    correction_state:  CorrectionCycleState
    confidence_score:  float    # computed after each verifier run (approach 4)

    # ── Replan (NEW — approach 2, 4) ──────────────────────────────────────
    replan_count: int

    # ── Speculative branching ─────────────────────────────────────────────
    active_branches:        list[str]
    exhausted_branches:     list[str]
    speculative_stash_ref:  str | None  # stash ref for restoring pre-speculation state (BUG-013)

    # ── HITL ──────────────────────────────────────────────────────────────
    escalation_payload: EscalationPayload | None

    # ── Memory ────────────────────────────────────────────────────────────
    procedural_constraints: list[str]
    lesson_learned:         LessonLearned | None
    arch_rules_updated:     bool    # whether arch rule files were written (approach 11)

    # ── Debugger output (IntelligentDebuggerNode → ActorNode) ─────────────
    debug_observations:     DebugObservations | None
