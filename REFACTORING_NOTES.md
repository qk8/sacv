# SACV Workflow — Refactoring Notes

## Overview

This refactoring applies approaches 1–11 from the agentic coding workflow
literature review, organized into five themes.

---

## Theme 1: Pre-Critic Preflight Layer (Approaches 1, 9, 10)

### What changed
- **New node: `PrefLightNode`** (`nodes/preflight_node.py`)
  - Runs after Actor, before the LLM Critic fan-out
  - Check 1 — LSP/Compile: `tsc --noEmit` (TS) or `mvn compile` (Java)
  - Check 2 — StructuralCheck: `dependency-cruiser` (TS) or `ArchUnit` (Java)
  - Token cost: 0. Runtime: < 5 000 ms.

- **Graph rewiring** (`orchestration/graph.py`)
  - Old: `actor → [Send: security, style, consistency]`
  - New: `actor → preflight_node → (violations → actor) | (clean → [Send: critics])`

- **`route_after_preflight`** — new pure edge function in `edges.py`
  Returns `"actor"` on violations, `list[Send]` to all three critics when clean.

- **Actor now receives preflight feedback** — LSP errors and arch violations
  are formatted into the Build Agent system prompt on retry.

- **`arch_check` dimension** added to `CHECK_MATRIX` in `check_profiles.py`
  for all `standard` and `full` profiles.

- **Dockerfile.sandbox**: added `dependency-cruiser@16`, `typescript-language-server`,
  `pyright`, and ArchUnit JAR pre-download.

### Why
Pattern Drift (LLM violating Clean Architecture silently) is now caught
deterministically in milliseconds — before the expensive test suite runs.
LSP errors (missing imports, type mismatches) are caught before any LLM Critic
sees the diff, saving tokens and reducing retry depth.

---

## Theme 2: Confidence Score Decay & Replan (Approaches 2, 4)

### What changed
- **`compute_confidence_score()`** — new pure function in `edges.py`
  Composite signal: attempt penalty + stagnation penalty + blast-radius penalty
  + critical-finding penalty. Returns `float` in `[0, 1]`.

- **`route_after_verifier`** now checks confidence score first.
  If `score < config.confidence_escalation_threshold` (default 0.25),
  escalates to HITL regardless of raw `attempt_count`.

- **New node: `ReplanNode`** (`nodes/replan.py`)
  - Triggered when all speculative branches fail and `replan_count < max_replan_attempts`
  - Plan Agent role (read-only, no tools) generates alternative strategies
    with explicit "avoids" explanation for each
  - Routes back to `value_node` (rescores the new strategy tree)
  - Increments `replan_count`; resets `attempt_count` and `exhausted_branches`

- **`route_after_speculative_branch`** updated: returns `"replan"` before
  `"hitl_escalation"` when replan budget remains.

- **New state fields**: `confidence_score: float`, `replan_count: int`

- **`WorkflowConfig`**: new fields `confidence_escalation_threshold: float = 0.25`,
  `max_replan_attempts: int = 1`

### Why
Raw `attempt_count >= MAX` is a blunt instrument.  Confidence decay catches
semantic stagnation, high blast-radius risk, and accumulated critical findings
early — before wasting more tokens on a path that cannot succeed.
ReplanNode gives the agent one structured second chance with a fresh strategy
tree before forcing human intervention.

---

## Theme 3: Living Test Inventory + Two-Phase Guardrail (Approaches 6, 7, 8)

### What changed
- **`tdd_gate.py` rewrite**
  - Test Oracle generates tests with permanent paths:
    - Frontend: `tests/e2e/features/<feature>.spec.ts` (Playwright, `getByRole` only)
    - Backend API: `tests/api/routes/<feature>.spec.ts` (sequence-based, multi-step)
    - Backend domain: `src/test/java/…/<feature>Test.java` (JUnit 5)
  - Enforces permanent directory convention — rejects `/tmp` paths
  - Tracks `test_inventory_paths` in state

- **`verifier.py` rewrite — Two-Phase Guardrail**
  - Phase 1: runs ALL tests in `/tests/` and `src/test/` (full regression sweep)
    → Phase 1 failure = immediate FIX_IMPL, Phase 2 never runs
  - Phase 2: runs only `test_inventory_paths` (new feature tests)
    → Phase 2 failure = implementation incomplete
  - Test deletion check: if `diff.operation == "delete"` on any `tests/` path → immediate rejection, Docker never called
  - `VerifierVerdict` now includes `phase1_passed: bool`, `phase2_passed: bool`

- **`memory_consolidation.py`**: commits test inventory files via `git add` + `git commit` with `[tests]` tag

- **Blast-radius cross-domain routing** (approach 5): when `blast_radius_map.schema_impact` is non-empty, Verifier also runs `tests/api/` suite

- **New `check_profiles.py` function**: `get_active_checks_with_blast_radius()`

### Why
Tests written in one session that are not committed disappear silently.
Agents in later sessions have no memory of them and repeat the same errors.
Committing tests as first-class code artifacts makes the regression history
cumulative: each session adds to the protection floor, never removes from it.

---

## Theme 4: Living Blueprints (Approaches 3, 11)

### What changed
- **`scout.py`**: reads `AGENTS.md` if present and injects as `agents_md_context`
  (truncated to 4 000 chars). Available to Actor, ValueNode, and ReplanNode.

- **`memory_consolidation.py`** — two new async helpers:
  - `_update_agents_md()`: Plan Agent appends to "Common Mistakes" and
    "Architecture Decisions" sections only; all other content preserved.
    Commits with `[skip ci]` tag.
  - `_update_arch_rules()`: Plan Agent writes ONE new rule to
    `.dependency-cruiser.json` (TS) or `ArchitectureTest.java` (Java)
    for each structural violation found this session.

- **`actor.py`**: `agents_md_context` injected into Build Agent system prompt
  under "AGENTS.MD PROJECT CONVENTIONS".

- **New files in repo root**:
  - `AGENTS.md` — human-readable living blueprint with placeholder sections
  - `.dependency-cruiser.json` — default TypeScript arch rules
  - `src/test/java/com/sacv/ArchitectureTest.java` — default Java arch rules

- **New state field**: `arch_rules_updated: bool`, `agents_md_context: str | None`

### Why
AgentMemory stores constraints as opaque vectors. AGENTS.md makes them
human-readable and version-controlled. Arch rule files make them
machine-executable. Together they form the "digital constitution":
the LLM cannot drift past these boundaries because they are enforced both
by procedural memory at prompt time and by deterministic tools at build time.

---

## Theme 5: Dual-Agent Terminology (Approach 2)

### What changed
- Actor system prompt role renamed from `"implementer"` to `"build_agent"`
- ReplanNode and ValueNode use `"plan_agent_*"` role names
- AGENTS.md updater uses `"plan_agent_docs"` role
- Arch rules updater uses `"plan_agent_arch_rules"` role

These are system prompt role labels that communicate intent to the LLM,
not separate API calls. The DIP boundary (AgentProvider ABC) remains unchanged;
swapping to a different provider is still one adapter file.

---

## Summary of New / Changed Files

| File | Status | Theme |
|---|---|---|
| `orchestration/state.py` | Modified | 1, 2, 3, 4 |
| `orchestration/config.py` | Modified | 2 |
| `orchestration/edges.py` | Modified | 1, 2 |
| `orchestration/graph.py` | Modified | 1, 2 |
| `nodes/bootstrap.py` | Modified | all (resets new fields) |
| `nodes/scout.py` | Modified | 4 |
| `nodes/actor.py` | Modified | 1, 4 |
| `nodes/tdd_gate.py` | Modified | 3 |
| `nodes/verifier.py` | Modified | 3, 5 |
| `nodes/memory_consolidation.py` | Modified | 3, 4 |
| `nodes/preflight_node.py` | **NEW** | 1, 9, 10 |
| `nodes/replan.py` | **NEW** | 2, 4 |
| `checks/routing/check_profiles.py` | Modified | 5, 9 |
| `Dockerfile.sandbox` | Modified | 1, 9, 10 |
| `AGENTS.md` | **NEW** | 3 |
| `.dependency-cruiser.json` | **NEW** | 9, 10, 11 |
| `src/test/java/…/ArchitectureTest.java` | **NEW** | 9, 10, 11 |
| `tests/unit/test_preflight_routing.py` | **NEW** | 1, 9, 10 |
| `tests/unit/test_confidence_score.py` | **NEW** | 2, 4 |
| `tests/unit/test_replan_routing.py` | **NEW** | 2, 4 |
| `tests/integration/test_two_phase_verifier.py` | **NEW** | 3, 5, 8 |
| `tests/integration/test_preflight_node.py` | **NEW** | 1, 9, 10 |

**Unchanged files** (copied verbatim from Phase 4):
`nodes/value_node.py`, `nodes/_scoring.py`, `nodes/_stagnation.py`,
`nodes/critics/*`, `nodes/speculative_branch.py`, `nodes/hitl_escalation.py`,
`nodes/mode_router.py`, all `adapters/`, `git/`, `docker/container_manager.py`,
`memory/`, `modes/`, `testing/stub_providers.py`, `testing/vcr_recorder.py`
