"""
nodes/memory_consolidation.py
==============================
Post-task memory consolidation — the "sleep cycle".

Refactoring additions (approaches 3, 8, 11):
  1. COMMIT TEST INVENTORY (approach 8):
     All test files in state.test_inventory_paths are committed to the repo
     alongside the production code. Tests become first-class git citizens.

  2. UPDATE AGENTS.MD (approach 3):
     The Lesson Learned payload is summarised and written back to AGENTS.md
     under "## Common Mistakes" and "## Architecture Decisions" sections.

  3. UPDATE ARCH RULES (approach 11):
     If structural violations were found this session, the Plan Agent
     updates .dependency-cruiser.json (TypeScript) or adds an ArchUnit rule
     (Java) to prevent the same violation in future sessions.

  4. RECORD GREEN COMMIT (approach 8):
     After a successful commit, the SHA is recorded as the new green baseline
     for HITL hard-reset.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase, LessonLearned
from sacv.interfaces.memory_provider import EpisodicEvent
from sacv.interfaces.agent_provider import AgentConfig

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_AGENTS_MD = Path("AGENTS.md")

_AGENTS_MD_UPDATER_SYSTEM = """\
You are a technical writer maintaining a project's AGENTS.md file.
Update ONLY the '## Common Mistakes' and '## Architecture Decisions' sections.
Keep ALL other existing content exactly as-is.
Output the complete updated AGENTS.md content.
No explanation. No markdown fences. Only the raw file content.
"""

_ARCH_RULE_UPDATER_SYSTEM = """\
You are an architect updating structural linting rules.
Given a list of architecture violations that occurred in this session,
add ONE new rule to prevent them from recurring.

For TypeScript output a JSON object matching .dependency-cruiser.json "forbidden" rule format.
For Java output a JUnit 5 ArchUnit @ArchTest method body.
Output ONLY the new rule. No explanation. No markdown.
"""


def make_memory_consolidation_node(deps: "NodeDeps"):

    async def memory_consolidation_node(state: "WorkflowState") -> dict:
        session_id = state["session_id"]
        task_id    = state["task_id"]
        correction = state["correction_state"]
        verdict    = state.get("verifier_verdict")
        escalation = state.get("escalation_payload")
        findings   = state.get("critic_findings", [])
        inv_paths  = state.get("test_inventory_paths", [])
        module     = state["module_type"]

        log.info("memory_consolidation.start", task_id=task_id)

        # ── 1. COMMIT TEST INVENTORY (approach 8) ─────────────────────────
        committed_tests: list[str] = []
        if inv_paths:
            committed_tests = await _commit_test_inventory(inv_paths, task_id, deps)

        # ── 2. COMMIT PRODUCTION CODE ─────────────────────────────────────
        green_sha = await _commit_production_code(task_id, deps)

        # ── 3. BUILD LESSON LEARNED ───────────────────────────────────────
        correction_type = "none"
        if escalation:
            correction_type = "hitl"
        elif correction["attempt_count"] > 1:
            correction_type = "self_correction"
        elif findings:
            correction_type = "critic_guided"

        lesson = LessonLearned(
            task_id=task_id,
            pattern_discovered=_derive_pattern(state),
            negative_constraints=_extract_constraints(findings, escalation),
            blast_radius_learned=state.get("blast_radius_map") or {},
            correction_type=correction_type,
            session_duration_ms=0,
        )

        # ── 4. PERSIST TO AGENTMEMORY ─────────────────────────────────────
        await deps.memory.store_episodic(EpisodicEvent(
            session_id=session_id,
            event_type="lesson_learned",
            payload=dict(lesson),
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        await deps.memory.purge_noise(session_id)
        await deps.memory.consolidate_session(session_id)

        # ── 5. UPDATE AGENTS.MD (approach 3) ──────────────────────────────
        agents_md_updated = await _update_agents_md(lesson, state, deps)

        # ── 6. UPDATE ARCH RULES (approach 11) ────────────────────────────
        arch_rules_updated = False
        preflight = state.get("preflight_result") or {}
        arch_violations = preflight.get("arch_violations", [])
        if arch_violations:
            arch_rules_updated = await _update_arch_rules(arch_violations, module, deps)

        log.info(
            "memory_consolidation.complete",
            correction_type=correction_type,
            tests_committed=len(committed_tests),
            agents_md_updated=agents_md_updated,
            arch_rules_updated=arch_rules_updated,
        )

        return {
            "current_phase":     WorkflowPhase.COMPLETE.value,
            "lesson_learned":    lesson,
            "arch_rules_updated": arch_rules_updated,
        }

    return memory_consolidation_node


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _commit_test_inventory(
    paths: list[str], task_id: str, deps: "NodeDeps"
) -> list[str]:
    """Non-blocking wrapper: all subprocess work runs in a thread pool."""

    def _sync_work() -> list[str]:
        import subprocess
        for p in paths:
            if Path(p).exists():
                subprocess.run(
                    ["git", "add", p],
                    capture_output=True, timeout=10,
                )
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
        )
        staged = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if staged:
            subprocess.run(
                ["git", "commit", "-m",
                 f"sacv: add test inventory for {task_id} [tests]"],
                capture_output=True, timeout=15,
            )
            log.info("memory_consolidation.tests_committed", count=len(staged))
        return staged

    try:
        return await asyncio.to_thread(_sync_work)
    except Exception as exc:
        log.warning("memory_consolidation.test_commit_failed", error=str(exc))
        return []


async def _commit_production_code(task_id: str, deps: "NodeDeps") -> str:
    """Commit production code and record green SHA (non-blocking)."""
    def _sync_work() -> str:
        try:
            sha = deps.git.commit(
                f"sacv: implement {task_id}", add_all=True
            )
            deps.git.record_green_commit(sha)
            return sha
        except Exception as exc:
            log.warning("memory_consolidation.commit_failed", error=str(exc))
            return ""
    return await asyncio.to_thread(_sync_work)


async def _update_agents_md(
    lesson: "LessonLearned",
    state:  "WorkflowState",
    deps:   "NodeDeps",
) -> bool:
    """
    Append new learnings to AGENTS.md (approach 3).
    Uses Plan Agent in read-only mode to produce the updated file.
    """
    try:
        current = _AGENTS_MD.read_text(encoding="utf-8") if _AGENTS_MD.exists() else _default_agents_md()

        result = await deps.agent.run_task(
            prompt=(
                f"New lesson learned:\n{json.dumps(dict(lesson), indent=2)}\n\n"
                f"Violations fixed this session:\n"
                f"{json.dumps(state.get('preflight_result') or {}, indent=2)}\n\n"
                f"Current AGENTS.md:\n{current[:3000]}"
            ),
            context={},
            config=AgentConfig(
                role="plan_agent_docs",
                system_prompt=_AGENTS_MD_UPDATER_SYSTEM,
                max_turns=1,
                allowed_tools=[],
            ),
        )
        _AGENTS_MD.write_text(result.content, encoding="utf-8")
        deps.git.commit(
            f"sacv: update AGENTS.md after {state['task_id']} [skip ci]",
            add_all=False,
        )
        return True
    except Exception as exc:
        log.warning("memory_consolidation.agents_md_failed", error=str(exc))
        return False


async def _update_arch_rules(
    violations: list[dict],
    module_type: str,
    deps: "NodeDeps",
) -> bool:
    """
    Add a new rule to .dependency-cruiser.json (TS) or ArchUnit test (Java)
    to prevent recurring architectural violations (approach 11).
    """
    try:
        is_frontend = "frontend" in module_type
        config_file = Path(".dependency-cruiser.json" if is_frontend else
                           "src/test/java/com/sacv/ArchitectureTest.java")

        current_content = config_file.read_text(encoding="utf-8") if config_file.exists() else ""

        result = await deps.agent.run_task(
            prompt=(
                f"Architecture violations found:\n{json.dumps(violations, indent=2)}\n\n"
                f"Current rule file content:\n{current_content[:2000]}\n\n"
                f"Output only the new rule to add (not the full file)."
            ),
            context={},
            config=AgentConfig(
                role="plan_agent_arch_rules",
                system_prompt=_ARCH_RULE_UPDATER_SYSTEM,
                max_turns=1,
                allowed_tools=[],
            ),
        )

        new_rule = result.content.strip()
        if new_rule and is_frontend:
            _inject_depcruiser_rule(config_file, new_rule)
        elif new_rule:
            _inject_archunit_rule(config_file, new_rule)

        log.info("memory_consolidation.arch_rule_added", module=module_type)
        return True
    except Exception as exc:
        log.warning("memory_consolidation.arch_rules_failed", error=str(exc))
        return False


def _inject_depcruiser_rule(config_file: Path, new_rule: str) -> None:
    try:
        config = json.loads(config_file.read_text()) if config_file.exists() else {"forbidden": []}
        rule   = json.loads(new_rule)
        config.setdefault("forbidden", []).append(rule)
        config_file.write_text(json.dumps(config, indent=2))
    except Exception:
        config_file.write_text(new_rule)   # fallback: write raw


def _inject_archunit_rule(config_file: Path, new_rule: str) -> None:
    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(_default_archunit_class(new_rule))
        return
    content = config_file.read_text()
    # Insert before the last closing brace
    content = content.rstrip()
    if content.endswith("}"):
        content = content[:-1] + f"\n\n    {new_rule}\n}}"
    config_file.write_text(content)


def _derive_pattern(state: "WorkflowState") -> str:
    parts = [f"module={state['module_type']}", f"mode={state['project_mode']}"]
    verdict = state.get("verifier_verdict")
    if verdict and verdict["test_result"] == "PASS":
        parts.append(f"resolved_in={state['correction_state']['attempt_count']}_attempts")
    if state["correction_state"].get("stagnation_pattern", "none") != "none":
        parts.append(f"stagnation={state['correction_state']['stagnation_pattern']}")
    if state.get("replan_count", 0) > 0:
        parts.append(f"replanned={state['replan_count']}x")
    return " | ".join(parts)


def _extract_constraints(findings: list[dict], escalation: dict | None) -> list[str]:
    cs: list[str] = []
    for f in findings:
        if f.get("severity") == "critical":
            cs.append(f"[{f['critic'].upper()}] {f['message']} → {f['resolution_hint']}")
    if escalation:
        for h in escalation.get("resolution_hints", []):
            cs.append(f"[HITL] {h['hint']}")
    return cs


def _default_agents_md() -> str:
    return (
        "# AGENTS.md — Living Project Blueprint\n\n"
        "## Architecture Overview\n_Auto-generated by SACV workflow._\n\n"
        "## Common Mistakes\n_Populated automatically after each session._\n\n"
        "## Architecture Decisions\n_Populated automatically after each session._\n\n"
        "## Module Conventions\n_See .dependency-cruiser.json and ArchitectureTest.java._\n"
    )


def _default_archunit_class(rule: str) -> str:
    return (
        "package com.sacv;\n\nimport com.tngtech.archunit.junit.ArchTest;\n"
        "import com.tngtech.archunit.lang.ArchRule;\nimport static "
        "com.tngtech.archunit.library.Architectures.layeredArchitecture;\n\n"
        "class ArchitectureTest {\n\n"
        f"    {rule}\n}}\n"
    )
