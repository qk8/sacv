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
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.orchestration.state import WorkflowPhase, LessonLearned
from sacv.interfaces.memory_provider import EpisodicEvent
from sacv.interfaces.agent_provider import AgentConfig
from sacv.orchestration.verifier_utils import add_agent_cost
from sacv.nodes._structured_output import extract_structured, AgentsMdUpdate, StructuredOutputError

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_AGENTS_MD_UPDATER_SYSTEM = """\
You are a technical writer. Based on the new lesson learned, output ONLY the
updated content for two sections as a JSON object with keys
"common_mistakes" and "architecture_decisions".
No other keys. No explanation. Only the JSON object.
"""

_ARCH_RULE_UPDATER_SYSTEM = """\
You are an architect updating structural linting rules.
Given a list of architecture violations that occurred in this session,
add ONE new rule to prevent them from recurring.

For TypeScript output a JSON object matching .dependency-cruiser.json "forbidden" rule format.
For Java output a JUnit 5 ArchUnit @ArchTest method body.
Output ONLY the new rule. No explanation. No markdown.
"""


def make_memory_consolidation_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def memory_consolidation_node(state: "WorkflowState") -> dict[str, object]:
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

        # ── 2. COMMIT PRODUCTION CODE (do NOT record green SHA yet) ──────
        await _commit_production_code_no_record(task_id, deps)

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

        # ── 5. UPDATE AGENTS.MD (approach 3) ──────────────────────────────
        agents_md_updated, cost_after_agents = await _update_agents_md(lesson, state, deps)

        # ── 6. UPDATE ARCH RULES (approach 11) ────────────────────────────
        arch_rules_updated = False
        cost_after_arch = cost_after_agents
        preflight = state.get("preflight_result") or {}
        arch_violations = preflight.get("arch_violations", [])
        if arch_violations:
            arch_rules_updated, cost_after_arch = await _update_arch_rules(
                arch_violations, module, deps, cost_after_agents,
            )
        else:
            arch_rules_updated = False

        # ── 7. RECORD GREEN SHA LAST — after all commits are done ─────────
        # (BUG-011 fix: was recorded at step 2, losing AGENTS.md and arch
        # rule commits on HITL hard-reset)
        try:
            final_sha = await asyncio.to_thread(deps.git.head_sha)
        except RuntimeError as exc:
            log.warning("memory_consolidation.get_head_sha_failed", error=str(exc))
            final_sha = ""
        if final_sha:
            await asyncio.to_thread(deps.git.record_green_commit, final_sha)
        else:
            log.warning("memory_consolidation.skipping_green_sha_record")

        # ── 8. Clean up speculative branch stash refs ─────────────────────
        stash_ref = state.get("speculative_stash_ref")
        if stash_ref:
            try:
                # DROP the stash — do not reapply. The spec branch work is
                # superseded by the committed production code. Reapplying
                # would reintroduce rejected changes on top of the green
                # baseline.
                await asyncio.to_thread(deps.git.stash_drop, stash_ref)
                log.info("memory_consolidation.stash_dropped", ref=stash_ref)
            except Exception as exc:
                # Non-fatal: log and continue (stash may already be gone)
                log.warning("memory_consolidation.stash_drop_failed",
                            error=str(exc))

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
            "cumulative_cost_dollars": cost_after_arch,
        }

    return memory_consolidation_node


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _commit_test_inventory(
    paths: list[str], task_id: str, deps: "NodeDeps"
) -> list[str]:
    """Commit test inventory files via GitProvider (testable, CWD-independent)."""
    staged: list[str] = []
    for p in paths:
        if not Path(p).exists():
            log.warning("memory_consolidation.test_file_missing", path=p)
            continue
        try:
            await asyncio.to_thread(deps.git.stage_file, p)
            staged.append(p)
        except RuntimeError as exc:
            log.error("memory_consolidation.git_add_failed", path=p, error=str(exc))

    if not staged:
        return []

    try:
        await asyncio.to_thread(
            deps.git.commit,
            f"sacv: add test inventory for {task_id} [tests]",
            add_all=False,  # files already staged above
        )
        log.info("memory_consolidation.tests_committed", count=len(staged))
        return staged
    except RuntimeError as exc:
        log.error("memory_consolidation.test_commit_failed", error=str(exc))
        return []



async def _commit_production_code_no_record(task_id: str, deps: "NodeDeps") -> str:
    """Commit production code WITHOUT recording green SHA (non-blocking).

    Used when additional commits (AGENTS.md, arch rules) will follow;
    green SHA is recorded after all commits are complete (BUG-011 fix).
    """
    def _sync_work() -> str:
        try:
            sha = deps.git.commit(
                f"sacv: implement {task_id}", add_all=True
            )
            return str(sha)
        except Exception as exc:
            log.warning("memory_consolidation.commit_failed", error=str(exc))
            return ""
    return str(await asyncio.to_thread(_sync_work))


async def _update_agents_md(
    lesson: "LessonLearned",
    state:  "WorkflowState",
    deps:   "NodeDeps",
) -> tuple[bool, float]:
    """
    Append new learnings to AGENTS.md (approach 3).
    Uses section-targeted update: LLM outputs only the two updated
    sections as JSON, which are spliced back into the existing file.
    This prevents truncation-based data loss when the file grows.
    """
    cost = state.get("cumulative_cost_dollars", 0.0)
    try:
        agents_md_path = deps.repo_root / "AGENTS.md"
        current = agents_md_path.read_text(encoding="utf-8") if agents_md_path.exists() else _default_agents_md()

        # ── 5b. LLM call with structured output + retry ───────────────────
        try:
            structured = await extract_structured(
                agent=deps.agent,
                prompt=(
                    f"New lesson learned:\n{json.dumps(dict(lesson), indent=2)}\n\n"
                    f"Violations fixed this session:\n"
                    f"{json.dumps(state.get('preflight_result') or {}, indent=2)}\n\n"
                    f"Current AGENTS.md sections (for context):\n"
                    f"{_extract_section(current, 'Common Mistakes')}\n"
                    f"{_extract_section(current, 'Architecture Decisions')}"
                ),
                response_model=AgentsMdUpdate,
                system_prompt=_AGENTS_MD_UPDATER_SYSTEM,
                context={},
                max_retries=3,
                allowed_tools=[],
            )
            updates = structured.data.model_dump()
            # ── Token budget tracking ────────────────────────────────────
            if structured.agent_result:
                cost = add_agent_cost(structured.agent_result, cost, deps.config)
        except StructuredOutputError:
            log.warning("memory_consolidation.agents_md_parse_failed")
            return False, state.get("cumulative_cost_dollars", 0.0)

        updated = _splice_sections(current, updates)
        agents_md_path.write_text(updated, encoding="utf-8")

        await asyncio.to_thread(deps.git.stage_file, str(agents_md_path))
        await asyncio.to_thread(
            deps.git.commit,
            f"sacv: update AGENTS.md after {state['task_id']} [skip ci]",
            add_all=False,
        )
        return True, cost
    except Exception as exc:
        log.warning("memory_consolidation.agents_md_failed", error=str(exc))
        return False, state.get("cumulative_cost_dollars", 0.0)


def _extract_section(content: str, section_name: str) -> str:
    """Extract the content of a ## Section from AGENTS.md for context."""
    import re
    pattern = rf"## {re.escape(section_name)}\s*\n(.*?)(?=## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return f"## {section_name}:\n{match.group(1).strip()[:500]}"
    return f"## {section_name}: _not present_"


def _splice_sections(content: str, updates: dict[str, Any]) -> str:
    """
    Replace ## Common Mistakes and ## Architecture Decisions sections
    in the existing AGENTS.md content with updated versions from the LLM.
    Returns the full updated content.
    """
    import re

    for section_name, new_content in updates.items():
        if not new_content:
            continue
        pattern = rf"(## {re.escape(section_name)}\s*\n)(.*?)(?=## |\Z)"
        # Use lambda to avoid regex backreference interpretation of new_content
        content = re.sub(
            pattern,
            lambda m: m.group(1) + new_content.strip() + "\n",
            content,
            count=1,
            flags=re.DOTALL,
        )
    return content


async def _update_arch_rules(
    violations: list[dict[str, Any]],
    module_type: str,
    deps: "NodeDeps",
    current_cost: float = 0.0,
) -> tuple[bool, float]:
    """
    Add a new rule to .dependency-cruiser.json (TS) or ArchUnit test (Java)
    to prevent recurring architectural violations (approach 11).

    Returns (success, updated_cost).
    Cost is tracked via add_agent_cost and carried forward.
    """
    new_cost = current_cost
    try:
        is_frontend = "frontend" in module_type
        user_package = deps.config.debug.user_java_package
        config_file = Path(".dependency-cruiser.json" if is_frontend else
                           f"src/test/java/{user_package.replace('.', '/')}/ArchitectureTest.java")

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

        # ── Token budget tracking (CRIT-002) ──────────────────────────────
        new_cost = add_agent_cost(result, current_cost, deps.config)

        new_rule = result.content.strip()
        if new_rule and is_frontend:
            _inject_depcruiser_rule(config_file, new_rule)
        elif new_rule:
            _inject_archunit_rule(config_file, new_rule, user_package)

        # Commit the updated rule file so it survives HITL reset_hard
        await asyncio.to_thread(deps.git.stage_file, str(config_file))
        await asyncio.to_thread(
            deps.git.commit,
            f"sacv: add arch rule for {module_type} violations [skip ci]",
            add_all=False,
        )

        log.info("memory_consolidation.arch_rule_added", module=module_type)
        return True, new_cost
    except Exception as exc:
        log.warning("memory_consolidation.arch_rules_failed", error=str(exc))
        return False, new_cost


def _inject_depcruiser_rule(config_file: Path, new_rule: str) -> None:
    """Inject a dep-cruiser rule with validation (ARCH-003 fix)."""
    try:
        rule = json.loads(new_rule)
    except json.JSONDecodeError as exc:
        log.error("memory_consolidation.invalid_depcruiser_rule", error=str(exc),
                  content=new_rule[:200])
        return   # Do not write a broken rule

    # Validate required fields before appending
    required = {"name", "from", "to"}
    if not required.issubset(rule.keys()):
        log.warning("memory_consolidation.depcruiser_rule_missing_fields",
                    missing=required - rule.keys())
        return

    config_text = config_file.read_text() if config_file.exists() else '{"forbidden":[]}'
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        config = {"forbidden": []}

    # ── IDEMPOTENCY: skip if rule with same name already exists ─────────
    existing_names = {r.get("name") for r in config.get("forbidden", [])}
    if rule.get("name") in existing_names:
        log.info("memory_consolidation.depcruiser_rule_already_exists",
                 name=rule.get("name"))
        return

    config.setdefault("forbidden", []).append(rule)
    config_file.write_text(json.dumps(config, indent=2))


def _inject_archunit_rule(
    config_file: Path,
    new_rule: str,
    user_package: str = "com.sacv",
) -> None:
    """Inject an ArchUnit rule with validation (ARCH-003 fix)."""
    # Validate: must be a syntactically plausible @ArchTest method
    if "@ArchTest" not in new_rule:
        log.warning("memory_consolidation.archunit_rule_no_annotation",
                    content=new_rule[:200])
        return
    # Count braces — unbalanced braces would break the class
    if new_rule.count("{") != new_rule.count("}"):
        log.warning("memory_consolidation.archunit_rule_unbalanced_braces")
        return
    # Must contain an ArchRule return type
    if "ArchRule" not in new_rule and "ArchRules" not in new_rule:
        log.warning("memory_consolidation.archunit_rule_missing_archrule_type",
                    content=new_rule[:100])
        return

    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(_default_archunit_class(new_rule, user_package))
        return

    content = config_file.read_text()

    # ── IDEMPOTENCY: extract rule name and check for duplicates ────────
    import re
    name_match = re.search(r"public\s+static\s+final\s+Arch[Rr]ule\s+(\w+)", new_rule)
    if name_match:
        rule_name = name_match.group(1)
        # Check if a rule with the same name already exists in the file
        if re.search(rf"public\s+static\s+final\s+Arch[Rr]ule\s+{re.escape(rule_name)}", content):
            log.info("memory_consolidation.archunit_rule_already_exists",
                     name=rule_name)
            return

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


def _extract_constraints(findings: list[dict[str, Any]], escalation: dict[str, Any] | None) -> list[str]:
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


def _default_archunit_class(rule: str, user_package: str = "com.sacv") -> str:
    return (
        f"package {user_package};\n\n"
        "import com.tngtech.archunit.junit.ArchTest;\n"
        "import com.tngtech.archunit.lang.ArchRule;\n"
        "import static com.tngtech.archunit.library.Architectures.layeredArchitecture;\n\n"
        "class ArchitectureTest {\n\n"
        f"    {rule}\n}}\n"
    )
