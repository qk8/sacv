"""
nodes/tdd_gate.py
=================
Test Oracle Agent — writes PERMANENT, COMMITTED failing tests first.

Refactoring changes (approaches 6, 7, 8):
  - Tests are written to the project's permanent test inventory:
      /tests/e2e/features/<feature_id>.spec.ts   (frontend)
      /tests/api/routes/<feature_id>.spec.ts     (backend API)
      /tests/unit/<module>/<feature_id>Test.java (backend domain)
  - Tests use accessibility tree selectors (getByRole) for frontend (approach 6).
  - Backend tests are sequence-based, multi-step state validators (approach 7).
  - The evidence file records which permanent paths were written (approach 8).
  - Test files are tracked in state.test_inventory_paths for Phase 1 guardrail.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase
from sacv.interfaces.agent_provider import AgentConfig
from sacv.orchestration.verifier_utils import add_agent_cost

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_ORACLE_BACKEND_SYSTEM = """\
You are a Test Oracle for backend services (Java/Spring Boot or TypeScript/Node).
Write FAILING tests that will pass once the implementation is complete.

Rules:
- Use JUnit 5 for Java, Vitest/Supertest for TypeScript.
- Write SEQUENCE-BASED tests: multi-step state chains (create → update → verify → list).
- Each test validates the full state after each step (not just status codes).
- Target public interfaces only — never implementation details.
- Tests must fail BEFORE implementation and pass AFTER.
- Output ONLY a JSON array: [{{"file_path": "tests/api/...", "content": "..."}}]
No explanation. No markdown. Only the JSON array.
"""

_ORACLE_FRONTEND_SYSTEM = """\
You are a Test Oracle for frontend components (TypeScript/React).
Write FAILING Playwright tests using ACCESSIBILITY TREE selectors only.

Rules:
- Use getByRole(), getByLabel(), getByText() — NEVER use CSS selectors or data-testid.
- Semantic visual assertions: describe what the user sees, not pixel coordinates.
- Each test must correspond to one acceptance criterion.
- Tests must fail BEFORE implementation and pass AFTER.
- Output ONLY a JSON array: [{{"file_path": "tests/e2e/features/...", "content": "..."}}]
No explanation. No markdown. Only the JSON array.
"""


def make_tdd_gate_node(deps: "NodeDeps"):

    async def tdd_gate_node(state: "WorkflowState") -> dict:
        task_id   = state["task_id"]
        strategy  = state.get("selected_strategy")
        desc      = state.get("task_description", "")
        module    = state["module_type"]
        is_front  = "frontend" in module

        log.info("tdd_gate.start", task_id=task_id, module=module)

        # ── Skip for test scenarios ────────────────────────────────────────
        if state.get("skip_tdd_gate"):
            log.info("tdd_gate.skipped", task_id=task_id)
            return {
                "red_phase_evidence_path": ".workflow/tdd-evidence/skipped.json",
                "test_inventory_paths":    [],
                "cumulative_cost_dollars": state.get("cumulative_cost_dollars", 0.0),
            }

        if strategy is None:
            log.error("tdd_gate.no_strategy")
            return {
                "red_phase_evidence_path": None,
                "test_inventory_paths":    [],
                "tdd_gate_attempts":       state.get("tdd_gate_attempts", 0) + 1,
            }

        # ── 1. Generate tests via Test Oracle ─────────────────────────────
        result = await deps.agent.run_task(
            prompt=(
                f"Task: {desc}\n\n"
                f"Strategy to implement:\n{json.dumps(strategy, indent=2)}\n\n"
                f"Feature ID: {_feature_id(task_id)}\n"
                "Write failing tests for this strategy."
            ),
            context={"strategy": strategy},
            config=AgentConfig(
                role="test_oracle",
                system_prompt=_ORACLE_FRONTEND_SYSTEM if is_front else _ORACLE_BACKEND_SYSTEM,
                max_turns=3,
                allowed_tools=[],
            ),
        )

        # ── Token budget tracking (CRIT-002) ──────────────────────────────
        new_cost = add_agent_cost(
            result, state.get("cumulative_cost_dollars", 0.0), deps.config,
        )

        try:
            test_files: list[dict] = json.loads(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("tdd_gate.parse_error", error=str(exc))
            return {
                "red_phase_evidence_path": None,
                "test_inventory_paths":    [],
                "tdd_gate_attempts":       state.get("tdd_gate_attempts", 0) + 1,
                "cumulative_cost_dollars": new_cost,
            }

        # ── 2. Write test files to PERMANENT locations in sandbox ──────────
        handle = await deps.sandbox.warm_container()

        try:
            permanent_paths: list[str] = []

            for tf in test_files:
                file_path = tf.get("file_path", "")
                content   = tf.get("content", "")
                if not file_path or not content:
                    continue
                # Enforce permanent directory convention (approach 8)
                file_path = _canonicalise_test_path(
                    file_path, module, task_id,
                    user_package=deps.config.debug.user_java_package,
                )
                # Sanitize file path to prevent shell injection; encode content
                # as base64 to avoid heredoc injection via test content.
                safe_path = shlex.quote(file_path)
                safe_dir = shlex.quote(str(Path(file_path).parent))
                encoded_content = base64.b64encode(content.encode()).decode("ascii")

                await deps.sandbox.exec_in_container(
                    handle,
                    f"mkdir -p {safe_dir}",
                    timeout=10,
                )
                await deps.sandbox.exec_in_container(
                    handle,
                    f"echo {shlex.quote(encoded_content)} | base64 -d > {safe_path}",
                    timeout=30,
                )
                permanent_paths.append(file_path)

            # ── 3. Run tests — MUST fail (red phase) ───────────────────────
            run_cmd = _test_command_for(module)
            run_result = await deps.sandbox.exec_in_container(
                handle, run_cmd, timeout=120,
            )

            if run_result.exit_code == 0:
                log.warning("tdd_gate.tests_passed_unexpectedly")
                return {
                    "red_phase_evidence_path": None,
                    "test_inventory_paths":    [],
                    "tdd_gate_attempts":       state.get("tdd_gate_attempts", 0) + 1,
                    "cumulative_cost_dollars": new_cost,
                }

            # ── 4. Commit test inventory to git (MEDIUM-003) ─────────────
            if permanent_paths:
                staged: list[str] = []
                for p in permanent_paths:
                    try:
                        await asyncio.to_thread(deps.git.stage_file, p)
                        staged.append(p)
                    except RuntimeError as exc:
                        log.error("tdd_gate.git_stage_failed", path=p, error=str(exc))

                if staged:
                    try:
                        await asyncio.to_thread(
                            deps.git.commit,
                            f"sacv: add test inventory for {task_id} [tests]",
                            add_all=False,
                        )
                        log.info("tdd_gate.tests_committed", count=len(staged))
                    except RuntimeError as exc:
                        log.error("tdd_gate.test_commit_failed", error=str(exc))

            # ── 5. Serialise evidence ────────────────────────────────────
            evidence_dir = deps.repo_root / ".workflow" / "tdd-evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_dir / f"{task_id}.json"
            evidence = {
                "task_id":          task_id,
                "permanent_paths":  permanent_paths,
                "failure_output":   (run_result.stdout + run_result.stderr)[:2000],
            }
            evidence_path.write_text(json.dumps(evidence, indent=2))

            log.info(
                "tdd_gate.red_phase_confirmed",
                files_written=len(permanent_paths),
                evidence=str(evidence_path),
            )

            return {
                "current_phase":          WorkflowPhase.ACTOR.value,
                "red_phase_evidence_path": str(evidence_path),
                "test_inventory_paths":   permanent_paths,
                "cumulative_cost_dollars": new_cost,
            }
        finally:
            await deps.sandbox.destroy_container(handle)

    return tdd_gate_node


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feature_id(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]", "-", task_id)[:32].lower()


def _canonicalise_test_path(
    path: str, module_type: str, task_id: str,
    user_package: str = "com.example",  # matches DebugConfig.user_java_package default
) -> str:
    """
    Enforce permanent test inventory directory convention (approach 8):
      frontend  → tests/e2e/features/<feature>.spec.ts
      backend   → tests/api/routes/<feature>.spec.ts  |  tests/unit/<module>/<feature>Test.java
    Rejects /tmp or any transient path.
    """
    fid = _feature_id(task_id)
    if "frontend" in module_type:
        if not path.startswith("tests/e2e/"):
            path = f"tests/e2e/features/{fid}.spec.ts"
    elif "api" in module_type:
        if not path.startswith("tests/api/"):
            path = f"tests/api/routes/{fid}.spec.ts"
    else:
        pkg_path = user_package.replace(".", "/")
        if not path.startswith("tests/unit/") and not path.startswith("src/test/"):
            path = f"src/test/java/{pkg_path}/{fid}Test.java"
    return path


def _test_command_for(module_type: str) -> str:
    if "frontend" in module_type:
        return "npx playwright test --reporter=line 2>&1 || npm test -- --watchAll=false --ci 2>&1"
    if "api" in module_type:
        return "npm test -- --testPathPattern=tests/api --watchAll=false 2>&1"
    return "mvn test -q 2>&1"
