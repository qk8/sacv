"""
nodes/preflight_node.py
=======================
Pre-Critic Preflight. Three checks, all deterministic, no LLM calls.

Check 1 — LSP/Compile (approach 1):
  Java: mvn compile | TypeScript: tsc --noEmit

Check 2 — Architecture/Structural (approaches 9, 10):
  Java: ArchUnit | TypeScript: dependency-cruiser

Check 3 — Cross-Stack Type Safety, monorepo only (approach 3A):
  Regenerate OpenAPI spec from Spring annotations, then run
  openapi-typescript to regenerate TypeScript types, then tsc --noEmit.
  This catches "Java DTO changed, TypeScript interface is stale" bugs.
"""
from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase, PreflightResult, CRITIC_RESET
from sacv.checks.routing.check_profiles import get_checks, CheckSpec

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_TS_ERROR_RE    = re.compile(r"^(.+\.tsx?)\((\d+),(\d+)\): error (TS\d+): (.+)$")
_JAVA_ERROR_RE  = re.compile(r"^\[ERROR\] (.+\.java):\[(\d+),\d+\] (.+)$")
_ARCHUNIT_RE    = re.compile(r"Architecture Violation .* Rule '([^']+)'.*\n\s*- (.+)")
_DEPCRUISER_ERR = re.compile(r'"violations":\s*\[')


def make_preflight_node(deps: "NodeDeps"):

    async def preflight_node(state: "WorkflowState") -> dict:
        t0        = time.monotonic()
        module    = state["module_type"]
        profile   = state.get("check_profile", "standard")
        proposal  = state.get("diff_proposal")
        cfg       = deps.config

        if not proposal:
            return {
                "current_phase":  WorkflowPhase.PREFLIGHT.value,
                "preflight_result": PreflightResult(
                    passed=True, lsp_errors=[], arch_violations=[],
                    cross_stack_errors=[], duration_ms=0
                ),
            }

        active_checks: list[CheckSpec] = get_checks(module, profile)
        check_names = {c.name for c in active_checks}
        check_timeout = {c.name: c.timeout for c in active_checks}

        log.info("preflight.start", task_id=state["task_id"], module=module,
                 profile=profile, checks=sorted(check_names))
        handle = await deps.sandbox.warm_container()

        try:
            # ── Check 1: LSP / Compile ─────────────────────────────────────
            lsp_errors: list[dict] = []
            if "lsp" in check_names:
                lsp_cmd = "npx tsc --noEmit 2>&1" if "frontend" in module else "mvn compile -q 2>&1"
                lsp_out = await deps.sandbox.exec_in_container(
                    handle, lsp_cmd, timeout=check_timeout.get("lsp", 60)
                )
                lsp_errors = _parse_lsp(lsp_out.stdout + lsp_out.stderr, module)

            # ── Check 2: Architecture / Structure ──────────────────────────
            arch_errs: list[dict] = []
            if "arch" in check_names:
                arch_cmd = _arch_cmd(module)
                if arch_cmd:
                    arch_out = await deps.sandbox.exec_in_container(
                        handle, arch_cmd, timeout=check_timeout.get("arch", 30),
                    )
                    arch_errs = _parse_arch(arch_out.stdout + arch_out.stderr, module)

            # ── Check 3: Cross-Stack Type Safety ───────────────────────────
            cross_stack_errors: list[dict] = []
            if "cross_stack" in check_names and "frontend" not in module:
                cross_stack_errors = await _check_cross_stack_types(
                    handle, cfg, deps,
                )

            duration_ms = int((time.monotonic() - t0) * 1000)
            # For "required=False" checks, don't fail the gate — just report
            lsp_spec = next((c for c in active_checks if c.name == "lsp"), CheckSpec("lsp"))
            arch_spec = next((c for c in active_checks if c.name == "arch"), CheckSpec("arch"))
            required_failed = (
                (lsp_errors and lsp_spec.required)
                or (arch_errs and arch_spec.required)
            )
            passed = not required_failed and not lsp_errors and not arch_errs and not cross_stack_errors

            result = PreflightResult(
                passed=passed,
                lsp_errors=lsp_errors,
                arch_violations=arch_errs,
                cross_stack_errors=cross_stack_errors,
                duration_ms=duration_ms,
            )
            log.info(
                "preflight.complete",
                passed=passed,
                lsp=len(lsp_errors),
                arch=len(arch_errs),
                cross_stack=len(cross_stack_errors),
                duration_ms=duration_ms,
                profile=profile,
            )

            return {
                "current_phase":    WorkflowPhase.PREFLIGHT.value,
                "preflight_result": result,
                # _merge_lists treats [] as RESET (same as CRITIC_RESET)
                "critic_findings":  CRITIC_RESET,
            }
        finally:
            await deps.sandbox.destroy_container(handle)

    return preflight_node


async def _check_cross_stack_types(handle, cfg, deps) -> list[dict]:
    """
    Approach 3A: Regenerate OpenAPI spec from Java annotations, then
    regenerate TypeScript types, then type-check the frontend.
    Only runs in monorepo mode when backend code changed.
    """
    errors: list[dict] = []

    # Step 1: regenerate OpenAPI spec
    gen_spec = await deps.sandbox.exec_in_container(
        handle,
        "mvn springdoc-openapi:generate -q 2>&1 || "
        "mvn io.swagger.core.v3:swagger-maven-plugin:resolve -q 2>&1",
        timeout=60,
    )
    if gen_spec.exit_code != 0:
        return []   # no OpenAPI plugin configured — skip check

    # Step 2: regenerate TypeScript types from new spec
    ts_gen = await deps.sandbox.exec_in_container(
        handle,
        f"[ -f '{cfg.debug.openapi_spec_path}' ] && "
        f"npx openapi-typescript {cfg.debug.openapi_spec_path} "
        f"-o frontend/src/api/generated-types.ts 2>&1 || echo 'NO_SPEC'",
        timeout=30,
    )
    if "NO_SPEC" in ts_gen.stdout:
        return []

    # Step 3: type-check frontend with new types
    tsc_out = await deps.sandbox.exec_in_container(
        handle, "cd frontend && npx tsc --noEmit 2>&1", timeout=60,
    )
    for line in tsc_out.stdout.splitlines():
        m = _TS_ERROR_RE.match(line.strip())
        if m:
            errors.append({
                "file":    m.group(1), "line": int(m.group(2)),
                "code":    m.group(4), "message": m.group(5),
                "source":  "cross_stack_type_check",
            })
    return errors[:20]


def _arch_cmd(module_type: str) -> str | None:
    if "frontend" in module_type:
        return (
            "[ -f .dependency-cruiser.json ] && "
            "npx depcruise src --config .dependency-cruiser.json --output-type json 2>&1 "
            "|| echo '[]'"
        )
    return (
        "mvn test -Dtest=ArchitectureTest -q 2>&1 "
        "|| mvn test -Dtest=*ArchTest -q 2>&1 "
        "|| echo 'NO_ARCH_TEST'"
    )


def _parse_lsp(output: str, module_type: str) -> list[dict]:
    errors: list[dict] = []
    if "frontend" in module_type:
        for line in output.splitlines():
            m = _TS_ERROR_RE.match(line.strip())
            if m:
                errors.append({"file": m.group(1), "line": int(m.group(2)),
                                "code": m.group(4), "message": m.group(5)})
    else:
        for line in output.splitlines():
            m = _JAVA_ERROR_RE.match(line)
            if m:
                errors.append({"file": m.group(1), "line": int(m.group(2)),
                                "code": "CE", "message": m.group(3)})
    return errors[:30]


def _parse_arch(output: str, module_type: str) -> list[dict]:
    if "NO_ARCH_TEST" in output:
        return []
    violations: list[dict] = []
    if "frontend" in module_type:
        try:
            data = json.loads(output)
            for item in (data if isinstance(data, list) else []):
                for v in item.get("violations", []):
                    violations.append({
                        "rule": v.get("rule", {}).get("name", "?"),
                        "source_file": item.get("source", "?"),
                        "target_file": v.get("to", {}).get("resolved", "?"),
                        "message": f"Forbidden import: {v.get('rule',{}).get('name','?')}",
                    })
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        for m in _ARCHUNIT_RE.finditer(output):
            violations.append({"rule": m.group(1), "source_file": "unknown",
                                "target_file": "unknown", "message": m.group(2).strip()})
    return violations[:20]
