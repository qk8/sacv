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
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.orchestration.state import WorkflowPhase, PreflightResult, CRITIC_RESET
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.checks.routing.check_profiles import get_checks, CheckSpec

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_TS_ERROR_RE    = re.compile(r"^(.+\.tsx?)\((\d+),(\d+)\): error (TS\d+): (.+)$")
_JAVA_ERROR_RE  = re.compile(r"^\[ERROR\] (.+\.java):\[(\d+),\d+\] (.+)$")
_ARCHUNIT_RE    = re.compile(r"Architecture Violation .* Rule '([^']+)'.*\n\s*- (.+)")
_ARCHUNIT_RULE_RE   = re.compile(r"Architecture Violation.*[Rr]ule '([^']+)'")
_ARCHUNIT_DETAIL_RE = re.compile(r"^\s+-\s+(.+)")
_DEPCRUISER_ERR = re.compile(r'"violations":\s*\[')


def make_preflight_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def preflight_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "preflight")
        async with node_timer("preflight", state=state) as timing:
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
                        cross_stack_errors=[], blast_errors=[],
                        repair_suggestions=[], duration_ms=0
                    ),
                    "critic_findings": CRITIC_RESET,
                }

            active_checks: list[CheckSpec] = get_checks(module, profile, monorepo_mode=cfg.monorepo_mode)
            check_names = {c.name for c in active_checks}
            check_timeout = {c.name: c.timeout for c in active_checks}

            log.info("preflight.start", task_id=state["task_id"], module=module,
                     profile=profile, checks=sorted(check_names))
            handle = await deps.sandbox.warm_container()

            try:
                # ── Check 1: LSP / Compile ─────────────────────────────────────
                lsp_errors: list[dict[str, Any]] = []
                if "lsp" in check_names:
                    lsp_cmd = "npx tsc --noEmit 2>&1" if "frontend" in module else "mvn compile -q 2>&1"
                    lsp_out = await deps.sandbox.exec_in_container(
                        handle, lsp_cmd, timeout=check_timeout.get("lsp", 60)
                    )
                    lsp_errors = _parse_lsp(lsp_out.stdout + lsp_out.stderr, module)

                # ── Check 2: Architecture / Structure ──────────────────────────
                arch_errs: list[dict[str, Any]] = []
                if "arch" in check_names:
                    arch_cmd = _arch_cmd(module)
                    if arch_cmd:
                        arch_out = await deps.sandbox.exec_in_container(
                            handle, arch_cmd, timeout=check_timeout.get("arch", 30),
                        )
                        arch_errs = _parse_arch(arch_out.stdout + arch_out.stderr, module)

                # ── Check 3: Cross-Stack Type Safety ───────────────────────────
                cross_stack_errors: list[dict[str, Any]] = []
                if "cross_stack" in check_names and "frontend" not in module:
                    cross_stack_errors = await _check_cross_stack_types(
                        handle, cfg, deps,
                    )

                # ── Check 4: Blast-radius file count guard ─────────────────────
                blast_errors: list[dict[str, Any]] = []
                blast_map = state.get("blast_radius_map") or {}
                if "blast_radius" in check_names:
                    affected_count = len(blast_map.get("affected_files", []))
                    max_files = cfg.max_blast_files
                    if affected_count > max_files:
                        blast_errors.append({
                            "rule": "blast_radius_limit",
                            "message": (
                                f"Change affects {affected_count} files "
                                f"(limit: {max_files}). Consider splitting the task."
                            ),
                        })

                duration_ms = int((time.monotonic() - t0) * 1000)
                # For "required=False" checks, don't fail the gate — just report.
                lsp_spec  = next((c for c in active_checks if c.name == "lsp"),  CheckSpec("lsp"))
                arch_spec = next((c for c in active_checks if c.name == "arch"), CheckSpec("arch"))
                # Only checks marked required=True block forward progress.
                # cross_stack is always required when it runs.
                required_failed = (
                    (lsp_errors   and lsp_spec.required)
                    or (arch_errs and arch_spec.required)
                    or bool(cross_stack_errors)
                )
                # passed iff no required check failed.
                # Non-required check findings are still recorded in PreflightResult for reporting.
                passed = not required_failed

                # CONCERN-2: Compute structured repair suggestions
                repair_suggestions = _compute_repair_suggestions(
                    lsp_errors, arch_errs, cross_stack_errors, blast_errors,
                    module, blast_map,
                )

                result = PreflightResult(
                    passed=passed,
                    lsp_errors=lsp_errors,
                    arch_violations=arch_errs,
                    cross_stack_errors=cross_stack_errors,
                    blast_errors=blast_errors,
                    repair_suggestions=repair_suggestions,
                    duration_ms=duration_ms,
                )
                timing["passed"] = passed
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
                    # CRITIC_RESET clears stale findings from the previous Actor attempt.
                    # NOTE: returning [] would NOT clear findings — it is a no-op in
                    # _merge_lists. Only CRITIC_RESET produces a reset. Do not change
                    # this to [].
                    "critic_findings":  CRITIC_RESET,
                }
            finally:
                await deps.sandbox.destroy_container(handle)

    return preflight_node


async def _check_cross_stack_types(handle: Any, cfg: Any, deps: "NodeDeps") -> list[dict[str, Any]]:
    """
    Approach 3A: Regenerate OpenAPI spec from Java annotations, then
    regenerate TypeScript types, then type-check the frontend.
    Only runs in monorepo mode when backend code changed.
    """
    errors: list[dict[str, Any]] = []

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


def _parse_lsp(output: str, module_type: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
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


def _parse_arch(output: str, module_type: str) -> list[dict[str, Any]]:
    if "NO_ARCH_TEST" in output:
        return []
    if "frontend" in module_type:
        violations: list[dict[str, Any]] = []
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
        return violations[:20]
    else:
        return _parse_java_archunit(output)


def _parse_java_archunit(output: str) -> list[dict[str, Any]]:
    """
    Parse ArchUnit violations using a two-pass state machine.
    Tolerates intermediate lines between the rule header and violation details.
    """
    violations: list[dict[str, Any]] = []
    current_rule: str | None = None

    for line in output.splitlines():
        rule_match = _ARCHUNIT_RULE_RE.search(line)
        if rule_match:
            current_rule = rule_match.group(1)
            continue

        if current_rule:
            detail_match = _ARCHUNIT_DETAIL_RE.match(line)
            if detail_match:
                violations.append({
                    "rule":        current_rule,
                    "source_file": "unknown",
                    "target_file": "unknown",
                    "message":     detail_match.group(1).strip(),
                })
            elif line.strip() and not line.strip().startswith("There is"):
                # If a non-empty, non-meta line appears that doesn't look like
                # a violation detail, assume we've moved past this rule block.
                if not _ARCHUNIT_RULE_RE.search(line):
                    current_rule = None

    return violations[:20]


def _compute_repair_suggestions(
    lsp_errors: list[dict[str, Any]],
    arch_violations: list[dict[str, Any]],
    cross_stack_errors: list[dict[str, Any]],
    blast_errors: list[dict[str, Any]],
    module: str,
    blast_map: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Compute structured repair suggestions for the Actor (CONCERN-2).

    Each suggestion is a dict with 'category' and 'text' keys.
    The actor's system prompt includes these so it knows *how* to fix
    violations rather than just *what* they are.
    """
    suggestions: list[dict[str, str]] = []

    if lsp_errors:
        by_file: dict[str, list[dict[str, Any]]] = {}
        for err in lsp_errors[:10]:
            by_file.setdefault(err.get("file", "?"), []).append(err)

        for fpath, errors in list(by_file.items())[:5]:
            symbols = []
            for e in errors:
                msg = e.get("message", "")
                for pattern in [
                    r"cannot find symbol\s+(?:method|variable|class)\s+(\w+)",
                    r"cannot find name\s+['\"](\w+)['\"]",
                    r"cannot find module\s+['\"]([^'\"]+)['\"]",
                    r"'(\w+)' does not exist",
                ]:
                    import re as _re
                    m = _re.search(pattern, msg)
                    if m:
                        symbols.append(m.group(1))
                        break
                if not symbols and errors.index(e) == 0:
                    parts = msg.split()
                    for p in reversed(parts):
                        if p.isalnum() and len(p) > 2:
                            symbols.append(p.rstrip("';,"))
                            break
            unique_symbols = list(dict.fromkeys(symbols))[:5]
            if unique_symbols:
                suggestions.append({
                    "category": "compile",
                    "text": f"Fix imports in {fpath}: missing symbol(s) {', '.join(unique_symbols)}. "
                            f"{'Add import or verify method name.' if len(unique_symbols) < 3 else 'Check method signatures and import paths.'}",
                })

    if arch_violations:
        for v in arch_violations[:5]:
            rule = v.get("rule", "")
            source = v.get("source_file", "?")
            target = v.get("target_file", "?")
            suggestions.append({
                "category": "architecture",
                "text": f"{rule}: {source} → {target}. "
                        f"Remove the forbidden import; route through the correct layer interface.",
            })

    if cross_stack_errors:
        for err in cross_stack_errors[:5]:
            suggestions.append({
                "category": "cross_stack",
                "text": f"{err.get('message', 'Type mismatch')} in {err.get('file', '?')}. "
                        f"Regenerate types from the OpenAPI spec or update the Java DTO to match.",
            })

    if blast_errors:
        affected = blast_map.get("affected_files", [])
        suggestions.append({
            "category": "blast_radius",
            "text": f"Change affects {len(affected)} files (exceeds limit). "
                    f"Affected: {', '.join(affected[:10])}{'...' if len(affected) > 10 else ''}. "
                    f"Consider splitting into smaller, focused changes.",
        })

    return suggestions
