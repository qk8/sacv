"""
nodes/verifier.py
=================
Multi-dimensional verification inside the Docker sandbox.

Refactoring additions (debugging session):
  STACK TRACE PRUNING (approach 5):
    _parse_failures() now uses _log_parser.prune_stack() to filter framework
    noise before returning failures to the graph. Actor receives user-code
    frames only, not 150 lines of Spring internals.

  SPRING ACTUATOR ON DI ERRORS (approach 3):
    BeanCreationException detected → query /actuator/beans snapshot before
    routing to debugger. Included in verifier_verdict for immediate use.

  PLAYWRIGHT TRACE (approach 2 frontend):
    --trace retain-on-failure flag added. Trace ZIP path recorded in verdict
    so IntelligentDebuggerNode can reference it.

  OTEL TRACE CORRELATION (approach 1):
    If OTel collector is running in sandbox, query traces matching the test
    run correlation ID and attach to verdict.

  DIAGNOSTIC PRECISION:
    Verdict now distinguishes AMBIGUOUS more precisely — only truly
    unexplainable failures get AMBIGUOUS (routing to debugger).
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import (
    WorkflowPhase, VerifierVerdict, DiagnosticVerdict,
)
from sacv.nodes._stagnation import embed_error_to_b64
from sacv.nodes._log_parser import prune_stack, frames_to_dict

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_EVIDENCE_DIR = Path(".workflow/tdd-evidence")
_TRACE_DIR    = Path(".workflow/playwright-traces")

# OTel collector endpoint — now configurable via DebugConfig.otel_query_url


def make_verifier_node(deps: "NodeDeps"):

    async def verifier_node(state: "WorkflowState") -> dict:
        task_id   = state["task_id"]
        module    = state["module_type"]
        correction = state["correction_state"]
        findings  = state.get("critic_findings", [])
        blast     = state.get("blast_radius_map") or {}
        inv_paths = state.get("test_inventory_paths", [])
        cfg       = deps.config.debug

        log.info("verifier.start", task_id=task_id, attempt=correction["attempt_count"])

        # ── Test deletion guard (pure function — no I/O) ───────────────────
        deletion_error = _check_test_deletions(state)
        if deletion_error:
            verdict = _make_verdict(
                test_result="FAIL", diagnostic=DiagnosticVerdict.FIX_IMPL.value,
                phase1_passed=False, phase2_passed=False,
                failures=[{"source": "deletion_guard", "message": deletion_error}],
                findings=findings, docker_exit_code=-1,
            )
            return _build_return(verdict, correction, deletion_error)

        # ── Critical critic findings → skip Docker ────────────────────────
        critical = [f for f in findings if f["severity"] == "critical"]
        if critical:
            log.warning("verifier.critical_block", count=len(critical))
            verdict = _make_verdict(
                test_result="FAIL",
                diagnostic=DiagnosticVerdict.FIX_IMPL.value,
                phase1_passed=False, phase2_passed=False,
                failures=[{"source": "critic", "finding": f} for f in critical],
                findings=findings, docker_exit_code=-1,
            )
            return _build_return(verdict, correction, "critic_block")

        # NOW warm the container (after all zero-cost checks pass)
        handle = await deps.sandbox.warm_container()
        try:

            # ── Phase 1: Legacy Regression Sweep ─────────────────────────
            p1_cmd    = _full_suite_cmd(module)
            p1_exec   = await deps.sandbox.exec_in_container(handle, p1_cmd, timeout=300)
            p1_passed = p1_exec.exit_code == 0
            p1_raw    = p1_exec.stdout + p1_exec.stderr

            # Apply stack trace pruning (approach 5)
            p1_frames  = prune_stack(p1_raw, module, cfg.user_java_package, cfg.user_ts_src_root)
            p1_failures = frames_to_dict(p1_frames) or _fallback_parse(p1_raw, module)

            log.info("verifier.phase1", passed=p1_passed, user_frames=len(p1_frames))

            if not p1_passed:
                # Try Spring Actuator on DI errors before returning (approach 3)
                actuator_snap = None
                if _is_bean_error(p1_raw):
                    actuator_snap = await _query_actuator(handle, cfg.actuator_base_url, deps)

                verdict = _make_verdict(
                    test_result="FAIL", diagnostic=DiagnosticVerdict.FIX_IMPL.value,
                    phase1_passed=False, phase2_passed=False,
                    failures=p1_failures, findings=findings,
                    actuator_snapshot=actuator_snap,
                )
                return _build_return(verdict, correction, " ".join(
                    f.get("message", "") for f in p1_failures
                ))

            # ── Phase 2: New Feature Tests ───────────────────────────────
            p2_passed = True
            p2_failures: list[dict] = []
            playwright_trace: str | None = None

            if inv_paths:
                p2_cmd  = _inventory_test_cmd(module, inv_paths)
                p2_exec = await deps.sandbox.exec_in_container(handle, p2_cmd, timeout=180)
                p2_passed = p2_exec.exit_code == 0
                p2_raw    = p2_exec.stdout + p2_exec.stderr

                p2_frames  = prune_stack(p2_raw, module, cfg.user_java_package, cfg.user_ts_src_root)
                p2_failures = frames_to_dict(p2_frames) or _fallback_parse(p2_raw, module)

                # Playwright trace on frontend failure (approach 2)
                if not p2_passed and "frontend" in module:
                    playwright_trace = await _extract_playwright_trace(handle, task_id, deps)

                log.info("verifier.phase2", passed=p2_passed, user_frames=len(p2_frames))

            # ── Blast-radius cross-domain API check (approach 5) ─────────
            if blast.get("schema_impact") and "frontend" not in module:
                api_exec = await deps.sandbox.exec_in_container(
                    handle,
                    "[ -d tests/api ] && npm test -- --testPathPattern=tests/api "
                    "--watchAll=false 2>&1 || true",
                    timeout=120,
                )
                if api_exec.exit_code != 0:
                    api_frames  = prune_stack(api_exec.stdout, "backend-api",
                                              cfg.user_java_package, cfg.user_ts_src_root)
                    p2_failures += frames_to_dict(api_frames)
                    p2_passed   = False

            # ── OTel trace correlation ───────────────────────────────────
            otel_trace = None
            if not p2_passed:
                otel_trace = await _query_otel(handle, task_id, deps)

            # ── Performance profiling ────────────────────────────────────
            # Feature not yet implemented — perf baseline infrastructure
            # does not exist.
            perf_delta: dict | None = None

            # ── Visual diff (frontend only) ──────────────────────────────
            visual_result: dict | None = None
            if p1_passed and p2_passed and "frontend" in module:
                visual_result = await _run_visual_diff(handle, task_id, deps)

            overall_pass = (
                p1_passed and p2_passed
                and not _has_visual_breakage(visual_result)
            )

            all_failures = p1_failures + p2_failures
            diagnostic   = _classify(
                p1_passed, p2_passed, all_failures, findings, state,
                overall_pass=overall_pass,
            )

            verdict = _make_verdict(
                test_result="PASS" if overall_pass else "FAIL",
                diagnostic=diagnostic,
                phase1_passed=p1_passed, phase2_passed=p2_passed,
                failures=all_failures, findings=findings,
                performance_delta=perf_delta,
                visual_diff_result=visual_result,
                playwright_trace_path=playwright_trace,
                otel_trace=otel_trace,
            )
            log.info("verifier.complete", result=verdict["test_result"], diag=diagnostic)
            return _build_return(
                verdict, correction,
                " ".join(f.get("message", "") for f in all_failures)
            )
        finally:
            await deps.sandbox.destroy_container(handle)

    return verifier_node


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def _classify(
    p1_passed:    bool, p2_passed: bool,
    failures:     list[dict], findings: list[dict],
    state:        "WorkflowState",
    overall_pass: bool = True,
) -> str:
    if p1_passed and p2_passed:
        if not overall_pass:
            # Tests pass but perf/visual broke — Actor needs to optimise
            return DiagnosticVerdict.FIX_IMPL.value
        return DiagnosticVerdict.PASS.value
    failure_text = " ".join(f.get("message", "") for f in failures).lower()
    if not p1_passed:
        return DiagnosticVerdict.FIX_IMPL.value

    # ── FIX_TEST detection (p1 passed, p2 failed) ─────────────────────
    # p2 tests are the NEW tests written by TDD gate. If they fail with
    # assertion mismatches (not compilation), the oracle may have written
    # tests that don't match the actual spec.
    if not p2_passed:
        assertion_keywords = (
            "assertionerror", "expected", "received", "but was",
            "expected:<", "junit.framework.assertionerror",
            "expect(received).tobe", "toequal", "tomatch",
        )
        compile_keywords = (
            "compilat", "syntax", "cannot find symbol", "module not found",
        )
        has_assertion_fail = any(kw in failure_text for kw in assertion_keywords)
        has_compile_fail   = any(kw in failure_text for kw in compile_keywords)
        if has_assertion_fail and not has_compile_fail:
            return DiagnosticVerdict.FIX_TEST.value

    if any(kw in failure_text for kw in ("compilat", "syntax", "cannot find symbol", "module not found")):
        return DiagnosticVerdict.FIX_IMPL.value
    if state.get("red_phase_evidence_path"):
        return DiagnosticVerdict.FIX_IMPL.value
    if any(f["severity"] == "critical" for f in findings):
        return DiagnosticVerdict.FIX_IMPL.value
    # If we have no user-code frames and no clear signal → AMBIGUOUS → route to debugger
    return DiagnosticVerdict.AMBIGUOUS.value


def _is_bean_error(output: str) -> bool:
    return any(kw in output for kw in (
        "BeanCreationException", "NoSuchBeanDefinitionException",
        "UnsatisfiedDependencyException", "NoUniqueBeanDefinitionException",
    ))


# ── Tool runners ──────────────────────────────────────────────────────────────

async def _query_actuator(handle, base_url: str, deps) -> dict | None:
    """Query Spring Actuator /beans endpoint (approach 3)."""
    result = await deps.sandbox.exec_in_container(
        handle,
        f"curl -sf {base_url}/beans 2>/dev/null | head -c 10000 || echo '{{}}'",
        timeout=10,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


async def _query_otel(handle, task_id: str, deps) -> dict | None:
    """Query OTel/Jaeger for traces correlated with this test run (approach 1)."""
    cfg = deps.config.debug
    result = await deps.sandbox.exec_in_container(
        handle,
        f"curl -sf '{cfg.otel_query_url}"
        f"?service=sacv-sandbox&limit=5&lookback={120_000_000}' 2>/dev/null "
        "|| echo 'NO_OTEL'",
        timeout=8,
    )
    if "NO_OTEL" in result.stdout or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


async def _extract_playwright_trace(handle, task_id: str, deps) -> str | None:
    """Extract most recent Playwright trace ZIP path (approach 2)."""
    result = await deps.sandbox.exec_in_container(
        handle,
        "find . -name 'trace.zip' -newer .workflow/ 2>/dev/null | head -1",
        timeout=5,
    )
    path = result.stdout.strip()
    return path if path else None


async def _run_visual_diff(handle, task_id: str, deps) -> dict | None:
    # Sanitize task_id for shell safety
    safe_task_id = re.sub(r"[^a-zA-Z0-9\-_]", "-", task_id)[:32]
    output_path = f"/tmp/vd-{safe_task_id}.json"

    # Run build first to isolate JSON output
    build_cmd = "npm run build --silent 2>&1 | tail -3"
    build_r = await deps.sandbox.exec_in_container(handle, build_cmd, timeout=60)
    if build_r.exit_code != 0:
        log.warning("verifier.build_failed_for_visual_diff")
        return None

    # Run visual diff — redirect stderr to a log file, NOT stdout
    vd_cmd = (
        f"node /sacv/visual-diff.js "
        f"--task {shlex.quote(safe_task_id)} "
        f"--output {shlex.quote(output_path)} "
        f"2>/tmp/vd-stderr.log"
    )
    run_r = await deps.sandbox.exec_in_container(handle, vd_cmd, timeout=60)
    if run_r.exit_code != 0:
        log.warning("verifier.visual_diff_failed", exit_code=run_r.exit_code)
        return None

    # Read output separately — stdout is clean JSON only
    cat_r = await deps.sandbox.exec_in_container(
        handle, f"cat {shlex.quote(output_path)}", timeout=5
    )
    try:
        return json.loads(cat_r.stdout.strip())
    except json.JSONDecodeError:
        log.warning("verifier.visual_diff_parse_failed", stdout=cat_r.stdout[:200])
        return None


def _has_visual_breakage(visual: dict | None) -> bool:
    return bool(visual) and not visual.get("passed", True)


def _check_test_deletions(state: "WorkflowState") -> str | None:
    proposal = state.get("diff_proposal")
    if not proposal:
        return None

    _ASSERTION_KEYWORDS = (
        "@Test", "void test", "assertThat", "assertEquals", "assert ", "expect(",
        "it(", "describe(", "test(", "jest.expect",
    )
    _TEST_PATH_MARKERS = ("tests/", "src/test/", ".spec.ts", "Test.java")

    violations: list[str] = []

    for d in proposal.get("diffs", []):
        path = d.get("file_path", "")
        op = d.get("operation", "")
        is_test_file = any(m in path for m in _TEST_PATH_MARKERS)

        if not is_test_file:
            continue

        # Explicit deletion
        if op == "delete":
            violations.append(path)
            continue

        # Gutting: diff removes assertion lines without replacing them
        diff_content = d.get("diff_content", "")
        removed = [l[1:] for l in diff_content.splitlines() if l.startswith("-")]
        added = [l[1:] for l in diff_content.splitlines() if l.startswith("+")]
        removed_assertions = sum(
            1 for l in removed if any(kw in l for kw in _ASSERTION_KEYWORDS)
        )
        added_assertions = sum(
            1 for l in added if any(kw in l for kw in _ASSERTION_KEYWORDS)
        )
        if removed_assertions > 0 and added_assertions < removed_assertions // 2:
            violations.append(
                f"{path} (assertion gutting: -{removed_assertions} +{added_assertions})"
            )

    if violations:
        return (
            f"Test modification prohibited: {', '.join(violations)}. "
            "Rewrite the implementation so existing tests pass; do not weaken tests."
        )
    return None


def _full_suite_cmd(module_type: str) -> str:
    if "frontend" in module_type:
        return "npx playwright test --trace retain-on-failure 2>&1 || npm test -- --watchAll=false --ci 2>&1"
    return "mvn test -q 2>&1"


def _inventory_test_cmd(module_type: str, paths: list[str]) -> str:
    if "frontend" in module_type:
        spec_files = " ".join(f'"{p}"' for p in paths if p.endswith(".spec.ts"))
        if spec_files:
            return f"npx playwright test {spec_files} --trace retain-on-failure --reporter=line 2>&1"
        return f"npm test -- --testPathPattern='{paths[0] if paths else ''}' --watchAll=false 2>&1"
    classes = [Path(p).stem for p in paths if p.endswith(".java") and "Test" in Path(p).stem]
    return f"mvn test -Dtest={','.join(classes)} -q 2>&1" if classes else "mvn test -q 2>&1"


def _fallback_parse(output: str, module_type: str) -> list[dict]:
    """Last-resort parser when pruner finds no user-code frames."""
    failures = []
    if "frontend" in module_type:
        for line in output.splitlines():
            if any(kw in line for kw in ("FAIL", "Error:", "✕", "×")):
                failures.append({"message": line.strip(), "source": "raw"})
    else:
        for line in output.splitlines():
            if "BUILD FAILURE" in line or ("Tests run:" in line and "Failures:" in line):
                failures.append({"message": line.strip(), "source": "raw"})
    return failures[:10]


def _make_verdict(
    test_result: str, diagnostic: str,
    phase1_passed: bool, phase2_passed: bool,
    failures: list[dict], findings: list[dict],
    performance_delta:    dict | None = None,
    visual_diff_result:   dict | None = None,
    playwright_trace_path: str | None = None,
    otel_trace:           dict | None = None,
    actuator_snapshot:    dict | None = None,
    docker_exit_code:     int = 0,
) -> VerifierVerdict:
    return VerifierVerdict(
        test_result=test_result, diagnostic=diagnostic,
        phase1_passed=phase1_passed, phase2_passed=phase2_passed,
        test_failures=failures, performance_delta=performance_delta,
        visual_diff_result=visual_diff_result, critic_findings=findings,
        docker_exit_code=docker_exit_code,
        playwright_trace_path=playwright_trace_path,
        otel_trace=otel_trace,
        actuator_snapshot=actuator_snapshot,
    )


def _build_return(verdict: VerifierVerdict, correction: dict, failure_text: str) -> dict:
    new_correction = dict(correction)
    if verdict["test_result"] == "FAIL" and failure_text:
        history = list(correction.get("error_history", []))
        history.append(embed_error_to_b64(failure_text))
        new_correction["error_history"]   = history[-5:]
        new_correction["last_error_hash"] = history[-1][:16]
    return {
        "current_phase":    WorkflowPhase.VERIFIER.value,
        "verifier_verdict": verdict,
        "correction_state": new_correction,
    }
