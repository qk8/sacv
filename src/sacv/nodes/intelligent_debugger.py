"""
nodes/intelligent_debugger.py
==============================
IntelligentDebuggerNode — triggered when Verifier returns diagnostic=AMBIGUOUS.

Instead of sending the Actor back to retry blindly, this node first collects
structured debug observations so the Actor knows exactly what to fix.

Pipeline:
  1. Prune stack trace → filter framework noise → find first user-code frame
  2. Classify error type (pure function — no LLM)
  3. Select debug strategy (pure function — no LLM)
  4. Execute strategy:
     a. BEAN_CREATION_ERROR     → Spring Actuator query (no debug session)
     b. VALIDATION_ERROR/HTTP_400 → Delta debug (binary search on payload)
     c. NULL_REFERENCE, etc.    → JDWP session (Java) or CDP session (Node.js)
     d. REACT_STATE_MISMATCH    → Playwright evaluate
  5. Synthesise root-cause hypothesis (one LLM call with structured data)
  6. Write debug_observations to state → Actor reads and targets the fix

The node ALWAYS routes to Actor afterwards — even partial observations help.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase, DebugObservations
from sacv.nodes._log_parser import prune_stack, frames_to_dict, format_for_actor
from sacv.nodes._debug_strategies import (
    classify_error, get_strategy, ErrorType,
    needs_jdwp, needs_cdp, needs_actuator, needs_delta_debug,
)
from sacv.interfaces.agent_provider import AgentConfig

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_ROOT_CAUSE_SYSTEM = """\
You are a debugging analyst. Given structured debug observations from a live
debug session, write a ONE-PARAGRAPH root cause hypothesis.

Be specific: name the exact file, line, variable, and the value that caused the problem.
Example: "In UserService.java:42, the `user` variable is null because findById() is called
outside an active @Transactional boundary. The JPA session has already closed by the time
the lazy-loaded association is accessed."

Output ONLY the hypothesis paragraph. No lists. No markdown. One paragraph.
"""


def make_intelligent_debugger_node(deps: "NodeDeps"):

    async def intelligent_debugger_node(state: "WorkflowState") -> dict:
        task_id   = state["task_id"]
        module    = state["module_type"]
        verdict   = state["verifier_verdict"] or {}
        cfg       = deps.config.debug

        log.info("debugger.start", task_id=task_id, module=module)

        # ── 1. Collect raw failure text ───────────────────────────────────
        raw_failure = "\n".join(
            f.get("message", "") for f in verdict.get("test_failures", [])
        )

        # ── 2. Prune stack trace ──────────────────────────────────────────
        pruned = prune_stack(
            raw_failure,
            module_type=module,
            user_package=cfg.user_java_package,
            src_root=cfg.user_ts_src_root,
        )
        pruned_dicts = frames_to_dict(pruned)
        log.info("debugger.pruned_stack", frames=len(pruned))

        # ── 3. Classify error + select strategy ───────────────────────────
        error_type = classify_error(raw_failure, module)
        strategy   = get_strategy(error_type)
        log.info("debugger.strategy", error_type=error_type.value,
                 tool=strategy.primary_tool.value)

        # ── 4. Execute strategy ───────────────────────────────────────────
        handle = await deps.sandbox.warm_container()
        try:
            observations = DebugObservations(
                error_type=error_type.value,
                root_cause="",
                breakpoint_hits=[],
                actuator_beans=None,
                actuator_env=None,
                minimal_payload=None,
                playwright_trace_path=None,
                otel_trace=None,
                pruned_stack=pruned_dicts,
            )

            if needs_actuator(strategy):
                observations = await _run_actuator_query(observations, handle, cfg, deps)

            elif needs_delta_debug(strategy):
                payload = _extract_request_payload(state)
                if payload:
                    observations = await _run_delta_debug(
                        observations, payload, state, handle, module, deps
                    )

            elif needs_jdwp(strategy) and pruned:
                observations = await _run_jdwp_session(
                    observations, pruned, strategy, handle, cfg, deps
                )

            elif needs_cdp(strategy) and pruned:
                observations = await _run_cdp_session(
                    observations, pruned, strategy, handle, cfg, deps
                )

            # ── 5. Synthesise root-cause hypothesis (one LLM call) ────────
            hypothesis = await _synthesise_hypothesis(observations, state, deps)
            observations["root_cause"] = hypothesis

            log.info("debugger.complete",
                     error_type=error_type.value,
                     hypothesis=hypothesis[:80])

            return {
                "current_phase":      WorkflowPhase.INTELLIGENT_DEBUGGER.value,
                "debug_observations": observations,
            }
        finally:
            await deps.sandbox.destroy_container(handle)

    return intelligent_debugger_node


# ── Strategy executors ────────────────────────────────────────────────────────

async def _run_actuator_query(
    obs:    DebugObservations,
    handle, cfg, deps,
) -> DebugObservations:
    """Query Spring Boot Actuator for live Bean map and environment."""
    for endpoint, key in [("/beans", "actuator_beans"), ("/env", "actuator_env")]:
        result = await deps.sandbox.exec_in_container(
            handle,
            f"curl -sf {cfg.actuator_base_url}{endpoint} 2>/dev/null || echo '{{}}'",
            timeout=10,
        )
        try:
            obs[key] = json.loads(result.stdout)
        except json.JSONDecodeError:
            obs[key] = {"raw": result.stdout[:1000]}
    log.debug("debugger.actuator_queried")
    return obs


async def _run_delta_debug(
    obs:         DebugObservations,
    payload:     dict,
    state:       "WorkflowState",
    handle,
    module_type: str,
    deps,
) -> DebugObservations:
    """
    Binary delta-debug: find minimal failing subset of request payload.
    Tries both halves at each bisection step.
    """
    fields = list(payload.items())
    if not fields:
        return obs

    minimal = await _delta_minimize(
        fields, state, handle, module_type, deps, depth=0,
    )
    obs["minimal_payload"] = dict(minimal)
    log.info("debugger.delta_debug", original=len(fields), minimal=len(minimal))
    return obs


async def _delta_minimize(
    fields: list,
    state: "WorkflowState",
    handle,
    module_type: str,
    deps,
    depth: int,
) -> list:
    """Recursive delta-debug. Returns smallest failing subset."""
    if depth > 6 or len(fields) <= 1:
        return fields

    mid = len(fields) // 2
    first_half  = fields[:mid]
    second_half = fields[mid:]

    # Try first half
    if first_half:
        result = await _test_payload(
            dict(first_half), state, handle, module_type, deps,
        )
        if result:
            return await _delta_minimize(
                first_half, state, handle, module_type, deps, depth + 1,
            )

    # Try second half
    if second_half:
        result = await _test_payload(
            dict(second_half), state, handle, module_type, deps,
        )
        if result:
            return await _delta_minimize(
                second_half, state, handle, module_type, deps, depth + 1,
            )

    # Error only with both halves together — try adding one field at a time
    for i, field in enumerate(fields):
        candidate = [field]
        rest = fields[:i] + fields[i + 1:]
        # Try candidate + each remaining field
        for other in rest:
            test_set = dict([field, other])
            if await _test_payload(test_set, state, handle, module_type, deps):
                return [field, other]
        if await _test_payload(dict(candidate), state, handle, module_type, deps):
            return candidate

    return fields  # fallback: can't reduce further


async def _test_payload(
    payload: dict, state: "WorkflowState", handle, module_type: str, deps
) -> bool:
    """Run a quick test with the given payload. Returns True if error persists."""
    payload_json = json.dumps(payload)
    endpoint     = _extract_endpoint(state)
    if not endpoint:
        return False
    # Sanitise endpoint to prevent shell injection via URL
    endpoint = endpoint.lstrip()
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    endpoint = re.sub(r"[^a-zA-Z0-9/_\-.]", "", endpoint)

    # Health-check: if no server is running, skip delta debug entirely
    # (BUG-009: fresh debug container has no Spring Boot app running)
    health_cmd = "curl -sf http://localhost:8080/actuator/health 2>/dev/null || echo NOSERVER"
    health = await deps.sandbox.exec_in_container(handle, health_cmd, timeout=5)
    if "NOSERVER" in health.stdout or health.exit_code != 0:
        log.warning("debugger.delta_debug_no_server")
        return False  # can't test; don't assert error present

    cmd = (
        f"echo {shlex.quote(payload_json)} | "
        f"curl -sf -X POST -H 'Content-Type: application/json' "
        f"--data-binary @- http://localhost:8080{endpoint} 2>&1 | head -5"
    )
    result = await deps.sandbox.exec_in_container(handle, cmd, timeout=10)
    return result.exit_code != 0 or "error" in result.stdout.lower()


async def _run_jdwp_session(
    obs:      DebugObservations,
    pruned,
    strategy,
    handle,
    cfg,
    deps,
) -> DebugObservations:
    """
    Start JVM in debug mode, set breakpoints, run the failing test, collect variable state.
    """
    from sacv.adapters.debug.jdwp_client import JdwpClient, BreakpointHitInfo

    first_frame = pruned[0]
    # Class name from "com.example.UserService.findById"
    class_name  = ".".join(first_frame.method.split(".")[:-1])
    target_line = max(1, first_frame.line + strategy.breakpoint_offset)

    # Start test in debug-suspend mode via Docker exec
    start_cmd = (
        "MAVEN_OPTS='-agentlib:jdwp=transport=dt_socket,server=y,"
        f"suspend=y,address=*:{cfg.jdwp_port}' mvn test -q 2>&1 &"
    )
    await deps.sandbox.exec_in_container(handle, start_cmd, timeout=5)

    # Brief pause for JVM to start
    import asyncio
    await asyncio.sleep(2)

    try:
        async with JdwpClient(
            host="localhost", port=deps.sandbox.get_host_jdwp_port()
        ) as jdb:
            await jdb.set_breakpoint_at_line(class_name, target_line)
            await jdb.run()

            hit = await jdb.wait_for_breakpoint_hit(timeout=cfg.debug_timeout_sec)
            if hit:
                variables = await jdb.get_local_variables()
                stack     = await jdb.get_call_stack()

                # Evaluate strategy-specific expressions
                extra_evals = {}
                for expr in strategy.evaluate_expressions:
                    try:
                        extra_evals[expr] = await jdb.evaluate(expr)
                    except Exception:
                        extra_evals[expr] = "<eval_failed>"

                obs["breakpoint_hits"].append({
                    "file":      hit.file,
                    "line":      hit.line,
                    "variables": {v.name: {"value": v.value, "type": v.type}
                                  for v in variables},
                    "call_stack": stack,
                    "thread_id":  hit.thread_name,
                    "extra_evals": extra_evals,
                })

                # Step further if strategy requires
                if strategy.step_type != "none":
                    for _ in range(min(strategy.max_steps, 3)):
                        step_hit = (
                            await jdb.step_over()
                            if strategy.step_type == "step_over"
                            else await jdb.step_into()
                        )
                        if step_hit:
                            step_vars = await jdb.get_local_variables()
                            obs["breakpoint_hits"].append({
                                "file":      step_hit.file,
                                "line":      step_hit.line,
                                "variables": {v.name: {"value": v.value, "type": v.type}
                                              for v in step_vars},
                                "call_stack": await jdb.get_call_stack(),
                                "thread_id":  step_hit.thread_name,
                            })

    except Exception as exc:
        log.warning("debugger.jdwp_session_error", error=str(exc))

    return obs


async def _run_cdp_session(
    obs:      DebugObservations,
    pruned,
    strategy,
    handle,
    cfg,
    deps,
) -> DebugObservations:
    """
    Debug TypeScript/Node.js via Chrome DevTools Protocol.
    """
    from sacv.adapters.debug.cdp_client import CdpClient

    first_frame = pruned[0]
    target_line = max(1, first_frame.line + strategy.breakpoint_offset)
    target_file = first_frame.file

    import asyncio

    # Start Node in inspect-brk mode
    # Use a safe shell pattern: capture glob result, check for empty,
    # and run node with the full path (not dist/dist/...).
    start_cmd = (
        f"ENTRY=$(ls dist/*.js 2>/dev/null | head -1); "
        f"[ -z \"$ENTRY\" ] && echo 'NO_DIST_JS' && exit 1; "
        f"node --inspect-brk=0.0.0.0:{cfg.cdp_port} "
        f"\"$ENTRY\" 2>&1 &"
    )
    result = await deps.sandbox.exec_in_container(handle, start_cmd, timeout=5)
    if "NO_DIST_JS" in result.stdout:
        log.warning("debugger.cdp_no_bundle")
        return obs

    await asyncio.sleep(2)

    try:
        async with CdpClient(
            host="localhost", port=deps.sandbox.get_host_cdp_port()
        ) as cdp:
            await cdp.enable_debugger()
            await cdp.set_breakpoint_by_url(target_file, target_line)
            await cdp.resume()

            paused = await cdp.wait_for_paused(timeout=cfg.debug_timeout_sec)
            if paused:
                variables = await cdp.get_scope_variables_from_paused(paused)

                # Evaluate strategy expressions
                extra_evals = {}
                if paused.call_frame_id:
                    for expr in strategy.evaluate_expressions:
                        try:
                            extra_evals[expr] = await cdp.evaluate_in_frame(
                                expr, paused.call_frame_id
                            )
                        except Exception:
                            extra_evals[expr] = "<eval_failed>"

                obs["breakpoint_hits"].append({
                    "file":       target_file,
                    "line":       target_line,
                    "variables":  variables,
                    "call_stack": [
                        f"{f.function}({f.url}:{f.line})"
                        for f in paused.call_frames
                    ],
                    "thread_id": "main",
                    "extra_evals": extra_evals,
                })

                if strategy.step_type != "none":
                    for _ in range(min(strategy.max_steps, 3)):
                        step_paused = (
                            await cdp.step_over()
                            if strategy.step_type == "step_over"
                            else await cdp.step_into()
                        )
                        if step_paused:
                            step_vars = await cdp.get_scope_variables_from_paused(step_paused)
                            obs["breakpoint_hits"].append({
                                "file":      step_paused.call_frames[0].url if step_paused.call_frames else "",
                                "line":      step_paused.call_frames[0].line if step_paused.call_frames else 0,
                                "variables": step_vars,
                                "call_stack": [f"{f.function}({f.url}:{f.line})"
                                               for f in step_paused.call_frames],
                                "thread_id": "main",
                            })
    except Exception as exc:
        log.warning("debugger.cdp_session_error", error=str(exc))

    return obs


async def _synthesise_hypothesis(
    obs:   DebugObservations,
    state: "WorkflowState",
    deps,
) -> str:
    """Single LLM call to produce a root-cause hypothesis from structured observations."""
    if not obs["breakpoint_hits"] and not obs["actuator_beans"] and not obs["minimal_payload"]:
        return "(debug session produced no observations — falling back to static analysis)"

    obs_summary = {
        "error_type":     obs["error_type"],
        "pruned_stack":   obs["pruned_stack"][:5],
        "breakpoints":    obs["breakpoint_hits"][:3],
        "actuator_beans": bool(obs["actuator_beans"]),
        "minimal_payload": obs["minimal_payload"],
    }

    result = await deps.agent.run_task(
        prompt=(
            f"Task: {state.get('task_description','')}\n\n"
            f"Debug observations:\n{json.dumps(obs_summary, indent=2)}"
        ),
        context={},
        config=AgentConfig(
            role="debug_analyst",
            system_prompt=_ROOT_CAUSE_SYSTEM,
            max_turns=1,
            allowed_tools=[],
        ),
    )
    return result.content.strip()[:500]


# ── Payload extraction helpers ────────────────────────────────────────────────

def _extract_request_payload(state: "WorkflowState") -> dict:
    """Try to extract the request payload from the test failure context."""
    for f in (state.get("verifier_verdict") or {}).get("test_failures", []):
        msg = f.get("message", "")
        if "{" in msg and "}" in msg:
            try:
                start = msg.index("{")
                end   = msg.rindex("}") + 1
                return json.loads(msg[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


def _extract_endpoint(state: "WorkflowState") -> str:
    """Try to guess the API endpoint from the task description or diff."""
    desc = state.get("task_description", "").lower()
    for pattern in ["/api/", "/v1/", "/v2/"]:
        if pattern in desc:
            idx = desc.index(pattern)
            return desc[idx:idx + 40].split()[0].rstrip(".")
    return ""
