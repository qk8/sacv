# SACV Workflow — Debugging Session Notes

## Overview

This session adds step-through debugging capability to the SACV workflow,
covering both Java/Spring Boot (via JDWP) and TypeScript/Node.js (via CDP).

---

## New Capability: IntelligentDebuggerNode

### When it triggers
`route_after_verifier` now routes to `intelligent_debugger` when the
Verifier returns `diagnostic = AMBIGUOUS` — meaning the failure cannot be
explained from logs alone. Previously, AMBIGUOUS caused a blind Actor retry.

Updated routing table:

| Verdict   | Attempt | Route (before)         | Route (after)               |
|---|---|---|---|
| PASS      | any     | memory_consolidation   | memory_consolidation        |
| FIX_IMPL  | 1       | actor                  | actor                       |
| FIX_IMPL  | 2       | speculative_branch     | speculative_branch          |
| AMBIGUOUS | **≤ 1** | **actor (blind)**      | **intelligent_debugger**    |
| AMBIGUOUS | **≥ 2** | **speculative_branch** | **speculative_branch** (ARCH-004) |
| FIX_IMPL  | ≥ MAX   | hitl_escalation        | hitl_escalation             |
| any       | low confidence | hitl_escalation   | hitl_escalation             |

### What the debugger does

1. **Stack trace pruning** — filter framework internals → user-code frames only
2. **Error classification** — pure function maps error text to `ErrorType`
3. **Strategy selection** — pure function maps `ErrorType` to `DebugStrategy`
4. **Execute strategy** (4 paths):
   - `BEAN_CREATION_ERROR` → query `/actuator/beans` (no debug session)
   - `VALIDATION_ERROR` / `HTTP_400` → delta debug (binary search on payload)
   - Java errors → JDWP session via `JdwpClient` (JDB subprocess)
   - TypeScript errors → CDP session via `CdpClient` (WebSocket)
5. **Root-cause hypothesis** — one LLM call with structured observations
6. **Write `debug_observations` to state** — Actor reads and targets the fix

### Actor integration

After the debug session, `debug_observations` is injected into the Actor's
system prompt under "DEBUG OBSERVATIONS (live variable state from debugger session)".
The Actor sees:
- Exact variable values at the breakpoint hit
- Call stack at that moment
- Root-cause hypothesis

---

## New Files

| File | Purpose |
|---|---|
| `nodes/_log_parser.py` | Stack trace pruning — pure functions |
| `nodes/_debug_strategies.py` | Error classification + strategy selection — pure |
| `nodes/intelligent_debugger.py` | Main debugger orchestration node |
| `adapters/debug/jdwp_client.py` | Java JDB-based step-through debugger |
| `adapters/debug/cdp_client.py` | Node.js CDP WebSocket debugger |
| `docker/sandbox-start.sh` | Sandbox startup with Jaeger |

## Modified Files

| File | Change |
|---|---|
| `orchestration/state.py` | Added `DebugObservations`, `BreakpointHit`, `debug_observations` field |
| `orchestration/config.py` | Added `DebugConfig` (ports, packages, timeouts) |
| `orchestration/edges.py` | `AMBIGUOUS → intelligent_debugger` routing |
| `orchestration/graph.py` | Added `intelligent_debugger` node, `intelligent_debugger → actor` edge |
| `nodes/bootstrap.py` | Resets `debug_observations: None` |
| `nodes/actor.py` | Reads `debug_observations`, adds `@ai-agent` comment instruction |
| `nodes/verifier.py` | Stack trace pruning, Spring Actuator, Playwright trace, OTel |
| `nodes/preflight_node.py` | Cross-stack type check (monorepo mode, approach 3A) |
| `Dockerfile.sandbox` | java-debug, OTel agent, Jaeger, source-map-support |
| `docker-compose.yml` | Ports 5005 (JDWP), 9229 (CDP), 16686 (Jaeger) |

## New Tests

| File | Coverage |
|---|---|
| `tests/unit/test_debug_strategies.py` | Error classification × 15 cases |
| `tests/unit/test_log_parser.py` | Stack trace pruning × 18 cases |
| `tests/unit/test_debug_routing.py` | AMBIGUOUS routing × 9 cases |
| `tests/integration/test_intelligent_debugger.py` | Full debug node × 10 cases |

---

## Docker Sandbox Debug Ports

```
Port 5005  — JDWP (Java Debug Wire Protocol)
             JVM starts with: -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:5005
             JdwpClient connects: localhost:5005

Port 9229  — V8 Inspector / CDP (Chrome DevTools Protocol)
             Node starts with: node --inspect-brk=0.0.0.0:9229 dist/app.js
             CdpClient connects: ws://localhost:9229

Port 16686 — Jaeger query UI + HTTP API
             OTel traces queried at: http://localhost:16686/api/traces

Port 4317  — OTel GRPC collector (receives spans from Spring Boot agent)
Port 4318  — OTel HTTP collector (receives spans from frontend)
```

## DebugConfig fields (workflow-config.json)

```json
{
  "debug": {
    "user_java_package":  "com.yourcompany",
    "user_ts_src_root":   "src",
    "jdwp_port":          5005,
    "cdp_port":           9229,
    "debug_timeout_sec":  30,
    "max_debug_steps":    10,
    "actuator_base_url":  "http://localhost:8080/actuator",
    "openapi_spec_path":  "contracts/openapi/api.yaml"
  }
}
```

**Important:** Set `user_java_package` to your actual base package (e.g. `com.yourcompany`).
The log parser uses this to filter Spring/Hibernate noise from stack traces.

---

## Error Type → Debug Strategy Reference

| Error Type | Primary Tool | Key Action |
|---|---|---|
| NULL_REFERENCE | JDWP | Break 1 line before NPE, dump all variables |
| CONCURRENT_MODIFICATION | JDWP | Break at mutation point, inspect thread |
| OPTIMISTIC_LOCK | JDWP | Evaluate `entity.getVersion()` vs DB |
| BEAN_CREATION_ERROR | Spring Actuator | Query `/actuator/beans` |
| VALIDATION_ERROR | Delta Debug | Binary search on DTO fields |
| ASYNC_RACE_CONDITION | CDP | Step into Promise chain |
| ASYNC_PROMISE_UNHANDLED | CDP | Find rejection without .catch() |
| REACT_STATE_MISMATCH | Playwright | Evaluate React component tree |
| CLASS_CAST | JDWP | Evaluate `obj.getClass().getName()` |
| HTTP_400 | Delta Debug | Binary search on request payload |
| UNKNOWN | JDWP | General variable dump at first user frame |
