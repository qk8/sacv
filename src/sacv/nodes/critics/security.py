"""
nodes/critics/security.py
=========================
SecurityCritic: OWASP Top 10, secrets detection, JWT/session security.
Covers both Java (Spring Security) and TypeScript (Next.js / Express) rules.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from sacv.nodes.critics.base import _run_critic
if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

_SECURITY_RULES = """
Java / Spring Security rules:
- Detect SQL injection: string concatenation in queries → use JPA/named params.
- Detect hardcoded secrets: API keys, passwords, tokens in source.
- Detect missing @PreAuthorize / @Secured on controller endpoints.
- Detect disabled CSRF protection without explicit justification.
- Detect BCryptPasswordEncoder replaced with weaker hash.
- Detect missing input validation (@Valid, @NotNull) on @RequestBody params.

TypeScript / Node rules:
- Detect eval(), Function() with dynamic strings.
- Detect missing helmet() middleware.
- Detect jwt.verify() without algorithm restriction.
- Detect process.env secrets logged or returned in responses.
- Detect CORS configured with origin: '*' in production contexts.
- Detect missing rate-limiting on auth endpoints.

Common:
- Detect DEBUG flags left enabled.
- Detect commented-out authentication checks.
"""


def make_security_critic_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":
    async def security_critic_node(state: "WorkflowState") -> dict[str, object]:
        findings, new_cost = await _run_critic(
            role="security engineer specialising in OWASP Top 10",
            critic_name="security",
            extra_rules=_SECURITY_RULES,
            state=state,
            deps=deps,
        )
        return {"critic_findings": findings, "cumulative_cost_dollars": new_cost}
    return security_critic_node
