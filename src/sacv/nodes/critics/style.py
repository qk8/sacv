"""
nodes/critics/style.py
======================
StyleCritic: naming conventions, DDD ubiquitous language, cyclomatic complexity.
Language-specific rules for Java (Spring Boot) and TypeScript (React).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from sacv.nodes.critics.base import _run_critic
if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

_STYLE_RULES = """
DDD / Clean Architecture:
- Domain entities must not import from infrastructure or application layers.
- Repository interfaces belong in the domain layer; implementations in infra.
- Use-case classes must not contain business logic — delegate to domain services.
- Ubiquitous language: class/method names must reflect the domain vocabulary.

Java rules:
- Service classes must end in 'Service'; repositories in 'Repository'.
- No public fields on domain entities — use private + getter/setter or records.
- Method cyclomatic complexity > 10 is a warning; > 15 is critical.
- Avoid checked exceptions leaking past the application layer boundary.

TypeScript rules:
- React components: PascalCase; hooks: camelCase prefixed with 'use'.
- No 'any' type without a comment explaining why.
- Props interfaces must be named '<ComponentName>Props'.
- Avoid nested ternaries deeper than 2 levels.

General:
- No TODO/FIXME comments in production-facing code paths.
- Magic numbers must be extracted to named constants.
"""


def make_style_critic_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":
    async def style_critic_node(state: "WorkflowState") -> dict[str, object]:
        findings, new_cost = await _run_critic(
            role="principal engineer enforcing DDD and Clean Architecture",
            critic_name="style",
            extra_rules=_STYLE_RULES,
            state=state,
            deps=deps,
        )
        return {"critic_findings": findings, "cumulative_cost_dollars": new_cost}
    return style_critic_node
