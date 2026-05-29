"""
nodes/critics/consistency.py
============================
ConsistencyCritic: local neighbourhood idiom-matching.

In BROWNFIELD mode, this critic is the most important: it ensures
the diff follows the patterns already established in the codebase
rather than introducing a foreign style.

In GREENFIELD mode, it checks for internal consistency within the
diff itself (e.g. if the diff introduces a naming convention, all
new files must follow it).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from sacv.nodes.critics.base import _run_critic
if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

_CONSISTENCY_RULES_BROWNFIELD = """
Brownfield mode — match existing patterns:
- Check that new classes follow the naming patterns of neighbouring classes.
- Check that exception handling follows the existing strategy (e.g. if the
  codebase uses a GlobalExceptionHandler, new exceptions must use it).
- Check that dependency injection style matches (constructor injection vs field).
- Check that logging framework usage matches (SLF4J, Logback, Log4j, etc.).
- Check that new API endpoints follow the existing URL versioning pattern.
- Flag any new third-party dependency not already in pom.xml / package.json.
"""

_CONSISTENCY_RULES_GREENFIELD = """
Greenfield mode — internal consistency within this diff:
- All new classes must follow the naming convention established in this diff.
- All new interfaces must follow the same pattern.
- Ensure logging is used consistently (not a mix of System.out and SLF4J).
- Ensure error handling is consistent across all new files.
"""


def make_consistency_critic_node(deps: "NodeDeps"):
    async def consistency_critic_node(state: "WorkflowState") -> dict:
        mode  = state.get("project_mode", "greenfield")
        rules = (
            _CONSISTENCY_RULES_BROWNFIELD
            if mode == "brownfield"
            else _CONSISTENCY_RULES_GREENFIELD
        )
        findings = await _run_critic(
            role="senior developer enforcing codebase consistency",
            critic_name="consistency",
            extra_rules=rules,
            state=state,
            deps=deps,
        )
        return {"critic_findings": findings}
    return consistency_critic_node
