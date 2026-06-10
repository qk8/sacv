"""
check_profiles.py
=================
Defines which checks (LSP, arch, cross-stack, visual, perf) are active
per module_type and profile_name combination.

CHECK_MATRIX[module_type][profile_name] -> list[CheckSpec]

Wired into preflight_node.py via get_checks().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CheckName = Literal[
    "lsp", "arch", "cross_stack", "visual_diff", "perf", "blast_radius",
]


@dataclass(frozen=True)
class CheckSpec:
    name:     CheckName
    timeout:  int    = 60    # seconds
    required: bool   = True  # if True, failure blocks critics


# Default check matrix — adjust per team conventions
_MATRIX: dict[str, dict[str, list[CheckSpec]]] = {
    "backend-domain": {
        "standard": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
        ],
        "full": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
            CheckSpec("cross_stack", timeout=90, required=False),
            CheckSpec("perf", timeout=120, required=False),
        ],
    },
    "backend-api": {
        "standard": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
            CheckSpec("blast_radius", timeout=30, required=False),
        ],
        "full": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
            CheckSpec("blast_radius", timeout=30),
            CheckSpec("cross_stack", timeout=90),
        ],
    },
    "frontend-feature": {
        "standard": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
        ],
        "full": [
            CheckSpec("lsp", timeout=60),
            CheckSpec("arch", timeout=30),
            CheckSpec("visual_diff", timeout=90, required=False),
        ],
    },
    "frontend-data": {
        "standard": [CheckSpec("lsp"), CheckSpec("arch")],
        "full": [
            CheckSpec("lsp"),
            CheckSpec("arch"),
            CheckSpec("cross_stack", timeout=90),
        ],
    },
    "infrastructure": {
        "standard": [CheckSpec("lsp")],
        "full":     [CheckSpec("lsp")],
    },
    "cross-cutting": {
        "standard": [CheckSpec("lsp")],
        "full":     [CheckSpec("lsp"), CheckSpec("arch")],
    },
}

_DEFAULT_PROFILE = [CheckSpec("lsp", timeout=60)]


def get_checks(
    module_type:   str,
    profile:       str  = "standard",
    monorepo_mode: bool = False,
) -> list[CheckSpec]:
    """Return active checks for the given module_type and profile name.

    When ``monorepo_mode=False``, the ``cross_stack`` check is removed
    from the result since cross-stack type checks are only relevant
    for monorepo projects.
    """
    checks = _MATRIX.get(module_type, {}).get(profile, _DEFAULT_PROFILE)
    # Preserve _DEFAULT_PROFILE identity when no filtering needed
    if checks is _DEFAULT_PROFILE or monorepo_mode:
        return checks
    return [c for c in checks if c.name != "cross_stack"]


def get_active_checks_with_blast_radius(
    module_type: str,
    has_schema_impact: bool = False,
) -> list[CheckSpec]:
    """
    Extended check list for blast-radius scenarios (approach 5 from
    REFACTORING_NOTES). When schema_impact is non-empty, adds cross_stack
    check for backend modules.
    """
    checks = list(get_checks(module_type, "standard"))
    if has_schema_impact and "frontend" not in module_type:
        checks.append(CheckSpec("cross_stack", timeout=90, required=False))
    return checks
