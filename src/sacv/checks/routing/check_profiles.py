"""
checks/routing/check_profiles.py
=================================
Module-type × profile → active check dimensions.

Refactoring addition (approach 5):
  Cross-domain pipeline routing: when blast_radius_map.schema_impact is
  non-empty, the Verifier adds the "cross_domain_api" check dimension
  regardless of the primary module type.  This is detected at runtime
  by the Verifier node; this file defines the static base matrix.

Refactoring addition (approaches 9, 10):
  "arch_check" dimension added to all module types at standard/full profiles.
  This maps to the Preflight node's StructuralCheck (dependency-cruiser / ArchUnit).
  NOTE: arch_check runs in the Preflight node (before Critics), not in the
  Verifier. The dimension here is for documentation and tooling discovery only.
"""
from __future__ import annotations

# Module type and profile string constants
# (mirrors state.py enums but kept as plain strings for zero-import use)
_BACKEND_DOMAIN   = "backend-domain"
_BACKEND_API      = "backend-api"
_FRONTEND_FEATURE = "frontend-feature"
_FRONTEND_DATA    = "frontend-data"
_INFRASTRUCTURE   = "infrastructure"
_CROSS_CUTTING    = "cross-cutting"

_MINIMAL  = "minimal"
_STANDARD = "standard"
_FULL     = "full"

CHECK_MATRIX: dict[str, dict[str, list[str]]] = {
    _BACKEND_DOMAIN: {
        _MINIMAL:  ["test_execution", "code_quality"],
        _STANDARD: ["test_execution", "code_quality", "spec_compliance",
                    "security", "test_design", "arch_check"],
        _FULL:     ["test_execution", "code_quality", "spec_compliance",
                    "security", "test_design", "arch_check",
                    "integration", "resilience", "performance"],
    },
    _BACKEND_API: {
        _MINIMAL:  ["test_execution", "security"],
        _STANDARD: ["test_execution", "code_quality", "security",
                    "spec_compliance", "integration", "arch_check"],
        _FULL:     ["test_execution", "code_quality", "security",
                    "spec_compliance", "integration", "arch_check",
                    "resilience", "performance"],
    },
    _FRONTEND_FEATURE: {
        _MINIMAL:  ["test_execution", "code_quality"],
        _STANDARD: ["test_execution", "code_quality", "spec_compliance",
                    "test_design", "arch_check", "visual_diff"],
        _FULL:     ["test_execution", "code_quality", "spec_compliance",
                    "test_design", "arch_check", "visual_diff", "performance"],
    },
    _FRONTEND_DATA: {
        _MINIMAL:  ["test_execution"],
        _STANDARD: ["test_execution", "code_quality",
                    "integration", "arch_check"],
        _FULL:     ["test_execution", "code_quality",
                    "integration", "arch_check", "performance"],
    },
    _INFRASTRUCTURE: {
        _MINIMAL:  ["test_execution", "security"],
        _STANDARD: ["test_execution", "security",
                    "integration", "arch_check"],
        _FULL:     ["test_execution", "security",
                    "integration", "arch_check", "resilience"],
    },
    _CROSS_CUTTING: {
        _MINIMAL:  ["test_execution", "code_quality"],
        _STANDARD: ["test_execution", "code_quality", "spec_compliance",
                    "security", "arch_check"],
        _FULL:     ["test_execution", "code_quality", "spec_compliance",
                    "security", "arch_check", "integration",
                    "resilience", "performance"],
    },
}

# Dimensions always added when blast_radius_map.schema_impact is non-empty
# (approach 5 — cross-domain pipeline routing)
SCHEMA_IMPACT_EXTRA_CHECKS: list[str] = ["cross_domain_api"]


def get_active_checks(module_type: str, profile: str) -> list[str]:
    """
    Pure function. No I/O. Raises KeyError on unknown inputs.

    Returns the ordered list of check dimensions for the given
    module_type × profile combination.
    """
    return CHECK_MATRIX[module_type][profile]


def get_active_checks_with_blast_radius(
    module_type:  str,
    profile:      str,
    schema_impact: list[str],
) -> list[str]:
    """
    Pure function. No I/O.

    Extends the base check list with cross-domain pipeline checks
    when the blast_radius_map reports schema-level impact (approach 5).
    """
    base = get_active_checks(module_type, profile)
    if schema_impact:
        extras = [c for c in SCHEMA_IMPACT_EXTRA_CHECKS if c not in base]
        return base + extras
    return base
