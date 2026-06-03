"""
tests/unit/test_check_profiles.py
===================================
Unit tests for check_profiles.py — the check matrix that defines which
preflight checks (LSP, arch, cross-stack, visual, perf, blast_radius)
are active per module_type and profile_name combination.

Tests cover:
1. get_checks — backend-domain standard profile
2. get_checks — backend-domain full profile
3. get_checks — backend-api standard profile (includes blast_radius)
4. get_checks — backend-api full profile (adds cross_stack)
5. get_checks — frontend-feature standard profile
6. get_checks — frontend-feature full profile (adds visual_diff)
7. get_checks — frontend-data profiles
8. get_checks — infrastructure profiles
9. get_checks — cross-cutting profiles
10. get_checks — unknown module_type falls back to default (lsp only)
11. get_checks — unknown profile falls back to default
12. get_checks — unknown module_type + unknown profile falls back to default
13. Default timeout is 60 seconds
14. Default required is True
15. get_active_checks_with_blast_radius — no schema impact returns standard checks
16. get_active_checks_with_blast_radius — schema impact adds cross_stack for backend
17. get_active_checks_with_blast_radius — schema impact skips cross_stack for frontend
18. get_active_checks_with_blast_radius — returns a fresh list (mutation-safe)
"""
from __future__ import annotations

import pytest

from sacv.checks.routing.check_profiles import (
    CheckSpec,
    get_checks,
    get_active_checks_with_blast_radius,
    _MATRIX,
    _DEFAULT_PROFILE,
)


class TestGetChecks:

    def test_backend_domain_standard(self):
        checks = get_checks("backend-domain", "standard")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch"]
        assert checks[0].timeout == 60
        assert checks[1].timeout == 30

    def test_backend_domain_full(self):
        checks = get_checks("backend-domain", "full")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch", "cross_stack", "perf"]
        assert checks[2].required is False
        assert checks[3].required is False

    def test_backend_api_standard(self):
        checks = get_checks("backend-api", "standard")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch", "blast_radius"]
        assert checks[2].required is False

    def test_backend_api_full(self):
        checks = get_checks("backend-api", "full")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch", "blast_radius", "cross_stack"]
        # In full mode, blast_radius and cross_stack are required
        assert checks[2].required is True
        assert checks[3].required is True

    def test_frontend_feature_standard(self):
        checks = get_checks("frontend-feature", "standard")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch"]

    def test_frontend_feature_full(self):
        checks = get_checks("frontend-feature", "full")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch", "visual_diff"]
        assert checks[2].timeout == 90
        assert checks[2].required is False

    def test_frontend_data_standard(self):
        checks = get_checks("frontend-data", "standard")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch"]

    def test_frontend_data_full(self):
        checks = get_checks("frontend-data", "full")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch", "cross_stack"]
        assert checks[2].timeout == 90

    def test_infrastructure_standard(self):
        checks = get_checks("infrastructure", "standard")
        assert [c.name for c in checks] == ["lsp"]

    def test_infrastructure_full(self):
        checks = get_checks("infrastructure", "full")
        assert [c.name for c in checks] == ["lsp"]

    def test_cross_cutting_standard(self):
        checks = get_checks("cross-cutting", "standard")
        assert [c.name for c in checks] == ["lsp"]

    def test_cross_cutting_full(self):
        checks = get_checks("cross-cutting", "full")
        names = [c.name for c in checks]
        assert names == ["lsp", "arch"]

    def test_unknown_module_type_falls_back_to_default(self):
        checks = get_checks("unknown-module", "standard")
        assert checks is _DEFAULT_PROFILE
        assert [c.name for c in checks] == ["lsp"]

    def test_unknown_profile_falls_back_to_default(self):
        checks = get_checks("backend-domain", "nonexistent")
        assert [c.name for c in checks] == ["lsp"]

    def test_both_unknown_falls_back_to_default(self):
        checks = get_checks("unknown", "unknown")
        assert checks is _DEFAULT_PROFILE


class TestCheckSpecDefaults:

    def test_default_timeout_is_60(self):
        spec = CheckSpec("lsp")
        assert spec.timeout == 60

    def test_default_required_is_true(self):
        spec = CheckSpec("lsp")
        assert spec.required is True

    def test_custom_timeout_and_required(self):
        spec = CheckSpec("cross_stack", timeout=90, required=False)
        assert spec.timeout == 90
        assert spec.required is False

    def test_checkspec_is_frozen(self):
        spec = CheckSpec("lsp")
        with pytest.raises(Exception):  # FrozenInstanceError (dataclass)
            spec.timeout = 120


class TestGetActiveChecksWithBlastRadius:

    def test_no_schema_impact_returns_standard(self):
        checks = get_active_checks_with_blast_radius("backend-domain", has_schema_impact=False)
        standard = get_checks("backend-domain", "standard")
        assert [c.name for c in checks] == [c.name for c in standard]
        assert len(checks) == len(standard)

    def test_schema_impact_adds_cross_stack_to_backend(self):
        checks = get_active_checks_with_blast_radius("backend-domain", has_schema_impact=True)
        names = [c.name for c in checks]
        assert "cross_stack" in names
        # Should have standard lsp + arch + cross_stack
        assert len(checks) == 3

    def test_schema_impact_adds_cross_stack_to_backend_api(self):
        checks = get_active_checks_with_blast_radius("backend-api", has_schema_impact=True)
        names = [c.name for c in checks]
        assert "cross_stack" in names
        # Standard backend-api already has blast_radius; cross_stack is added
        assert len(checks) == 4

    def test_schema_impact_skips_cross_stack_for_frontend(self):
        checks = get_active_checks_with_blast_radius("frontend-feature", has_schema_impact=True)
        names = [c.name for c in checks]
        assert "cross_stack" not in names
        # Frontend should not get cross_stack added
        assert len(checks) == 2

    def test_schema_impact_adds_cross_stack_for_infrastructure(self):
        checks = get_active_checks_with_blast_radius("infrastructure", has_schema_impact=True)
        names = [c.name for c in checks]
        # infrastructure doesn't contain "frontend", so cross_stack is added
        assert "cross_stack" in names
        assert names == ["lsp", "cross_stack"]

    def test_returns_fresh_list(self):
        checks1 = get_active_checks_with_blast_radius("backend-domain", has_schema_impact=True)
        checks2 = get_active_checks_with_blast_radius("backend-domain", has_schema_impact=True)
        assert checks1 is not checks2
        # Mutating one does not affect the other
        checks1.pop()
        assert len(checks2) == 3

    def test_unknown_module_type_gets_default_plus_cross_stack(self):
        checks = get_active_checks_with_blast_radius("unknown", has_schema_impact=True)
        # Unknown module_type falls back to _DEFAULT_PROFILE (lsp only), then cross_stack is added
        names = [c.name for c in checks]
        assert names == ["lsp", "cross_stack"]
