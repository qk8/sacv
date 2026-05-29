"""Unit tests for the check routing matrix."""
from __future__ import annotations
import pytest
from sacv.checks.routing.check_profiles import (
    get_active_checks, get_active_checks_with_blast_radius, CHECK_MATRIX
)


class TestCheckRouter:
    def test_backend_domain_full_has_resilience(self):
        assert "resilience" in get_active_checks("backend-domain", "full")

    def test_backend_domain_minimal_has_no_security(self):
        assert "security" not in get_active_checks("backend-domain", "minimal")

    def test_backend_api_standard_has_integration(self):
        assert "integration" in get_active_checks("backend-api", "standard")

    def test_frontend_feature_full_has_no_resilience(self):
        assert "resilience" not in get_active_checks("frontend-feature", "full")

    def test_arch_check_in_standard_and_full(self):
        """All non-minimal profiles must include arch_check (approach 9, 10)."""
        for mt in CHECK_MATRIX:
            for profile in ("standard", "full"):
                assert "arch_check" in get_active_checks(mt, profile), \
                    f"{mt}/{profile} missing arch_check"

    def test_arch_check_not_in_minimal(self):
        for mt in CHECK_MATRIX:
            checks = get_active_checks(mt, "minimal")
            assert "arch_check" not in checks, f"{mt}/minimal should not have arch_check"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_active_checks("unknown-module", "standard")

    def test_minimal_subset_of_standard(self):
        for mt in CHECK_MATRIX:
            m = set(get_active_checks(mt, "minimal"))
            s = set(get_active_checks(mt, "standard"))
            assert m <= s, f"{mt}: minimal not subset of standard"

    def test_standard_subset_of_full(self):
        for mt in CHECK_MATRIX:
            s = set(get_active_checks(mt, "standard"))
            f = set(get_active_checks(mt, "full"))
            assert s <= f, f"{mt}: standard not subset of full"

    def test_all_profiles_have_test_execution(self):
        for mt in CHECK_MATRIX:
            for profile in ("minimal", "standard", "full"):
                assert "test_execution" in get_active_checks(mt, profile)

    def test_schema_impact_adds_cross_domain_api(self):
        """Approach 5: schema impact triggers extra pipeline."""
        checks = get_active_checks_with_blast_radius(
            "backend-domain", "standard", schema_impact=["users_table"]
        )
        assert "cross_domain_api" in checks

    def test_no_schema_impact_no_extra(self):
        base    = get_active_checks("backend-domain", "standard")
        derived = get_active_checks_with_blast_radius("backend-domain", "standard", [])
        assert derived == base
