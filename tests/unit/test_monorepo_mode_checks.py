"""
tests/unit/test_monorepo_mode_checks.py
=========================================
LOW-04: monorepo_mode gates cross_stack checks.

Verifies:
  1. cross_stack check is included in full profile when monorepo_mode=True
  2. cross_stack check is excluded when monorepo_mode=False
  3. get_active_checks_with_blast_radius is unaffected
"""
from __future__ import annotations

import pytest

from sacv.checks.routing.check_profiles import get_checks, get_active_checks_with_blast_radius


class TestMonorepoModeGating:

    def test_cross_stack_included_when_monorepo_mode_true(self):
        """Full profile with monorepo_mode=True includes cross_stack check."""
        checks = get_checks("backend-domain", "full", monorepo_mode=True)
        names = {c.name for c in checks}
        assert "cross_stack" in names

    def test_cross_stack_excluded_when_monorepo_mode_false(self):
        """Full profile with monorepo_mode=False excludes cross_stack check."""
        checks = get_checks("backend-domain", "full", monorepo_mode=False)
        names = {c.name for c in checks}
        assert "cross_stack" not in names

    def test_standard_profile_unaffected_by_monorepo_mode(self):
        """Standard profile never includes cross_stack regardless of monorepo_mode."""
        for mode in (True, False):
            checks = get_checks("backend-domain", "standard", monorepo_mode=mode)
            names = {c.name for c in checks}
            assert "cross_stack" not in names, (
                f"standard profile should never include cross_stack (monorepo_mode={mode})"
            )

    def test_blast_radius_route_unaffected(self):
        """get_active_checks_with_blast_radius works without monorepo_mode param."""
        checks = get_active_checks_with_blast_radius("backend-domain", has_schema_impact=True)
        names = {c.name for c in checks}
        # Should have lsp, arch, and cross_stack (from blast radius logic)
        assert "lsp" in names
        assert "arch" in names
        assert "cross_stack" in names
