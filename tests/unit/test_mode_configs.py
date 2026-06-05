"""
tests/unit/test_mode_configs.py
================================
Unit tests for mode-specific configuration classes.

Tests cover:
1. GreenfieldConfig — check_overrides, critic_weights
2. BrownfieldConfig — check_overrides, critic_weights
3. ModeConfig base class
"""
from __future__ import annotations

import pytest
from sacv.modes.greenfield import GreenfieldConfig
from sacv.modes.brownfield import BrownfieldConfig
from sacv.modes.base import ModeConfig


class TestGreenfieldConfig:

    def test_check_overrides_enforces_ddd(self):
        config = GreenfieldConfig()
        overrides = config.check_overrides()
        assert overrides["enforce_ddd"] is True

    def test_check_overrides_enforces_solid(self):
        config = GreenfieldConfig()
        overrides = config.check_overrides()
        assert overrides["enforce_solid"] is True

    def test_check_overrides_disallows_legacy_patterns(self):
        config = GreenfieldConfig()
        overrides = config.check_overrides()
        assert overrides["allow_legacy_patterns"] is False

    def test_check_overrides_returns_dict(self):
        config = GreenfieldConfig()
        result = config.check_overrides()
        assert isinstance(result, dict)

    def test_critic_weights_style_emphasized(self):
        config = GreenfieldConfig()
        weights = config.critic_weights()
        assert weights["style"] == 1.2

    def test_critic_weights_consistency_de_emphasized(self):
        config = GreenfieldConfig()
        weights = config.critic_weights()
        assert weights["consistency"] == 0.8

    def test_critic_weights_security_neutral(self):
        config = GreenfieldConfig()
        weights = config.critic_weights()
        assert weights["security"] == 1.0

    def test_critic_weights_returns_dict(self):
        config = GreenfieldConfig()
        result = config.critic_weights()
        assert isinstance(result, dict)

    def test_all_critic_weights_present(self):
        config = GreenfieldConfig()
        weights = config.critic_weights()
        assert "security" in weights
        assert "style" in weights
        assert "consistency" in weights


class TestBrownfieldConfig:

    def test_check_overrides_allows_legacy_patterns(self):
        config = BrownfieldConfig()
        overrides = config.check_overrides()
        assert overrides["allow_legacy_patterns"] is True

    def test_check_overrides_disallows_ddd(self):
        config = BrownfieldConfig()
        overrides = config.check_overrides()
        assert overrides["enforce_ddd"] is False

    def test_check_overrides_disallows_solid(self):
        config = BrownfieldConfig()
        overrides = config.check_overrides()
        assert overrides["enforce_solid"] is False

    def test_check_overrides_requires_blast_radius(self):
        config = BrownfieldConfig()
        overrides = config.check_overrides()
        assert overrides["require_blast_radius"] is True

    def test_check_overrides_has_backward_compat_guard(self):
        config = BrownfieldConfig()
        overrides = config.check_overrides()
        assert overrides["backward_compat_guard"] is True

    def test_check_overrides_returns_dict(self):
        config = BrownfieldConfig()
        result = config.check_overrides()
        assert isinstance(result, dict)

    def test_critic_weights_style_de_emphasized(self):
        config = BrownfieldConfig()
        weights = config.critic_weights()
        assert weights["style"] == 0.7

    def test_critic_weights_consistency_emphasized(self):
        config = BrownfieldConfig()
        weights = config.critic_weights()
        assert weights["consistency"] == 1.5

    def test_critic_weights_security_neutral(self):
        config = BrownfieldConfig()
        weights = config.critic_weights()
        assert weights["security"] == 1.0

    def test_critic_weights_returns_dict(self):
        config = BrownfieldConfig()
        result = config.critic_weights()
        assert isinstance(result, dict)

    def test_all_critic_weights_present(self):
        config = BrownfieldConfig()
        weights = config.critic_weights()
        assert "security" in weights
        assert "style" in weights
        assert "consistency" in weights


class TestModeConfigBase:

    def test_base_class_has_check_overrides(self):
        """ModeConfig.base defines check_overrides method signature."""
        assert hasattr(ModeConfig, "check_overrides")

    def test_base_class_has_critic_weights(self):
        """ModeConfig.base defines critic_weights method signature."""
        assert hasattr(ModeConfig, "critic_weights")

    def test_greenfield_inherits_from_mode_config(self):
        assert issubclass(GreenfieldConfig, ModeConfig)

    def test_brownfield_inherits_from_mode_config(self):
        assert issubclass(BrownfieldConfig, ModeConfig)


class TestModeConfigDifferences:
    """Compare greenfield vs brownfield configurations."""

    def test_ddd_enforcement_differs(self):
        gf = GreenfieldConfig()
        bf = BrownfieldConfig()
        assert gf.check_overrides()["enforce_ddd"] != bf.check_overrides()["enforce_ddd"]

    def test_legacy_patterns_differs(self):
        gf = GreenfieldConfig()
        bf = BrownfieldConfig()
        assert gf.check_overrides()["allow_legacy_patterns"] != bf.check_overrides()["allow_legacy_patterns"]

    def test_style_weight_differs(self):
        gf = GreenfieldConfig()
        bf = BrownfieldConfig()
        assert gf.critic_weights()["style"] != bf.critic_weights()["style"]

    def test_consistency_weight_differs(self):
        gf = GreenfieldConfig()
        bf = BrownfieldConfig()
        assert gf.critic_weights()["consistency"] != bf.critic_weights()["consistency"]

    def test_security_weight_same(self):
        gf = GreenfieldConfig()
        bf = BrownfieldConfig()
        assert gf.critic_weights()["security"] == bf.critic_weights()["security"]

    def test_brownfield_has_extra_overrides(self):
        """Brownfield has blast_radius and backward_compat guards not in greenfield."""
        bf = BrownfieldConfig()
        overrides = bf.check_overrides()
        assert "require_blast_radius" in overrides
        assert "backward_compat_guard" in overrides
