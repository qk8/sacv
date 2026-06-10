"""
tests/unit/test_cadence_reserved.py
=====================================
LOW-01: CadenceConfig fields are reserved, not consumed.

Verifies:
  1. CadenceConfig has a docstring documenting it as reserved
  2. CadenceConfig fields parse correctly from JSON
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import CadenceConfig, WorkflowConfig


class TestCadenceConfigReserved:

    def test_cadence_config_has_reserved_docstring(self):
        """CadenceConfig documents its fields as reserved for future implementation."""
        assert CadenceConfig.__doc__ is not None
        assert "reserved" in CadenceConfig.__doc__.lower()

    def test_cadence_config_defaults(self):
        """CadenceConfig has expected default values."""
        cfg = CadenceConfig()
        assert cfg.cleanup_interval == 25
        assert cfg.llm_quality_interval == 10
        assert cfg.drift_check_interval == {"simple": 20, "medium": 15, "complex": 10}

    def test_cadence_config_from_json(self, tmp_path):
        """CadenceConfig values are read from JSON config."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            '{"cadence": {"cleanup_interval": 50, "llm_quality_interval": 20}}'
        )
        cfg = WorkflowConfig.from_json(cfg_file)
        assert cfg.cadence.cleanup_interval == 50
        assert cfg.cadence.llm_quality_interval == 20
        # drift_check_interval should use default since not specified
        assert cfg.cadence.drift_check_interval == {"simple": 20, "medium": 15, "complex": 10}
