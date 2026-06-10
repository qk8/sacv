"""
tests/unit/test_config_cadence.py
==================================
Tests for ST-003: CadenceConfig fields are documented as unimplemented.
"""
from __future__ import annotations

import pytest


class TestCadenceConfigDocumentation:

    def test_docstring_warns_fields_are_unimplemented(self):
        """CadenceConfig docstring must clearly state fields have no effect."""
        from sacv.orchestration.config import CadenceConfig
        doc = CadenceConfig.__doc__
        assert doc is not None
        doc_lower = doc.lower()
        # Must contain unimplemented warning
        assert "not yet implemented" in doc_lower or "unimplemented" in doc_lower or "no effect" in doc_lower, (
            "CadenceConfig docstring must clearly state that these fields are unimplemented "
            "and have no effect on workflow behavior"
        )

    def test_default_values_are_reasonable(self):
        """CadenceConfig defaults should be sensible numbers."""
        from sacv.orchestration.config import CadenceConfig
        c = CadenceConfig()
        assert isinstance(c.cleanup_interval, int)
        assert c.cleanup_interval > 0
        assert isinstance(c.llm_quality_interval, int)
        assert c.llm_quality_interval > 0
        assert isinstance(c.drift_check_interval, dict)
