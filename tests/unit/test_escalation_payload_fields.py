"""
tests/unit/test_escalation_payload_fields.py
==============================================
AUD-003: Verify EscalationPayload includes diagnostic fields
(debug_observations, last_preflight, last_test_failures).
"""
from __future__ import annotations

from sacv.orchestration.state import EscalationPayload


class TestEscalationPayloadFields:

    def test_has_debug_observations_field(self):
        """EscalationPayload must include debug_observations for root-cause inspection."""
        assert "debug_observations" in EscalationPayload.__annotations__

    def test_has_last_preflight_field(self):
        """EscalationPayload must include last_preflight for preflight loop diagnosis."""
        assert "last_preflight" in EscalationPayload.__annotations__

    def test_has_last_test_failures_field(self):
        """EscalationPayload must include last_test_failures for immediate inspection."""
        assert "last_test_failures" in EscalationPayload.__annotations__

    def test_all_new_fields_optional(self):
        """All three new fields accept None (they may not be present in state)."""
        from sacv.orchestration.state import PreflightResult, TestFailure
        annotations = EscalationPayload.__annotations__
        # debug_observations should accept None or DebugObservations
        ann = str(annotations["debug_observations"])
        assert "None" in ann or "NoneType" in ann or "Union" in ann, (
            f"debug_observations should accept None: {ann}"
        )
        # last_preflight should accept None or PreflightResult
        ann = str(annotations["last_preflight"])
        assert "None" in ann or "NoneType" in ann or "Union" in ann, (
            f"last_preflight should accept None: {ann}"
        )
        # last_test_failures should be a list of TestFailure
        ann = str(annotations["last_test_failures"])
        assert "TestFailure" in ann, f"last_test_failures should contain TestFailure: {ann}"
