"""
tests/unit/test_verifier_span_events.py
==========================================
OTEL-001: Verify verifier module has span_event import and calls.
"""
from __future__ import annotations

import inspect


class TestVerifierSpanEvents:

    def test_verifier_imports_span_event(self):
        """verifier.py imports span_event from sacv.tracing."""
        from sacv.nodes import verifier
        source = inspect.getsource(verifier)
        assert "from sacv.tracing import span_event" in source, (
            "verifier.py should import span_event from sacv.tracing"
        )

    def test_verifier_calls_span_event_on_complete(self):
        """verifier.py calls span_event('verifier.complete')."""
        from sacv.nodes import verifier
        source = inspect.getsource(verifier)
        assert 'span_event("verifier.complete"' in source, (
            "verifier.py should call span_event('verifier.complete')"
        )
