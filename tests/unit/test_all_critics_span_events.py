"""
tests/unit/test_all_critics_span_events.py
=============================================
OTEL-001: Verify all_critics node records span_event() at key transitions.
"""
from __future__ import annotations

import inspect


class TestAllCriticsSpanEvents:

    def test_graph_imports_span_event(self):
        """graph.py imports span_event from sacv.tracing."""
        from sacv.orchestration import graph
        source = inspect.getsource(graph)
        assert "from sacv.tracing import span_event" in source, (
            "graph.py should import span_event from sacv.tracing"
        )

    def test_graph_calls_span_event_for_critics(self):
        """graph.py calls span_event('critics.complete') in all_critics node."""
        from sacv.orchestration import graph
        source = inspect.getsource(graph)
        assert 'span_event("critics.complete"' in source, (
            "graph.py should call span_event('critics.complete')"
        )
