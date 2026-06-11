"""
tests/unit/test_stagnation_normalize.py
========================================
DBG-002: Verify that error text normalization strips variable noise
(timestamps, UUIDs, container IDs) so identical errors on different
runs produce similar embeddings.
"""
from __future__ import annotations

from sacv.nodes._stagnation import (
    _normalize_error_text,
    embed_error_to_b64,
    _cosine_similarity_from_b64,
)


class TestNormalizeErrorText:

    def test_strips_iso_timestamps(self):
        text = "2024-01-15T14:32:01.234Z ERROR something failed"
        result = _normalize_error_text(text)
        assert "2024-01-15" not in result
        assert "14:32:01" not in result

    def test_strips_uuids(self):
        text = "req-id: 550e8400-e29b-41d4-a716-446655440000 failed"
        result = _normalize_error_text(text)
        assert "550e8400" not in result

    def test_strips_container_ids(self):
        text = "container=sacv-sandbox-abc123 ERROR failed"
        result = _normalize_error_text(text)
        assert "container=" not in result

    def test_strips_correlation_ids(self):
        text = "correlation=req-abc123 ERROR failed"
        result = _normalize_error_text(text)
        assert "correlation=" not in result

    def test_strips_req_ids(self):
        text = "req-789 failed with error"
        result = _normalize_error_text(text)
        # req-789 matches the \breq-[a-z0-9]+\b pattern
        assert "req-789" not in result

    def test_collapses_whitespace(self):
        text = "error    with     lots    of     spaces"
        result = _normalize_error_text(text)
        assert "  " not in result
        assert result == "error with lots of spaces"


class TestStagnationEmbeddingNoiseResistance:

    def test_same_error_different_timestamps_similar(self):
        """Two identical errors with different timestamps must produce similar embeddings."""
        template = (
            "{ts} ERROR container=sacv-sandbox-abc123\n"
            "java.lang.NullPointerException: Cannot invoke getId() on null\n"
            "\tat com.example.UserService.findUser(UserService.java:42)\n"
        )
        e1 = embed_error_to_b64(template.format(ts="2024-01-01T00:00:00Z"))
        e2 = embed_error_to_b64(template.format(ts="2024-06-15T12:34:56Z"))
        sim = _cosine_similarity_from_b64(e1, e2)
        assert sim >= 0.90, (
            f"Expected high similarity for same error with different timestamps, "
            f"got {sim:.3f}"
        )

    def test_same_error_different_uuids_similar(self):
        """Same error with different UUIDs must produce similar embeddings."""
        template = (
            "ERROR req-{uuid} container=sacv-sandbox-abc123\n"
            "java.lang.NullPointerException: findById returned null\n"
            "\tat com.example.UserRepo.findById(UserRepo.java:15)\n"
        )
        e1 = embed_error_to_b64(template.format(uuid="550e8400-e29b-41d4-a716-446655440000"))
        e2 = embed_error_to_b64(template.format(uuid="6ba7b810-9dad-11d1-80b4-00c04fd430c8"))
        sim = _cosine_similarity_from_b64(e1, e2)
        assert sim >= 0.90, (
            f"Expected high similarity for same error with different UUIDs, "
            f"got {sim:.3f}"
        )
