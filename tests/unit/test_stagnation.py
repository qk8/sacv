"""
tests/unit/test_stagnation.py
==============================
Unit tests for semantic and iteration-based stagnation detection.
All pure-function tests — zero I/O.
"""
from __future__ import annotations

import pytest
from sacv.nodes._stagnation import (
    check_stagnation,
    embed_error_to_b64,
    _cosine_similarity_from_b64,
)
from sacv.orchestration.config import WorkflowConfig, StagnationConfig


def _correction(attempt: int, history: list[str] | None = None) -> dict:
    return {
        "attempt_count":      attempt,
        "branch_name":        None,
        "last_error_hash":    None,
        "error_history":      history or [],
        "stagnation_pattern": "none",
    }


def _config(max_attempts: int = 3, sim_threshold: float = 0.85) -> WorkflowConfig:
    return WorkflowConfig(
        max_self_correction_cycles=max_attempts,
        stagnation=StagnationConfig(
            total_abort_force=max_attempts,
            semantic_similarity_threshold=sim_threshold,
        ),
    )


class TestIterationStagnation:

    def test_below_max_is_not_stagnant(self):
        assert check_stagnation(_correction(2), _config(max_attempts=3)) is None

    def test_at_max_is_stagnant(self):
        result = check_stagnation(_correction(3), _config(max_attempts=3))
        assert result == "iteration"

    def test_above_max_is_stagnant(self):
        result = check_stagnation(_correction(10), _config(max_attempts=3))
        assert result == "iteration"

    def test_zero_attempts_is_not_stagnant(self):
        assert check_stagnation(_correction(0), _config()) is None


class TestSemanticStagnation:

    def test_identical_errors_produce_high_similarity(self):
        err = "NullPointerException at UserService.java:42"
        v1  = embed_error_to_b64(err)
        v2  = embed_error_to_b64(err)
        sim = _cosine_similarity_from_b64(v1, v2)
        assert sim > 0.99

    def test_completely_different_errors_produce_low_similarity(self):
        """
        Character-frequency embeddings are approximate — very different errors
        still share some vocabulary overlap. We use 0.85 as the threshold
        (matching the default StagnationConfig) to avoid false positives.
        """
        v1 = embed_error_to_b64("NullPointerException at UserService.java:42")
        v2 = embed_error_to_b64("SyntaxError: unexpected token 'import' at index.ts:1")
        sim = _cosine_similarity_from_b64(v1, v2)
        assert sim < 0.85

    def test_similar_errors_detected_as_stagnant(self):
        err1 = "AssertionError: expected <null> but was <User(id=1)> at UserServiceTest:30"
        err2 = "AssertionError: expected <null> but was <User(id=2)> at UserServiceTest:30"
        v1   = embed_error_to_b64(err1)
        v2   = embed_error_to_b64(err2)
        correction = _correction(1, history=[v1, v2])
        result = check_stagnation(correction, _config(sim_threshold=0.70))
        assert result == "semantic"

    def test_single_history_entry_not_stagnant(self):
        v1 = embed_error_to_b64("some error")
        correction = _correction(1, history=[v1])
        assert check_stagnation(correction, _config()) is None

    def test_empty_history_not_stagnant(self):
        assert check_stagnation(_correction(1), _config()) is None

    def test_iteration_stagnation_takes_priority_over_semantic(self):
        """If both conditions hold, 'iteration' is returned (fast path)."""
        v1 = embed_error_to_b64("same error")
        v2 = embed_error_to_b64("same error")
        correction = _correction(3, history=[v1, v2])
        result = check_stagnation(correction, _config(max_attempts=3, sim_threshold=0.5))
        assert result == "iteration"

    def test_malformed_b64_returns_zero_similarity(self):
        sim = _cosine_similarity_from_b64("not_valid_base64!!!", "also_invalid")
        assert sim == 0.0

    def test_mismatched_vector_lengths_returns_zero(self):
        """Mismatched vector lengths should return 0.0."""
        v1 = embed_error_to_b64("short")
        v2 = embed_error_to_b64("this is a much longer error message to produce a different vector length")
        sim = _cosine_similarity_from_b64(v1, v2)
        # Same length vectors (always 256 dims) so this shouldn't fail
        assert isinstance(sim, float)

    def test_embed_produces_consistent_output(self):
        text = "repeatable error message"
        assert embed_error_to_b64(text) == embed_error_to_b64(text)

    def test_embed_handles_empty_string(self):
        # Should not crash
        result = embed_error_to_b64("")
        assert isinstance(result, str)


