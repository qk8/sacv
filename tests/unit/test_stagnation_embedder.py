"""
tests/unit/test_stagnation_embedder.py
=======================================
LOW-02: Pluggable embedder for stagnation detection.

Tests for the Embedder Protocol, CharFrequencyEmbedder, and embed_error_to_b64.
"""
from __future__ import annotations

import base64
import math
import struct

import pytest

from sacv.nodes._stagnation import (
    CharFrequencyEmbedder,
    Embedder,
    _DEFAULT_EMBEDDER,
    embed_error_to_b64,
)


class TestCharFrequencyEmbedder:

    def test_returns_list_of_256_floats(self):
        vec = CharFrequencyEmbedder().embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 256
        assert all(isinstance(v, float) for v in vec)

    def test_normalized_magnitude_is_one(self):
        vec = CharFrequencyEmbedder().embed("some error message")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert pytest.approx(magnitude, abs=1e-9) == 1.0

    def test_empty_string_produces_zero_vector(self):
        vec = CharFrequencyEmbedder().embed("")
        assert all(v == 0.0 for v in vec)

    def test_deterministic(self):
        vec1 = CharFrequencyEmbedder().embed("duplicate error")
        vec2 = CharFrequencyEmbedder().embed("duplicate error")
        assert vec1 == vec2

    def test_same_errors_similar_vectors(self):
        error = "TypeError: cannot read property 'length' of undefined"
        vec1 = CharFrequencyEmbedder().embed(error)
        vec2 = CharFrequencyEmbedder().embed(error)
        dot = sum(a * b for a, b in zip(vec1, vec2))
        assert pytest.approx(dot, abs=1e-9) == 1.0


class TestEmbedErrorToB64:

    def test_returns_base64_string(self):
        result = embed_error_to_b64("SyntaxError: unexpected token")
        assert isinstance(result, str)
        # Valid base64 decodes back to bytes
        decoded = base64.b64decode(result)
        assert isinstance(decoded, bytes)

    def test_decodes_to_256_floats(self):
        result = embed_error_to_b64("error msg")
        raw = base64.b64decode(result)
        n = len(raw) // 4
        vec = struct.unpack(f"{n}f", raw)
        assert len(vec) == 256

    def test_deterministic(self):
        b64_1 = embed_error_to_b64("repeat error")
        b64_2 = embed_error_to_b64("repeat error")
        assert b64_1 == b64_2

    def test_truncates_long_errors(self):
        """Embedder caps at 2000 chars — extra chars should be ignored."""
        short = embed_error_to_b64("x" * 2000)
        long_ = embed_error_to_b64("x" * 3000)
        assert short == long_


class TestDefaultEmbedder:

    def test_is_char_frequency_embedder(self):
        assert isinstance(_DEFAULT_EMBEDDER, CharFrequencyEmbedder)

    def test_satisfies_embedder_protocol(self):
        """CharFrequencyEmbedder satisfies the Embedder Protocol."""
        embedder: Embedder = _DEFAULT_EMBEDDER
        vec = embedder.embed("protocol check")
        assert len(vec) == 256
