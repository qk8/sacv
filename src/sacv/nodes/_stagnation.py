"""
nodes/_stagnation.py
====================
Stagnation detection for the Actor node.

Two modes:
1. Iteration-based: attempt_count >= threshold → stagnation
2. Semantic: cosine similarity of last two error embeddings >= threshold

Both are pure functions (given the embedded float vectors already computed).
The embedding computation itself (external API call) is done once in the
Verifier node and stored in ``correction_state.error_history``.
"""
from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from sacv.orchestration.config import WorkflowConfig
    from sacv.orchestration.state import CorrectionCycleState

log = structlog.get_logger(__name__)


class Embedder(Protocol):
    """Interface for error embedding functions."""
    def embed(self, text: str) -> list[float]: ...


class CharFrequencyEmbedder:
    """Default embedder: character-frequency vector (256 dimensions).

    Fast, deterministic, no external API calls. Detects textually similar
    errors (repeated compilation failures, identical test failure messages).
    """
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 256
        for ch in text[:2000]:
            vec[ord(ch) % 256] += 1.0
        magnitude = sum(v * v for v in vec) ** 0.5
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec


_DEFAULT_EMBEDDER: Embedder = CharFrequencyEmbedder()


def check_stagnation(
    correction: "CorrectionCycleState",
    config:     "WorkflowConfig",
) -> str | None:
    """
    Returns the stagnation pattern name if stagnation is detected, else None.

    This is called at the START of each Actor invocation so the graph can
    short-circuit to HITL before wasting a full diff generation attempt.
    """
    attempt = correction["attempt_count"]
    history = correction.get("error_history", [])

    # Iteration-based stagnation (fast path — no vector math)
    # Use max_self_correction_cycles as the single source of truth for the
    # iteration-based abort threshold (BUG-011 fix).
    abort_threshold = config.max_self_correction_cycles
    if attempt >= abort_threshold:
        return "iteration"

    # Semantic stagnation (requires at least 2 error records)
    if len(history) >= 2:
        sim = _cosine_similarity_from_b64(history[-1], history[-2])
        if sim >= 0.70:
            log.debug(
                "stagnation.similarity_tested",
                similarity=sim,
                threshold=config.stagnation.semantic_similarity_threshold,
            )
        if sim >= config.stagnation.semantic_similarity_threshold:
            return "semantic"

    return None


def embed_error_to_b64(error_text: str) -> str:
    """
    Produces a deterministic, lightweight embedding of an error message
    for stagnation detection.  Delegates to ``_DEFAULT_EMBEDDER``.
    """
    vec = _DEFAULT_EMBEDDER.embed(error_text)
    packed = struct.pack(f"{len(vec)}f", *vec)
    return base64.b64encode(packed).decode("ascii")


def _cosine_similarity_from_b64(b64_a: str, b64_b: str) -> float:
    """Deserialise two base64 vectors and compute cosine similarity."""
    try:
        raw_a = base64.b64decode(b64_a)
        raw_b = base64.b64decode(b64_b)
        n = len(raw_a) // 4
        vec_a = struct.unpack(f"{n}f", raw_a)
        vec_b = struct.unpack(f"{n}f", raw_b)
    except Exception:
        log.warning("stagnation.similarity_error",
                     a_len=len(b64_a), b_len=len(b64_b), exc_info=True)
        return 0.0

    dot   = float(sum((a * b for a, b in zip(vec_a, vec_b)), 0.0))
    mag_a = float(float(sum((a * a for a in vec_a), 0.0)) ** 0.5)
    mag_b = float(float(sum((b * b for b in vec_b), 0.0)) ** 0.5)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(dot / (mag_a * mag_b))
