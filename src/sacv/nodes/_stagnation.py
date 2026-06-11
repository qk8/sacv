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
import re
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

_NOISE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?"),  # ISO timestamps
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # UUIDs
    re.compile(r"\bcontainer=[a-z0-9\-]+\b"),     # Docker container IDs
    re.compile(r"\bcorrelation=[a-z0-9\-]+\b"),   # correlation IDs
    re.compile(r"\breq-[a-z0-9]+\b"),             # request IDs
]


def _normalize_error_text(text: str) -> str:
    """
    Strip variable noise from error text before embedding.

    Timestamps, UUIDs, and correlation IDs change on every run but carry
    no semantic information for stagnation detection. Removing them ensures
    that two identical errors on different runs produce similar embeddings.
    """
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


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


def compute_outcome_signature(
    preflight_result: dict[str, Any] | None,
    critic_findings:  list[dict[str, Any]],
) -> str:
    """
    Compute a deterministic signature from preflight violations and
    critical critic findings. Used for outcome-based stagnation detection.

    The signature captures the KEY PROBLEMS that persist across attempts —
    if the same signature appears in consecutive attempts, the actor is
    changing code but not fixing the underlying problem.
    """
    import hashlib

    parts: list[str] = []

    if preflight_result:
        for key in ("lsp_errors", "arch_violations", "cross_stack_errors", "blast_errors"):
            violations = preflight_result.get(key, [])
            for v in violations:
                # Use only the rule/message, not file paths (which change)
                rule = v.get("rule", "") or v.get("code", "") or v.get("message", "")[:80]
                if rule:
                    parts.append(f"preflight:{key}:{rule}")

    for f in critic_findings:
        if f.get("severity") == "critical":
            rule = f.get("rule_id", "") or f.get("message", "")[:80]
            if rule:
                parts.append(f"critic:{f.get('critic', '?')}:{rule}")

    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()[:16]


def check_outcome_stagnation(
    correction: "CorrectionCycleState",
    current_sig: str,
) -> bool:
    """
    Detect outcome-based stagnation: the same preflight/critic problem
    persists across consecutive attempts.

    Returns True if stagnation is detected.
    """
    prev_sig = correction.get("last_outcome_signature")
    if prev_sig and current_sig == prev_sig and current_sig != "":
        return True
    return False


def embed_error_to_b64(error_text: str) -> str:
    """
    Produces a deterministic, lightweight embedding of an error message
    for stagnation detection.  Strips variable noise (timestamps, UUIDs)
    before embedding so that identical errors on different runs produce
    similar embeddings.
    """
    normalized = _normalize_error_text(error_text)
    vec = _DEFAULT_EMBEDDER.embed(normalized)
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
