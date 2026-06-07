"""
nodes/_scoring.py
=================
Pure, side-effect-free scoring functions used by ValueNode.
All logic here is deterministic and directly unit-testable without mocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sacv.orchestration.config import WorkflowConfig
    from sacv.orchestration.state import StrategyCandidate


@dataclass(frozen=True)
class ScoringWeights:
    token_depth:  float = 0.40
    collision:    float = 0.30
    blast_radius: float = 0.30

    def __post_init__(self) -> None:
        total = self.token_depth + self.collision + self.blast_radius
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ScoringWeights must sum to 1.0; got {total:.4f}")


def score_strategy(
    affected_files:      list[str],
    collision_ratio:     float,
    blast_radius_impact: float,
    config:              "WorkflowConfig | None" = None,
    weights:             ScoringWeights = ScoringWeights(),
) -> float:
    """
    Pure function — no I/O.

    Returns a composite score in [0, 1]; higher is better.

    Args:
        affected_files:      files this strategy will touch
        collision_ratio:     0..1 — fraction of affected files shared with peer strategies
        blast_radius_impact: 0..1 — normalised blast-radius overlap (Brownfield only)
        config:              provides ``max_blast_files`` denominator; defaults to 50
        weights:             scoring weight distribution; must sum to 1.0
    """
    max_files = config.max_blast_files if config else 50

    score_token    = max(0.0, 1.0 - len(affected_files) / max_files)
    score_collision = max(0.0, 1.0 - collision_ratio)
    score_blast    = max(0.0, 1.0 - blast_radius_impact)

    return (
        weights.token_depth   * score_token
        + weights.collision   * score_collision
        + weights.blast_radius * score_blast
    )


def prune_strategies(
    candidates: list["StrategyCandidate"],
    config:     "WorkflowConfig | None" = None,
) -> list["StrategyCandidate"]:
    """
    Pure function — no I/O.

    Filters candidates below ``min_strategy_score`` and returns
    the top ``max_strategies`` sorted by ``composite_score`` descending.
    """
    min_score     = config.min_strategy_score if config else 0.3
    max_strategies = config.max_strategies    if config else 3

    passing = [c for c in candidates if c["composite_score"] >= min_score]
    return sorted(passing, key=lambda x: x["composite_score"], reverse=True)[:max_strategies]


def detect_collision_pairs(
    strategies: list["StrategyCandidate"],
) -> list[tuple[str, str, list[str]]]:
    """
    Pure function — no I/O.

    Returns ``(strategy_a_id, strategy_b_id, shared_files)`` for every pair
    of strategies that would touch at least one common file.
    """
    pairs: list[tuple[str, str, list[str]]] = []
    for i, a in enumerate(strategies):
        for b in strategies[i + 1:]:
            shared = sorted(set(a["affected_files"]) & set(b["affected_files"]))
            if shared:
                pairs.append((a["strategy_id"], b["strategy_id"], shared))
    return pairs


def compute_semantic_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two embedding vectors.
    Pure function — used by stagnation detection in edges.py.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot   = float(sum((a * b for a, b in zip(vec_a, vec_b)), 0.0))
    mag_a = float(float(sum((a * a for a in vec_a), 0.0)) ** 0.5)
    mag_b = float(float(sum((b * b for b in vec_b), 0.0)) ** 0.5)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(dot / (mag_a * mag_b))
