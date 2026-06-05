"""Unit tests for the Value Node pure scoring functions."""
from __future__ import annotations
import pytest
from sacv.nodes._scoring import (
    score_strategy, prune_strategies, detect_collision_pairs,
    ScoringWeights, compute_semantic_similarity,
)
from sacv.orchestration.config import WorkflowConfig

def _c(sid, score, files=None):
    return {"strategy_id":sid,"description":"","affected_files":files or [],
            "token_depth_score":score,"collision_score":score,
            "blast_radius_score":score,"composite_score":score}

class TestScoreStrategy:
    def test_perfect_scores_one(self):
        assert score_strategy([], 0.0, 0.0) == pytest.approx(1.0)

    def test_max_files_drives_token_to_zero(self):
        cfg = WorkflowConfig(max_blast_files=10)
        s   = score_strategy(["f"]*10, 0.0, 0.0, config=cfg)
        assert 0.55 < s < 0.65

    def test_full_collision_loses_30pct(self):
        s = score_strategy([], 1.0, 0.0)
        assert s == pytest.approx(0.70)

    def test_scores_clamped_at_zero(self):
        assert score_strategy(["f"]*9999, 2.0, 5.0) == 0.0

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            ScoringWeights(token_depth=0.5, collision=0.5, blast_radius=0.5)

class TestPruneStrategies:
    def test_below_threshold_pruned(self):
        cfg = WorkflowConfig(min_strategy_score=0.5)
        r   = prune_strategies([_c("a",0.2), _c("b",0.8)], config=cfg)
        assert len(r)==1 and r[0]["composite_score"]==0.8

    def test_capped_at_max(self):
        cfg = WorkflowConfig(max_strategies=2, min_strategy_score=0.0)
        r   = prune_strategies([_c(str(i),float(i)) for i in range(5)], config=cfg)
        assert len(r)==2

    def test_sorted_descending(self):
        cfg = WorkflowConfig(min_strategy_score=0.0)
        r   = prune_strategies([_c("lo",0.3), _c("hi",0.9)], config=cfg)
        assert r[0]["strategy_id"]=="hi"

class TestCollisionDetection:
    def test_no_shared_files(self):
        a = _c("a",1.0,["A.java"]); b = _c("b",1.0,["B.java"])
        assert detect_collision_pairs([a,b]) == []

    def test_shared_file_detected(self):
        a = _c("a",1.0,["Shared.java","A.java"])
        b = _c("b",1.0,["Shared.java","B.java"])
        pairs = detect_collision_pairs([a,b])
        assert len(pairs)==1
        assert "Shared.java" in pairs[0][2]


class TestComputeSemanticSimilarity:

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert compute_semantic_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert compute_semantic_similarity(a, b) == pytest.approx(0.0)

    def test_negative_similarity(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert compute_semantic_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert compute_semantic_similarity([], []) == pytest.approx(0.0)

    def test_one_empty_returns_zero(self):
        assert compute_semantic_similarity([1.0], []) == pytest.approx(0.0)
        assert compute_semantic_similarity([], [1.0]) == pytest.approx(0.0)

    def test_different_lengths(self):
        assert compute_semantic_similarity([1.0, 2.0], [1.0]) == pytest.approx(0.0)

    def test_zero_magnitude(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert compute_semantic_similarity(a, b) == pytest.approx(0.0)

    def test_scaled_vectors_same_direction(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert compute_semantic_similarity(a, b) == pytest.approx(1.0)

    def test_high_dimensional_similarity(self):
        import random
        random.seed(42)
        vec = [random.gauss(0, 1) for _ in range(1000)]
        assert compute_semantic_similarity(vec, vec) == pytest.approx(1.0)
