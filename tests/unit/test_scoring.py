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


class TestScoreStrategyEdgeCases:

    def test_zero_files_with_default_config(self):
        """No files → token_depth=1.0, perfect score."""
        s = score_strategy([], 0.0, 0.0)
        assert s == pytest.approx(1.0)

    def test_exactly_max_files(self):
        """Exactly max_blast_files → token_depth=0."""
        cfg = WorkflowConfig(max_blast_files=10)
        s = score_strategy(["f"] * 10, 0.0, 0.0, config=cfg)
        # token = max(0, 1-10/10) = 0
        # score = 0.4*0 + 0.3*1 + 0.3*1 = 0.6
        assert s == pytest.approx(0.6)

    def test_no_config_uses_default_max_files(self):
        """Without config, max_blast_files defaults to 50."""
        s = score_strategy(["f"] * 50, 0.0, 0.0)
        # token = max(0, 1-50/50) = 0
        # score = 0.4*0 + 0.3*1 + 0.3*1 = 0.6
        assert s == pytest.approx(0.6)

    def test_custom_weights_summing_to_one(self):
        """Custom weights that sum to 1.0 are accepted."""
        w = ScoringWeights(token_depth=0.5, collision=0.25, blast_radius=0.25)
        s = score_strategy([], 0.0, 0.0, weights=w)
        assert s == pytest.approx(1.0)

    def test_custom_weights_affect_score(self):
        """Different weights produce different scores."""
        w1 = ScoringWeights(token_depth=0.9, collision=0.05, blast_radius=0.05)
        w2 = ScoringWeights(token_depth=0.05, collision=0.45, blast_radius=0.5)
        s1 = score_strategy(["f", "g"], 0.5, 0.5, weights=w1)
        s2 = score_strategy(["f", "g"], 0.5, 0.5, weights=w2)
        assert s1 != s2

    def test_negative_collision_ratio_clamped(self):
        """Negative collision_ratio should be clamped via max(0.0, ...)."""
        s = score_strategy([], -0.5, 0.0)
        # collision_score = max(0, 1-(-0.5)) = 1.5
        # score = 0.4*1 + 0.3*1.5 + 0.3*1 = 0.4 + 0.45 + 0.3 = 1.15
        assert s == pytest.approx(1.15)

    def test_negative_blast_radius_clamped(self):
        """Negative blast_radius_impact should be clamped via max(0.0, ...)."""
        s = score_strategy([], 0.0, -0.5)
        # blast_score = max(0, 1-(-0.5)) = 1.5
        # score = 0.4*1 + 0.3*1 + 0.3*1.5 = 0.4 + 0.3 + 0.45 = 1.15
        assert s == pytest.approx(1.15)

    def test_all_max_scores(self):
        """All inputs at best values → score = 1.0."""
        s = score_strategy([], 0.0, 0.0)
        assert s == pytest.approx(1.0)

    def test_all_max_penalty(self):
        """All inputs at worst values → score = 0.0."""
        s = score_strategy(["f"] * 100, 1.0, 1.0)
        assert s == pytest.approx(0.0)

    def test_halfway_scores(self):
        """Halfway inputs produce proportional score."""
        # files=25, max=50 → token=0.5; collision=0.5; blast=0.5
        # score = 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
        s = score_strategy(["f"] * 25, 0.5, 0.5)
        assert s == pytest.approx(0.5)

    def test_single_file_ideal_conditions(self):
        """Single file, no collision, no blast → high score."""
        cfg = WorkflowConfig(max_blast_files=50)
        s = score_strategy(["one_file.txt"], 0.0, 0.0, config=cfg)
        # token = 1 - 1/50 = 0.98
        # score = 0.4*0.98 + 0.3*1 + 0.3*1 = 0.992
        assert s == pytest.approx(0.992)

    def test_many_files_one_over_limit(self):
        """One file over max_blast_files → token_depth = 0."""
        cfg = WorkflowConfig(max_blast_files=10)
        s = score_strategy(["f"] * 11, 0.0, 0.0, config=cfg)
        # token = max(0, 1-11/10) = 0
        # score = 0.4*0 + 0.3*1 + 0.3*1 = 0.6
        assert s == pytest.approx(0.6)


class TestPruneStrategiesEdgeCases:

    def test_empty_list(self):
        cfg = WorkflowConfig()
        assert prune_strategies([], config=cfg) == []

    def test_all_below_threshold(self):
        cfg = WorkflowConfig(min_strategy_score=0.9)
        r = prune_strategies([_c("a", 0.5), _c("b", 0.7)], config=cfg)
        assert r == []

    def test_all_above_threshold(self):
        cfg = WorkflowConfig(min_strategy_score=0.1)
        r = prune_strategies([_c("a", 0.5), _c("b", 0.7)], config=cfg)
        assert len(r) == 2

    def test_exact_threshold_boundary(self):
        """Score exactly at threshold is included."""
        cfg = WorkflowConfig(min_strategy_score=0.5)
        r = prune_strategies([_c("a", 0.5), _c("b", 0.49)], config=cfg)
        assert len(r) == 1
        assert r[0]["strategy_id"] == "a"

    def test_default_min_score_is_0_3(self):
        """Without config, min_strategy_score defaults to 0.3."""
        r = prune_strategies([_c("a", 0.29), _c("b", 0.31)])
        assert len(r) == 1
        assert r[0]["strategy_id"] == "b"

    def test_default_max_strategies_is_3(self):
        """Without config, max_strategies defaults to 3."""
        cfg = WorkflowConfig(min_strategy_score=0.0)
        r = prune_strategies([_c(str(i), float(i + 1)) for i in range(10)], config=cfg)
        assert len(r) == 3


class TestDetectCollisionPairsEdgeCases:

    def test_single_strategy(self):
        a = _c("a", 1.0, ["X.java"])
        assert detect_collision_pairs([a]) == []

    def test_three_strategies_one_shared(self):
        a = _c("a", 1.0, ["Shared.java", "A.java"])
        b = _c("b", 1.0, ["Shared.java", "B.java"])
        c = _c("c", 1.0, ["C.java"])
        pairs = detect_collision_pairs([a, b, c])
        assert len(pairs) == 1
        assert pairs[0][0] == "a"
        assert pairs[0][1] == "b"

    def test_three_strategies_all_shared(self):
        a = _c("a", 1.0, ["Shared.java"])
        b = _c("b", 1.0, ["Shared.java"])
        c = _c("c", 1.0, ["Shared.java"])
        pairs = detect_collision_pairs([a, b, c])
        assert len(pairs) == 3  # (a,b), (a,c), (b,c)

    def test_shared_files_sorted(self):
        a = _c("a", 1.0, ["Zebra.java", "Alpha.java"])
        b = _c("b", 1.0, ["Zebra.java", "Alpha.java"])
        pairs = detect_collision_pairs([a, b])
        assert pairs[0][2] == ["Alpha.java", "Zebra.java"]

    def test_no_shared_files(self):
        a = _c("a", 1.0, ["A.java"])
        b = _c("b", 1.0, ["B.java"])
        c = _c("c", 1.0, ["C.java"])
        assert detect_collision_pairs([a, b, c]) == []
