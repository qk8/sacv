"""
tests/unit/test_confidence_weights.py
======================================
HIGH-03: ConfidenceWeights dataclass and wiring into compute_confidence_score.

Verifies:
  1. ConfidenceWeights dataclass exists with correct defaults
  2. WorkflowConfig.confidence_weights field exists
  3. from_json() reads confidence_weights from config file
  4. compute_confidence_score uses config weights instead of hardcoded values
"""
from __future__ import annotations

import json
import pathlib

import pytest

from sacv.orchestration.config import WorkflowConfig, ConfidenceWeights
from sacv.orchestration.edges import compute_confidence_score


# ── Config field tests ────────────────────────────────────────────────────────


class TestConfidenceWeightsConfig:
    """Verify ConfidenceWeights defaults and from_json loading."""

    def test_defaults(self) -> None:
        """ConfidenceWeights defaults match original hardcoded values."""
        cw = ConfidenceWeights()
        assert cw.stagnation_penalty == 0.40
        assert cw.blast_penalty_scale == 0.30
        assert cw.critic_penalty_per_crit == 0.10
        assert cw.max_critic_penalty == 0.30

    def test_workflow_config_has_confidence_weights(self) -> None:
        """WorkflowConfig includes confidence_weights sub-config."""
        cfg = WorkflowConfig()
        assert isinstance(cfg.confidence_weights, ConfidenceWeights)
        assert cfg.confidence_weights.stagnation_penalty == 0.40

    def test_from_json_reads_confidence_weights(self, tmp_path: pathlib.Path) -> None:
        """from_json() reads confidence_weights from config file."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "confidence_weights": {
                "stagnation_penalty": 0.50,
                "blast_penalty_scale": 0.25,
                "critic_penalty_per_crit": 0.05,
                "max_critic_penalty": 0.20,
            },
        }))
        cfg = WorkflowConfig.from_json(cfg_file)
        assert cfg.confidence_weights.stagnation_penalty == 0.50
        assert cfg.confidence_weights.blast_penalty_scale == 0.25
        assert cfg.confidence_weights.critic_penalty_per_crit == 0.05
        assert cfg.confidence_weights.max_critic_penalty == 0.20

    def test_from_json_default_when_absent(self, tmp_path: pathlib.Path) -> None:
        """When absent, confidence_weights falls back to defaults."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("{}")
        cfg = WorkflowConfig.from_json(cfg_file)
        assert cfg.confidence_weights.stagnation_penalty == 0.40
        assert cfg.confidence_weights.blast_penalty_scale == 0.30


# ── Edge routing tests ────────────────────────────────────────────────────────


def _s(**kw):
    base = {
        "session_id": "t", "task_id": "t", "task_description": "",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "context_skeleton": None, "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None, "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [], "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
    }
    base.update(kw)
    return base


class TestConfidenceScoreUsesConfigWeights:
    """compute_confidence_score must use config weights, not hardcoded values."""

    def test_stagnation_penalty_from_config(self):
        """stagnation_penalty is read from config, not hardcoded to 0.40."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            confidence_weights=ConfidenceWeights(stagnation_penalty=0.60),
        )
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "semantic"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=0.0,
        )
        # With 0.60 stagnation_penalty: 1.0 - 0.60 = 0.40
        assert compute_confidence_score(s, cfg) == pytest.approx(0.40, abs=1e-9)

    def test_blast_penalty_from_config(self):
        """blast_penalty_scale is read from config, not hardcoded to 0.30."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            confidence_weights=ConfidenceWeights(blast_penalty_scale=0.50),
        )
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map={"risk_score": 0.6},
            critic_findings=[], cumulative_cost_dollars=0.0,
        )
        # blast_penalty = 0.6 * 0.50 = 0.30
        assert compute_confidence_score(s, cfg) == pytest.approx(0.70, abs=1e-9)

    def test_critic_penalty_from_config(self):
        """critic_penalty_per_crit and max_critic_penalty are read from config."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            confidence_weights=ConfidenceWeights(
                critic_penalty_per_crit=0.15,
                max_critic_penalty=0.25,
            ),
        )
        findings = [
            {"severity": "critical", "critic": "s", "message": "m",
             "file": "a.java", "line": 1, "rule_id": "r1", "resolution_hint": "fix"},
            {"severity": "critical", "critic": "s", "message": "m",
             "file": "b.java", "line": 1, "rule_id": "r2", "resolution_hint": "fix"},
            {"severity": "critical", "critic": "s", "message": "m",
             "file": "c.java", "line": 1, "rule_id": "r3", "resolution_hint": "fix"},
        ]
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=findings,
            cumulative_cost_dollars=0.0,
        )
        # critic_penalty = min(0.25, 3 * 0.15) = 0.25 (capped)
        assert compute_confidence_score(s, cfg) == pytest.approx(0.75, abs=1e-9)

    def test_default_weights_match_original_hardcoded_values(self):
        """Default weights produce the same results as the original hardcoded values."""
        cfg = WorkflowConfig(max_self_correction_cycles=10)
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "semantic"},
            blast_radius_map={"risk_score": 0.8},
            critic_findings=[
                {"severity": "critical", "critic": "s", "message": "m",
                 "file": "a.java", "line": 1, "rule_id": "r1", "resolution_hint": "fix"},
                {"severity": "warning", "critic": "style", "message": "z",
                 "file": "b.java", "line": 1, "rule_id": "r2", "resolution_hint": "fix"},
            ],
            cumulative_cost_dollars=0.0,
        )
        # stagnation=0.40, blast=0.8*0.30=0.24, critic=min(0.30, 1*0.10)=0.10
        # total = 0.40 + 0.24 + 0.10 = 0.74
        assert compute_confidence_score(s, cfg) == pytest.approx(0.26, abs=1e-9)
