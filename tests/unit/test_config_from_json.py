"""Unit tests for WorkflowConfig.from_json() field propagation."""
from __future__ import annotations

import json
import pathlib
import pytest

from sacv.orchestration.config import (
    WorkflowConfig, StagnationConfig,
    TokenBudget, CadenceConfig, DebugConfig,
)


# ── Happy path: all fields ────────────────────────────────────────────────────


def test_from_json_all_fields(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "max_self_correction_cycles": 5,
        "confidence_escalation_threshold": 0.5,
        "max_replan_attempts": 2,
        "max_tdd_gate_attempts": 5,
        "max_parallel_branches": 4,
        "max_parallel_critics": 3,
        "min_strategy_score": 0.4,
        "max_strategies": 5,
        "max_blast_files": 100,
        "monorepo_mode": True,
        "agents_md_prompt_chars": 3000,
        "stagnation": {
            "total_abort_force": 5,
            "drift_revision_limit": 3,
            "semantic_similarity_threshold": 0.9,
        },
        "token_budget": {
            "cost_per_m_input": 10.0,
            "cost_per_m_output": 60.0,
            "critical_dollar": 150.0,
            "warning_dollar": 100.0,
        },
        "cadence": {
            "cleanup_interval": 50,
            "llm_quality_interval": 20,
            "drift_check_interval": {"simple": 30, "medium": 25, "complex": 15},
        },
        "debug": {
            "user_java_package": "com.myapp",
            "user_ts_src_root": "app",
            "jdwp_port": 6006,
            "cdp_port": 9339,
            "debug_timeout_sec": 60,
            "max_debug_steps": 20,
            "actuator_base_url": "http://api:8080/actuator",
            "openapi_spec_path": "spec/openapi.yaml",
            "otel_query_url": "http://jaeger:16686/api/traces",
        },
    }))
    cfg = WorkflowConfig.from_json(cfg_file)

    assert cfg.max_self_correction_cycles == 5
    assert cfg.confidence_escalation_threshold == 0.5
    assert cfg.max_replan_attempts == 2
    assert cfg.max_tdd_gate_attempts == 5
    assert cfg.max_parallel_branches == 4
    assert cfg.max_parallel_critics == 3
    assert cfg.min_strategy_score == 0.4
    assert cfg.max_strategies == 5
    assert cfg.max_blast_files == 100
    assert cfg.monorepo_mode is True
    assert cfg.agents_md_prompt_chars == 3000
    assert cfg.stagnation == StagnationConfig(5, 3, 0.9)
    assert cfg.token_budget == TokenBudget(10.0, 60.0, 150.0, 100.0)
    assert cfg.cadence == CadenceConfig(50, 20, {"simple": 30, "medium": 25, "complex": 15})
    assert cfg.debug == DebugConfig(
        user_java_package="com.myapp", user_ts_src_root="app",
        jdwp_port=6006, cdp_port=9339, debug_timeout_sec=60,
        max_debug_steps=20, actuator_base_url="http://api:8080/actuator",
        openapi_spec_path="spec/openapi.yaml", otel_query_url="http://jaeger:16686/api/traces",
    )


# ── Default values ────────────────────────────────────────────────────────────


def test_from_json_default_agents_md_prompt_chars(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.agents_md_prompt_chars == 2_000


def test_from_json_defaults_for_all_top_level(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.max_self_correction_cycles == 3
    assert cfg.confidence_escalation_threshold == 0.25
    assert cfg.max_replan_attempts == 1
    assert cfg.max_tdd_gate_attempts == 3
    assert cfg.max_parallel_branches == 2
    assert cfg.max_parallel_critics == 2
    assert cfg.min_strategy_score == 0.3
    assert cfg.max_strategies == 3
    assert cfg.max_blast_files == 50
    assert cfg.monorepo_mode is False


def test_from_json_defaults_stagnation(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.stagnation == StagnationConfig()


def test_from_json_defaults_token_budget(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.token_budget == TokenBudget()


def test_from_json_defaults_cadence(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.cadence == CadenceConfig()


def test_from_json_defaults_debug(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.debug == DebugConfig()


# ── Nested field overrides ────────────────────────────────────────────────────


def test_from_json_partial_nested_overrides(tmp_path: pathlib.Path) -> None:
    """Only some nested fields provided — rest use defaults."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "token_budget": {"cost_per_m_input": 1.0},
    }))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.token_budget.cost_per_m_input == 1.0
    assert cfg.token_budget.cost_per_m_output == 30.0  # default


def test_from_json_stagnation_backfill_for_self_correction(tmp_path: pathlib.Path) -> None:
    """When max_self_correction_cycles not provided, falls back to stagnation.total_abort_force."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "stagnation": {"total_abort_force": 7},
    }))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.max_self_correction_cycles == 7


def test_from_json_top_level_overrides_stagnation(tmp_path: pathlib.Path) -> None:
    """Explicit top-level max_self_correction_cycles takes priority over stagnation fallback."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "max_self_correction_cycles": 10,
        "stagnation": {"total_abort_force": 7},
    }))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.max_self_correction_cycles == 10


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_from_json_empty_object(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert isinstance(cfg, WorkflowConfig)


def test_from_json_string_path(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(str(cfg_file))
    assert cfg is not None


def test_from_json_pathlib_path(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert isinstance(cfg, WorkflowConfig)


def test_from_json_missing_file_raises(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        WorkflowConfig.from_json(cfg_file)


def test_max_empty_diff_retries_is_read_from_json(tmp_path: pathlib.Path) -> None:
    """Regression: max_empty_diff_retries was in KNOWN_TOP_LEVEL_KEYS but not in from_json()."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"max_empty_diff_retries": 7}))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.max_empty_diff_retries == 7


def test_max_empty_diff_retries_default_when_absent(tmp_path: pathlib.Path) -> None:
    """When not provided, max_empty_diff_retries falls back to the dataclass default of 3."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.max_empty_diff_retries == 3


def test_from_json_invalid_json_raises(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        WorkflowConfig.from_json(cfg_file)


# ── CFG-001: Config value range validation ────────────────────────────────────


class TestConfigValidation:

    def test_rejects_zero_correction_cycles(self):
        with pytest.raises(ValueError, match="max_self_correction_cycles"):
            WorkflowConfig(max_self_correction_cycles=0)

    def test_rejects_negative_correction_cycles(self):
        with pytest.raises(ValueError, match="max_self_correction_cycles"):
            WorkflowConfig(max_self_correction_cycles=-1)

    def test_accepts_one_correction_cycle(self):
        # 1 is the minimum valid value
        cfg = WorkflowConfig(max_self_correction_cycles=1)
        assert cfg.max_self_correction_cycles == 1

    def test_rejects_confidence_threshold_zero(self):
        with pytest.raises(ValueError, match="confidence_escalation_threshold"):
            WorkflowConfig(confidence_escalation_threshold=0.0)

    def test_rejects_confidence_threshold_above_one(self):
        with pytest.raises(ValueError, match="confidence_escalation_threshold"):
            WorkflowConfig(confidence_escalation_threshold=1.5)

    def test_accepts_confidence_threshold_one(self):
        cfg = WorkflowConfig(confidence_escalation_threshold=1.0)
        assert cfg.confidence_escalation_threshold == 1.0

    def test_rejects_negative_replan_attempts(self):
        with pytest.raises(ValueError, match="max_replan_attempts"):
            WorkflowConfig(max_replan_attempts=-1)

    def test_accepts_zero_replan_attempts(self):
        cfg = WorkflowConfig(max_replan_attempts=0)
        assert cfg.max_replan_attempts == 0

    def test_rejects_zero_tdd_gate_attempts(self):
        with pytest.raises(ValueError, match="max_tdd_gate_attempts"):
            WorkflowConfig(max_tdd_gate_attempts=0)

    def test_rejects_zero_empty_diff_retries(self):
        with pytest.raises(ValueError, match="max_empty_diff_retries"):
            WorkflowConfig(max_empty_diff_retries=0)

    def test_rejects_inverted_budget(self):
        with pytest.raises(ValueError, match="warning_dollar"):
            WorkflowConfig(token_budget=TokenBudget(warning_dollar=100.0, critical_dollar=50.0))

    def test_rejects_equal_budget_values(self):
        with pytest.raises(ValueError, match="warning_dollar"):
            WorkflowConfig(token_budget=TokenBudget(warning_dollar=80.0, critical_dollar=80.0))

    def test_accepts_valid_budget(self):
        cfg = WorkflowConfig(token_budget=TokenBudget(warning_dollar=50.0, critical_dollar=80.0))
        assert cfg.token_budget.warning_dollar < cfg.token_budget.critical_dollar

    def test_rejects_zero_critical_dollar(self):
        with pytest.raises(ValueError, match="critical_dollar"):
            WorkflowConfig(token_budget=TokenBudget(critical_dollar=0.0))

    def test_rejects_excessive_critical_dollar(self):
        with pytest.raises(ValueError, match="critical_dollar"):
            WorkflowConfig(token_budget=TokenBudget(critical_dollar=1001.0))

    def test_rejects_low_similarity_threshold(self):
        with pytest.raises(ValueError, match="semantic_similarity_threshold"):
            WorkflowConfig(stagnation=StagnationConfig(semantic_similarity_threshold=0.0))

    def test_rejects_similarity_threshold_above_one(self):
        with pytest.raises(ValueError, match="semantic_similarity_threshold"):
            WorkflowConfig(stagnation=StagnationConfig(semantic_similarity_threshold=1.1))

    def test_accepts_boundary_similarity_threshold(self):
        cfg = WorkflowConfig(stagnation=StagnationConfig(semantic_similarity_threshold=0.5))
        assert cfg.stagnation.semantic_similarity_threshold == 0.5

    def test_rejects_negative_min_strategy_score(self):
        with pytest.raises(ValueError, match="min_strategy_score"):
            WorkflowConfig(min_strategy_score=-0.1)

    def test_rejects_min_strategy_score_one(self):
        with pytest.raises(ValueError, match="min_strategy_score"):
            WorkflowConfig(min_strategy_score=1.0)

    def test_accepts_boundary_min_strategy_score(self):
        cfg = WorkflowConfig(min_strategy_score=0.0)
        assert cfg.min_strategy_score == 0.0

    def test_default_config_passes_validation(self):
        """Default WorkflowConfig must pass all validation checks."""
        cfg = WorkflowConfig()
        assert cfg.max_self_correction_cycles == 3
        assert 0.0 < cfg.confidence_escalation_threshold <= 1.0
