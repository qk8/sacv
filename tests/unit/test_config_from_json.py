"""Unit tests for WorkflowConfig.from_json() field propagation."""
from __future__ import annotations

import json
import pathlib
import pytest

from sacv.orchestration.config import (
    WorkflowConfig, IterationLimits, StagnationConfig,
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
        "iteration_limits": {
            "implement_loop": 200,
            "clarify_round": 10,
            "spec_audit": 5,
            "plan_review": 5,
        },
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
    assert cfg.iteration_limits == IterationLimits(200, 10, 5, 5)
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


def test_from_json_defaults_iteration_limits(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.iteration_limits == IterationLimits()


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
        "iteration_limits": {"implement_loop": 50},
        "token_budget": {"cost_per_m_input": 1.0},
    }))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.iteration_limits.implement_loop == 50
    assert cfg.iteration_limits.clarify_round == 5  # default
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


def test_from_json_invalid_json_raises(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        WorkflowConfig.from_json(cfg_file)
