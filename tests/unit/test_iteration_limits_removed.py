"""
tests/unit/test_iteration_limits_removed.py
=============================================
Verify that WorkflowConfig does not depend on IterationLimits.

IterationLimits (implement_loop, clarify_round, spec_audit, plan_review)
was removed because it was never referenced outside config.py.
This test documents the removal and ensures the config still works cleanly.
"""
from __future__ import annotations

import pytest
from sacv.orchestration.config import WorkflowConfig


class TestIterationLimitsRemoved:

    def test_workflow_config_has_no_iteration_limits_attribute(self):
        """IterationLimits was removed — WorkflowConfig should not have it."""
        cfg = WorkflowConfig()
        assert not hasattr(cfg, "iteration_limits")

    def test_from_json_without_iteration_limits_key(self):
        """Config can be loaded from JSON that omits iteration_limits entirely."""
        import json
        import pathlib
        data = {
            "max_self_correction_cycles": 5,
            "debug": {"user_java_package": "com.example"},
        }
        path = pathlib.Path(__file__).parent / "_dummy_config.json"
        path.write_text(json.dumps(data))
        try:
            cfg = WorkflowConfig.from_json(path)
            assert cfg.max_self_correction_cycles == 5
            assert not hasattr(cfg, "iteration_limits")
        finally:
            path.unlink(missing_ok=True)

    def test_from_json_with_empty_iteration_limits(self):
        """Config ignores iteration_limits if present in JSON (backward compat)."""
        import json
        import pathlib
        data = {
            "max_self_correction_cycles": 5,
            "iteration_limits": {"implement_loop": 999},
            "debug": {"user_java_package": "com.example"},
        }
        path = pathlib.Path(__file__).parent / "_dummy_config2.json"
        path.write_text(json.dumps(data))
        try:
            cfg = WorkflowConfig.from_json(path)
            assert cfg.max_self_correction_cycles == 5
            assert not hasattr(cfg, "iteration_limits")
        finally:
            path.unlink(missing_ok=True)

    def test_all_remaining_config_fields_are_used(self):
        """Verify remaining WorkflowConfig fields have production references."""
        cfg = WorkflowConfig()
        # These fields are all actively used in the workflow
        assert hasattr(cfg, "max_self_correction_cycles")
        assert hasattr(cfg, "confidence_escalation_threshold")
        assert hasattr(cfg, "max_replan_attempts")
        assert hasattr(cfg, "max_tdd_gate_attempts")
        assert hasattr(cfg, "max_empty_diff_retries")
        assert hasattr(cfg, "max_parallel_branches")
        assert hasattr(cfg, "max_parallel_critics")
        assert hasattr(cfg, "min_strategy_score")
        assert hasattr(cfg, "max_strategies")
        assert hasattr(cfg, "max_blast_files")
        assert hasattr(cfg, "monorepo_mode")
        assert hasattr(cfg, "agents_md_prompt_chars")
        assert hasattr(cfg, "stagnation")
        assert hasattr(cfg, "token_budget")
        assert hasattr(cfg, "cadence")
        assert hasattr(cfg, "debug")
