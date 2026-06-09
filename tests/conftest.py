"""
pytest configuration: marks and shared fixtures.
"""
from __future__ import annotations

import pytest

# Configure structlog once for all tests so log output is consistent.
from sacv.logging_config import configure_logging  # noqa: E402
configure_logging()  # LOG_FORMAT defaults to "json" — fine for pytest output capture

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, ProjectMode
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests; no I/O")
    config.addinivalue_line("markers", "integration: requires stub providers; no live APIs")
    config.addinivalue_line("markers", "e2e: full graph via VCR cassettes; no live APIs")


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def workflow_config():
    """Provide a default WorkflowConfig."""
    return WorkflowConfig()


@pytest.fixture
def base_state():
    """Return a minimal base state dict for unit tests.

    Override via kwarg: base_state(task_id="custom-id")
    """
    def _make(**kw):
        state = {
            "session_id":             "sess-test",
            "task_id":                "task-unit-001",
            "project_mode":           "greenfield",
            "module_type":            "backend-domain",
            "current_phase":          WorkflowPhase.BOOTSTRAP.value,
            "task_description":       "Add test feature",
            "context_skeleton":       None,
            "blast_radius_map":       None,
            "agents_md_context":      None,
            "strategy_candidates":    [],
            "selected_strategy":      None,
            "pruned_strategies":      [],
            "red_phase_evidence_path": None,
            "test_inventory_paths":   [],
            "tdd_gate_attempts":      0,
            "diff_proposal":          None,
            "preflight_result":       None,
            "critic_findings":        [],
            "verifier_verdict":       None,
            "debug_observations":     None,
            "correction_state": {
                "attempt_count": 0,
                "branch_name": None,
                "last_error_hash": None,
                "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score":       1.0,
            "replan_count":           0,
            "active_branches":        [],
            "exhausted_branches":     [],
            "speculative_stash_ref":  None,
            "escalation_payload":     None,
            "procedural_constraints": [],
            "lesson_learned":         None,
            "arch_rules_updated":     False,
            "check_profile":          "standard",
            "cumulative_cost_dollars": 0.0,
            "skip_tdd_gate":          False,
        }
        state.update(kw)
        return state
    return _make


@pytest.fixture
def base_deps():
    """Return minimal NodeDeps with stub providers.

    Override via kwarg: base_deps(agent=custom_agent)
    """
    def _make(**kw):
        from sacv.orchestration.deps import NodeDeps
        defaults = {
            "agent": StubAgentProvider(),
            "memory": StubMemoryProvider(),
            "code_graph": StubCodeGraphProvider(),
            "cross_domain": StubCrossDomainProvider(),
            "git": StubGitProvider(),
            "sandbox": StubSandboxProvider(),
            "diff": StubDiffProvider(),
            "config": WorkflowConfig(),
        }
        defaults.update(kw)
        return NodeDeps(**defaults)
    return _make
