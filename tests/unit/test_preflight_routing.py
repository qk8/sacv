"""
tests/unit/test_preflight_routing.py
=====================================
Unit tests for route_after_preflight — pure function, no I/O.

After Issue 3 fix, route_after_preflight returns a single string:
  - "all_critics_router" when preflight is clean (no LSP errors, no arch violations)
  - "actor" when preflight reports violations (needs re-implementation)
"""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import route_after_preflight


def _s(**kw):
    base = {
        "session_id":"t", "task_id":"t", "task_description":"",
        "project_mode":"greenfield", "module_type":"backend-domain",
        "context_skeleton":None, "blast_radius_map":None, "agents_md_context":None,
        "strategy_candidates":[], "selected_strategy":None, "pruned_strategies":[],
        "red_phase_evidence_path":None, "test_inventory_paths":[],
        "diff_proposal":None, "preflight_result":None,
        "critic_findings":[], "verifier_verdict":None,
        "correction_state":{"attempt_count":1,"branch_name":"b",
                            "last_error_hash":None,"error_history":[],"stagnation_pattern":"none"},
        "confidence_score":1.0, "replan_count":0,
        "active_branches":[], "exhausted_branches":[], "escalation_payload":None,
        "procedural_constraints":[], "lesson_learned":None, "arch_rules_updated":False,
    }
    base.update(kw)
    return base


class TestRouteAfterPreflight:

    def test_clean_preflight_routes_to_all_critics(self):
        """Clean preflight → route to all_critics_router node."""
        state = _s(preflight_result={"passed":True,"lsp_errors":[],"arch_violations":[],"cross_stack_errors":[],"duration_ms":120})
        result = route_after_preflight(state)
        assert result == "all_critics_router"

    def test_lsp_error_routes_back_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors": [{"file":"A.ts","line":5,"code":"TS2322","message":"Type mismatch"}],
            "arch_violations": [],
            "cross_stack_errors": [],
            "duration_ms": 50,
        })
        assert route_after_preflight(state) == "actor"

    def test_arch_violation_routes_back_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [{"rule":"no-ui-to-db","source_file":"ui/A.ts","target_file":"db/B.ts","message":"forbidden import"}],
            "cross_stack_errors": [],
            "duration_ms": 30,
        })
        assert route_after_preflight(state) == "actor"

    def test_both_violations_routes_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors":      [{"file":"X.java","line":1,"code":"CE","message":"cannot find symbol"}],
            "arch_violations": [{"rule":"domain-isolation","source_file":"domain/X.java","target_file":"infra/Y.java","message":"layer violation"}],
            "cross_stack_errors": [],
            "duration_ms": 80,
        })
        assert route_after_preflight(state) == "actor"

    def test_cross_stack_error_routes_to_actor(self):
        """Cross-stack type errors also route back to actor."""
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [],
            "cross_stack_errors": [{"file":"frontend/api/types.ts","line":10,"code":"TS2345","message":"Type mismatch"}],
            "duration_ms": 40,
        })
        assert route_after_preflight(state) == "actor"

    def test_none_preflight_result_routes_to_all_critics(self):
        """No preflight yet (e.g. first pass) → treat as clean → all_critics_router."""
        state = _s(preflight_result=None)
        result = route_after_preflight(state)
        assert result == "all_critics_router"

    def test_empty_preflight_result_routes_to_all_critics(self):
        """Empty dict (missing 'passed' key) defaults to clean."""
        state = _s(preflight_result={})
        result = route_after_preflight(state)
        assert result == "all_critics_router"

    def test_returns_string_not_list(self):
        """After Issue 3 fix, route_after_preflight returns a string, not list[Send]."""
        state = _s(preflight_result={"passed":True,"lsp_errors":[],"arch_violations":[],"cross_stack_errors":[],"duration_ms":10})
        result = route_after_preflight(state)
        assert isinstance(result, str)
        assert result in ("all_critics_router", "actor")
