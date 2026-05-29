"""
tests/unit/test_preflight_routing.py
=====================================
Unit tests for route_after_preflight — pure function, no I/O.
Validates LSP error and arch violation routing decisions.
"""
from __future__ import annotations
import pytest
from langgraph.types import Send
from sacv.orchestration.edges import route_after_preflight
from sacv.orchestration.state import WorkflowPhase


def _s(**kw):
    base = {
        "session_id":"t", "task_id":"t", "task_description":"",
        "project_mode":"greenfield", "module_type":"backend-domain",
        "check_profile":"standard", "current_phase":WorkflowPhase.PREFLIGHT.value,
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

    def test_clean_preflight_fans_out_to_critics(self):
        state  = _s(preflight_result={"passed":True,"lsp_errors":[],"arch_violations":[],"duration_ms":120})
        result = route_after_preflight(state)
        assert isinstance(result, list)
        node_names = [send.node for send in result]
        assert set(node_names) == {"security_critic", "style_critic", "consistency_critic"}

    def test_lsp_error_routes_back_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors": [{"file":"A.ts","line":5,"code":"TS2322","message":"Type mismatch"}],
            "arch_violations": [],
            "duration_ms": 50,
        })
        assert route_after_preflight(state) == "actor"

    def test_arch_violation_routes_back_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors": [],
            "arch_violations": [{"rule":"no-ui-to-db","source_file":"ui/A.ts","target_file":"db/B.ts","message":"forbidden import"}],
            "duration_ms": 30,
        })
        assert route_after_preflight(state) == "actor"

    def test_both_violations_routes_to_actor(self):
        state = _s(preflight_result={
            "passed": False,
            "lsp_errors":      [{"file":"X.java","line":1,"code":"CE","message":"cannot find symbol"}],
            "arch_violations": [{"rule":"domain-isolation","source_file":"domain/X.java","target_file":"infra/Y.java","message":"layer violation"}],
            "duration_ms": 80,
        })
        assert route_after_preflight(state) == "actor"

    def test_none_preflight_result_fans_out(self):
        """No preflight yet (e.g. first pass) → treat as clean → fan out."""
        state = _s(preflight_result=None)
        result = route_after_preflight(state)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_fan_out_produces_exactly_three_sends(self):
        state  = _s(preflight_result={"passed":True,"lsp_errors":[],"arch_violations":[],"duration_ms":10})
        result = route_after_preflight(state)
        assert len(result) == 3
        assert all(isinstance(s, Send) for s in result)

    def test_empty_preflight_result_fans_out(self):
        """Empty dict (missing 'passed' key) defaults to clean."""
        state  = _s(preflight_result={})
        result = route_after_preflight(state)
        assert isinstance(result, list)
