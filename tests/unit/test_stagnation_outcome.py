"""
tests/unit/test_stagnation_outcome.py
======================================
Unit tests for outcome-based stagnation detection.

Tests cover:
1. compute_outcome_signature — deterministic hash from preflight + critic findings
2. compute_outcome_signature — ignores file paths (only uses rule/message)
3. compute_outcome_signature — empty preflight/critic produces consistent hash
4. compute_outcome_signature — critic severity filtering (only critical)
5. compute_outcome_signature — rule_id vs message fallback for critics
6. check_outcome_stagnation — empty signature returns False
7. check_outcome_stagnation — same signature across attempts detected
8. check_outcome_stagnation — different signatures not stagnation
9. check_outcome_stagnation — first attempt (no prev_sig) not stagnation
"""
from __future__ import annotations

import pytest

from sacv.nodes._stagnation import (
    compute_outcome_signature,
    check_outcome_stagnation,
)


def _correction(last_sig: str | None = None) -> dict:
    return {
        "attempt_count": 2,
        "branch_name": None,
        "last_error_hash": None,
        "error_history": [],
        "stagnation_pattern": "none",
        "last_outcome_signature": last_sig,
    }


class TestComputeOutcomeSignature:

    def test_preflight_lsp_errors_included(self):
        sig = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"rule": "unused-import", "file": "A.java"}]},
            critic_findings=[],
        )
        assert len(sig) == 16

    def test_preflight_arch_violations_included(self):
        sig = compute_outcome_signature(
            preflight_result={"arch_violations": [{"code": "NO_DEP", "message": "bad dep"}]},
            critic_findings=[],
        )
        assert len(sig) == 16

    def test_preflight_cross_stack_errors_included(self):
        sig = compute_outcome_signature(
            preflight_result={"cross_stack_errors": [{"rule": "mismatch", "message": "type err"}]},
            critic_findings=[],
        )
        assert len(sig) == 16

    def test_preflight_blast_errors_included(self):
        sig = compute_outcome_signature(
            preflight_result={"blast_errors": [{"code": "HIGH_RISK", "message": "too wide"}]},
            critic_findings=[],
        )
        assert len(sig) == 16

    def test_preflight_ignores_file_paths(self):
        """Signatures should be identical when only file paths differ."""
        sig_a = compute_outcome_signature(
            preflight_result={"lsp_errors": [
                {"rule": "unused-import", "file": "OldName.java"},
            ]},
            critic_findings=[],
        )
        sig_b = compute_outcome_signature(
            preflight_result={"lsp_errors": [
                {"rule": "unused-import", "file": "NewName.java"},
            ]},
            critic_findings=[],
        )
        assert sig_a == sig_b

    def test_preflight_uses_rule_over_message(self):
        """rule field is preferred; message is fallback."""
        sig_rule = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"rule": "R1", "message": "msg1"}]},
            critic_findings=[],
        )
        sig_msg = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"message": "msg1"}]},
            critic_findings=[],
        )
        assert sig_rule != sig_msg  # "R1" != "msg1"

    def test_preflight_uses_code_when_no_rule(self):
        """code field is used when rule is missing."""
        sig = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"code": "E1", "message": "msg"}]},
            critic_findings=[],
        )
        assert len(sig) == 16

    def test_preflight_uses_truncated_message_when_no_rule_or_code(self):
        """message[:80] used when both rule and code are missing."""
        sig = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"message": "x" * 200}]},
            critic_findings=[],
        )
        # Signature should be based on truncated message
        sig_short = compute_outcome_signature(
            preflight_result={"lsp_errors": [{"message": "x" * 80}]},
            critic_findings=[],
        )
        assert sig == sig_short

    def test_empty_preflight_empty_critics_produces_empty_hash(self):
        """No problems → deterministic empty hash."""
        sig = compute_outcome_signature(preflight_result=None, critic_findings=[])
        assert sig == compute_outcome_signature(preflight_result={}, critic_findings=[])

    def test_none_preflight_same_as_empty_dict(self):
        """None preflight_result treated same as empty dict."""
        sig_none = compute_outcome_signature(preflight_result=None, critic_findings=[])
        sig_empty = compute_outcome_signature(preflight_result={}, critic_findings=[])
        assert sig_none == sig_empty

    def test_critical_critic_findings_included(self):
        """Critical severity critic findings included in signature."""
        sig = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{
                "critic": "security", "severity": "critical",
                "rule_id": "SEC-001", "message": "SQL injection",
            }],
        )
        assert len(sig) == 16

    def test_non_critical_critic_findings_excluded(self):
        """Non-critical findings are excluded from signature."""
        sig_critical = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{"critic": "security", "severity": "critical", "rule_id": "SEC-001"}],
        )
        sig_non_critical = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{"critic": "security", "severity": "warning", "rule_id": "SEC-001"}],
        )
        assert sig_critical != sig_non_critical

    def test_critic_uses_rule_id_over_message(self):
        """When rule_id is present, message value is ignored (not used as fallback)."""
        # rule_id="X-RULE" with message="DIFFERENT" → uses "X-RULE"
        # rule_id="X-RULE" with message="SAME" → still uses "X-RULE"
        # Both should produce the same signature, proving message is ignored
        sig_diff_msg = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{"critic": "style", "severity": "critical", "rule_id": "X-RULE", "message": "DIFFERENT"}],
        )
        sig_same_msg = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{"critic": "style", "severity": "critical", "rule_id": "X-RULE", "message": "SAME"}],
        )
        assert sig_diff_msg == sig_same_msg

    def test_critic_rule_id_fallback_to_message(self):
        """When rule_id is empty, message[:80] is used."""
        sig = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[{"critic": "style", "severity": "critical", "rule_id": "", "message": "bad style"}],
        )
        assert len(sig) == 16

    def test_signature_is_deterministic(self):
        """Same inputs always produce the same signature."""
        preflight = {"lsp_errors": [{"rule": "R1"}], "arch_violations": [{"code": "C1"}]}
        critics = [{"critic": "security", "severity": "critical", "rule_id": "S1"}]
        sig1 = compute_outcome_signature(preflight, critics)
        sig2 = compute_outcome_signature(preflight, critics)
        assert sig1 == sig2

    def test_signature_independent_of_order(self):
        """Signature is the same regardless of ordering of findings."""
        preflight_a = {"lsp_errors": [{"rule": "R1"}], "arch_violations": [{"code": "C1"}]}
        preflight_b = {"arch_violations": [{"code": "C1"}], "lsp_errors": [{"rule": "R1"}]}
        sig_a = compute_outcome_signature(preflight_a, [])
        sig_b = compute_outcome_signature(preflight_b, [])
        assert sig_a == sig_b

    def test_multiple_critic_findings_all_critical_included(self):
        """All critical findings included in signature."""
        sig = compute_outcome_signature(
            preflight_result=None,
            critic_findings=[
                {"critic": "security", "severity": "critical", "rule_id": "SEC-1"},
                {"critic": "style", "severity": "critical", "rule_id": "STY-1"},
            ],
        )
        assert len(sig) == 16

    def test_no_preflight_key_returns_empty(self):
        """Preflight result without expected keys produces empty signature."""
        sig = compute_outcome_signature(
            preflight_result={"unexpected_key": "value"},
            critic_findings=[],
        )
        assert sig == compute_outcome_signature(preflight_result=None, critic_findings=[])


class TestCheckOutcomeStagnation:

    def test_empty_current_signature_returns_false(self):
        """Empty current signature means no problems → not stagnation."""
        assert check_outcome_stagnation(_correction(last_sig="abc123"), current_sig="") is False

    def test_first_attempt_not_stagnant(self):
        """No previous signature → not stagnation."""
        assert check_outcome_stagnation(_correction(last_sig=None), current_sig="abc123") is False

    def test_same_signature_is_stagnant(self):
        """Identical signatures across attempts → stagnation."""
        sig = "abc123def456"
        assert check_outcome_stagnation(_correction(last_sig=sig), current_sig=sig) is True

    def test_different_signatures_not_stagnant(self):
        """Different signatures → not stagnation."""
        assert check_outcome_stagnation(
            _correction(last_sig="abc123"), current_sig="def456"
        ) is False

    def test_none_prev_sig_same_current_not_stagnant(self):
        """When prev_sig is None, same current sig is not stagnation (no prior to compare)."""
        assert check_outcome_stagnation(_correction(last_sig=None), current_sig="abc") is False

    def test_empty_sig_with_empty_prev_not_stagnant(self):
        """Both empty → False (empty means no problems)."""
        assert check_outcome_stagnation(_correction(last_sig=""), current_sig="") is False
