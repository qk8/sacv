"""
tests/unit/test_debugger_helpers.py
=====================================
Unit tests for intelligent debugger helper functions — pure, no I/O.
"""
from __future__ import annotations

import pytest
from sacv.nodes.intelligent_debugger import (
    _extract_request_payload,
    _extract_endpoint,
)


def _s(**kw):
    base = {
        "session_id": "t", "task_id": "t", "task_description": "",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "verifier_verdict": None,
    }
    base.update(kw)
    return base


class TestExtractRequestPayload:

    def test_extracts_json_from_message(self):
        s = _s(verifier_verdict={
            "test_failures": [{
                "message": "Expected 200 but got 500: "
                           '{"userId":1,"action":"transfer"}'
            }],
        })
        result = _extract_request_payload(s)
        assert result == {"userId": 1, "action": "transfer"}

    def test_returns_empty_dict_no_json(self):
        s = _s(verifier_verdict={
            "test_failures": [{"message": "NullPointerException"}],
        })
        assert _extract_request_payload(s) == {}

    def test_returns_empty_dict_no_verdict(self):
        assert _extract_request_payload(_s(verifier_verdict=None)) == {}

    def test_returns_empty_dict_no_failures(self):
        s = _s(verifier_verdict={})
        assert _extract_request_payload(s) == {}

    def test_skips_broken_json(self):
        """Malformed JSON is skipped, empty dict returned."""
        s = _s(verifier_verdict={
            "test_failures": [{
                "message": 'Error: {"userId":1, "broken'
            }],
        })
        assert _extract_request_payload(s) == {}

    def test_uses_first_valid_json(self):
        """Takes the first failure message that contains valid JSON."""
        s = _s(verifier_verdict={
            "test_failures": [
                {"message": 'Error: {"valid": true}'},
                {"message": 'Error: {"also_valid": false}'},
            ],
        })
        result = _extract_request_payload(s)
        assert result == {"valid": True}

    def test_nested_json(self):
        """Nested JSON objects are correctly parsed."""
        s = _s(verifier_verdict={
            "test_failures": [{
                "message": "Response: "
                           '{"user":{"name":"Alice","roles":["admin"]}}'
            }],
        })
        result = _extract_request_payload(s)
        assert result["user"]["name"] == "Alice"
        assert result["user"]["roles"] == ["admin"]

    def test_rindex_finds_last_brace(self):
        """rindex('}') finds the last closing brace, not an early one."""
        s = _s(verifier_verdict={
            "test_failures": [{
                "message": "Error: {\"a\":1} extra } text {\"b\":2}"
            }],
        })
        result = _extract_request_payload(s)
        # rindex finds the last }, so it captures from first { to last }
        # which includes the middle } — this will be invalid JSON
        # Actually the function uses rindex which finds the LAST }
        # So it tries to parse: {"a":1} extra } text {"b":2}
        # This is invalid JSON, so it should return {}
        assert result == {}

    def test_multiple_failures_skips_non_json(self):
        """Skips failures without braces, tries next."""
        s = _s(verifier_verdict={
            "test_failures": [
                {"message": "NullPointerException at line 42"},
                {"message": "Error: " + '{"valid": true}'},
            ],
        })
        result = _extract_request_payload(s)
        assert result == {"valid": True}


class TestExtractEndpoint:

    def test_extracts_from_task_description(self):
        s = _s(task_description="Implement POST /api/users endpoint")
        assert _extract_endpoint(s) == "/api/users"

    def test_extracts_v1_endpoint(self):
        s = _s(task_description="Create a /v1/orders API")
        assert _extract_endpoint(s) == "/v1/orders"

    def test_extracts_v2_endpoint(self):
        s = _s(task_description="Update /v2/products endpoint")
        assert _extract_endpoint(s) == "/v2/products"

    def test_no_endpoint_returns_empty(self):
        s = _s(task_description="Fix a NullPointerException in UserService")
        assert _extract_endpoint(s) == ""

    def test_no_task_description(self):
        assert _extract_endpoint(_s(task_description="")) == ""

    def test_takes_first_pattern_match(self):
        """When multiple patterns exist, takes the first one found."""
        s = _s(
            task_description="Migrate /api/v1 to /v2/ endpoints",
        )
        result = _extract_endpoint(s)
        assert result.startswith("/api/")

    def test_truncates_to_40_chars(self):
        """Endpoint extraction is truncated at 40 characters."""
        long_path = "/api/" + "a" * 50
        s = _s(task_description=f"Implement {long_path}")
        result = _extract_endpoint(s)
        assert len(result) <= 40

    def test_strips_trailing_punctuation(self):
        s = _s(task_description="Call /api/users.")
        result = _extract_endpoint(s)
        assert result == "/api/users"

    def test_truncates_long_descriptions(self):
        """Long task descriptions are truncated at 40 chars from match."""
        long = "Implement the /api/very-long-endpoint-that-exceeds forty-char-limit for users"
        s = _s(task_description=long)
        result = _extract_endpoint(s)
        assert len(result) <= 40
        assert "/api/" in result

    def test_case_insensitive_pattern_match(self):
        s = _s(task_description="Fix the /API/USERS endpoint")
        result = _extract_endpoint(s)
        assert "/API/USERS" in result or "/api/users" in result.lower()
