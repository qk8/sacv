"""
tests/unit/test_debug_strategies_timeout.py
==============================================
Unit tests for TIMEOUT error classification in debug strategies.

Tests cover:
1. classify_error("timeout") returns ErrorType.TIMEOUT
2. classify_error("timed out") returns ErrorType.TIMEOUT
3. classify_error("request timeout") returns ErrorType.TIMEOUT
4. classify_error("connection timeout") returns ErrorType.TIMEOUT
5. classify_error("deadline exceeded") returns ErrorType.TIMEOUT
6. classify_error("ETIMEDOUT") returns ErrorType.TIMEOUT
7. get_strategy for TIMEOUT returns a valid strategy
"""
from __future__ import annotations

import pytest

from sacv.nodes._debug_strategies import classify_error, get_strategy, ErrorType


class TestClassifyErrorTimeout:

    def test_simple_timeout_string(self):
        assert classify_error("timeout", "backend-domain") == ErrorType.TIMEOUT

    def test_timed_out(self):
        assert classify_error("timed out", "backend-domain") == ErrorType.TIMEOUT

    def test_request_timeout(self):
        assert classify_error("RequestTimeoutException", "backend-domain") == ErrorType.TIMEOUT

    def test_connection_timeout(self):
        assert classify_error("Connection timed out", "backend-domain") == ErrorType.TIMEOUT

    def test_deadline_exceeded(self):
        assert classify_error("deadline exceeded", "backend-domain") == ErrorType.TIMEOUT

    def test_etimedout(self):
        assert classify_error("Error: ETIMEDOUT", "backend-domain") == ErrorType.TIMEOUT

    def test_nodejs_timeout(self):
        assert classify_error("Error: socket hang up, reason: connection timed out", "frontend-feature") == ErrorType.TIMEOUT

    def test_java_socket_timeout(self):
        assert classify_error("java.net.SocketTimeoutException: Read timed out", "backend-domain") == ErrorType.TIMEOUT

    def test_http_request_timeout(self):
        assert classify_error("HTTP request timed out after 30000ms", "backend-domain") == ErrorType.TIMEOUT


class TestGetStrategyTimeout:

    def test_timeout_returns_valid_strategy(self):
        """get_strategy for TIMEOUT returns a non-None DebugStrategy."""
        s = get_strategy(ErrorType.TIMEOUT)
        assert s is not None
        assert s.error_type == ErrorType.TIMEOUT

    def test_timeout_strategy_has_primary_tool(self):
        """TIMEOUT strategy has a valid primary tool."""
        from sacv.nodes._debug_strategies import DebugTool
        s = get_strategy(ErrorType.TIMEOUT)
        assert s.primary_tool in (DebugTool.JDWP, DebugTool.CDP, DebugTool.ACTUATOR,
                                   DebugTool.DELTA_DEBUG, DebugTool.PLAYWRIGHT)
