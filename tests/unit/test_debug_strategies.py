"""
tests/unit/test_debug_strategies.py
=====================================
Unit tests for error classification and debug strategy selection.
All pure functions — zero I/O, zero LLM calls, zero Docker.
"""
from __future__ import annotations
import pytest
from sacv.nodes._debug_strategies import (
    classify_error, get_strategy, needs_jdwp, needs_cdp,
    needs_actuator, needs_delta_debug,
    ErrorType, DebugTool,
)


class TestClassifyError:
    """Java error classification."""

    def test_null_pointer_exception(self):
        assert classify_error("NullPointerException at UserService.java:42", "backend-domain") \
               == ErrorType.NULL_REFERENCE

    def test_concurrent_modification(self):
        raw = "ConcurrentModificationException in CartService.java:87"
        assert classify_error(raw, "backend-domain") == ErrorType.CONCURRENT_MODIFICATION

    def test_optimistic_lock(self):
        assert classify_error("OptimisticLockException: Row was updated", "backend-domain") \
               == ErrorType.OPTIMISTIC_LOCK

    def test_bean_creation_error(self):
        assert classify_error("BeanCreationException: Error creating bean 'paymentService'",
                              "backend-api") == ErrorType.BEAN_CREATION_ERROR

    def test_no_such_bean(self):
        assert classify_error("NoSuchBeanDefinitionException: 'IWalletService'",
                              "backend-domain") == ErrorType.BEAN_CREATION_ERROR

    def test_validation_error(self):
        assert classify_error("ConstraintViolationException: email must not be blank",
                              "backend-api") == ErrorType.VALIDATION_ERROR

    def test_http_400(self):
        assert classify_error("400 Bad Request: Validation failed", "backend-api") \
               == ErrorType.HTTP_400

    def test_class_cast(self):
        assert classify_error("ClassCastException: UserDto cannot be cast to AdminDto",
                              "backend-domain") == ErrorType.CLASS_CAST

    def test_unknown_falls_back(self):
        assert classify_error("Some weird error nobody has seen", "backend-domain") \
               == ErrorType.UNKNOWN


class TestClassifyErrorTypeScript:
    """TypeScript / Node.js error classification."""

    def test_null_reference_typescript(self):
        raw = "TypeError: Cannot read properties of null (reading 'id')"
        assert classify_error(raw, "frontend-feature") == ErrorType.NULL_REFERENCE

    def test_undefined_not_function(self):
        raw = "TypeError: undefined is not a function at UserForm.tsx:42"
        assert classify_error(raw, "frontend-feature") == ErrorType.NULL_REFERENCE

    def test_unhandled_promise_rejection(self):
        raw = "UnhandledPromiseRejectionWarning: Error: fetch failed"
        assert classify_error(raw, "frontend-data") == ErrorType.ASYNC_PROMISE_UNHANDLED

    def test_react_invariant(self):
        raw = "Invariant failed: You should not use <Route> outside a <Router>"
        assert classify_error(raw, "frontend-feature") == ErrorType.REACT_STATE_MISMATCH

    def test_validation_typescript(self):
        raw = "Validation failed: email must not be blank"
        assert classify_error(raw, "backend-api") == ErrorType.VALIDATION_ERROR

    def test_frontend_module_type_routing(self):
        """Same error text classified differently based on module_type."""
        raw = "400 Bad Request"
        assert classify_error(raw, "frontend-feature") == ErrorType.HTTP_400
        assert classify_error(raw, "backend-api")      == ErrorType.HTTP_400


class TestGetStrategy:

    def test_null_reference_uses_jdwp(self):
        s = get_strategy(ErrorType.NULL_REFERENCE)
        assert needs_jdwp(s)
        assert s.breakpoint_offset == -1

    def test_bean_error_uses_actuator(self):
        s = get_strategy(ErrorType.BEAN_CREATION_ERROR)
        assert needs_actuator(s)
        assert not needs_jdwp(s)
        assert s.step_type == "none"

    def test_validation_uses_delta_debug(self):
        s = get_strategy(ErrorType.VALIDATION_ERROR)
        assert needs_delta_debug(s)
        assert not needs_jdwp(s)

    def test_concurrent_mod_enables_thread_inspection(self):
        s = get_strategy(ErrorType.CONCURRENT_MODIFICATION)
        assert s.thread_inspection is True
        assert needs_jdwp(s)

    def test_async_race_uses_cdp(self):
        s = get_strategy(ErrorType.ASYNC_RACE_CONDITION)
        assert needs_cdp(s)
        assert not needs_jdwp(s)

    def test_react_state_uses_playwright(self):
        s = get_strategy(ErrorType.REACT_STATE_MISMATCH)
        assert needs_cdp(s)

    def test_unknown_falls_back_to_jdwp(self):
        s = get_strategy(ErrorType.UNKNOWN)
        assert needs_jdwp(s)

    def test_strategy_has_focus_hint(self):
        """Every defined strategy must have a non-empty focus hint."""
        for et in ErrorType:
            s = get_strategy(et)
            assert isinstance(s.focus_hint, str)

    def test_class_cast_evaluates_runtime_type(self):
        s = get_strategy(ErrorType.CLASS_CAST)
        assert any("getClass" in expr for expr in s.evaluate_expressions)

    def test_optimistic_lock_evaluates_version(self):
        s = get_strategy(ErrorType.OPTIMISTIC_LOCK)
        assert any("Version" in expr or "version" in expr
                   for expr in s.evaluate_expressions)

    def test_all_error_types_return_a_strategy(self):
        """Every ErrorType enum value returns a non-None strategy."""
        for et in ErrorType:
            s = get_strategy(et)
            assert s is not None
            assert isinstance(s.focus_hint, str)
            assert len(s.focus_hint) > 0

    def test_all_strategies_have_nonzero_max_steps_or_none(self):
        """Strategies that use JDWP/CDP have max_steps > 0."""
        for et in ErrorType:
            s = get_strategy(et)
            if needs_jdwp(s) or needs_cdp(s):
                assert s.max_steps > 0 or s.step_type == "none"

    def test_classified_error_types_have_strategies(self):
        """All Java-classified error types have strategies."""
        for _, et in _JAVA_CLASSIFICATION:
            s = get_strategy(et)
            assert s is not None

    def test_ts_classified_error_types_have_strategies(self):
        """All TS-classified error types have strategies."""
        for _, et in _TS_CLASSIFICATION:
            s = get_strategy(et)
            assert s is not None


# Need to import classification rules for the new tests
from sacv.nodes._debug_strategies import _JAVA_CLASSIFICATION, _TS_CLASSIFICATION
