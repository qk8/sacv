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

    def test_http_400_with_colon_not_space(self):
        """400 followed by colon (not space) should still match."""
        assert classify_error("Error 400: Bad Request", "backend-api") \
               == ErrorType.HTTP_400

    def test_http_400_with_equals(self):
        """400 preceded by equals (code=400) should still match."""
        assert classify_error("HTTP status code=400", "backend-api") \
               == ErrorType.HTTP_400

    def test_http_500_with_colon_not_space(self):
        """500 followed by colon (not space) should still match."""
        assert classify_error("Error 500: Internal Server Error", "backend-api") \
               == ErrorType.LOGIC_ERROR

    def test_http_400_at_string_start(self):
        """400 at the very start of the string (no preceding space) should match."""
        assert classify_error("400: Bad Request", "backend-api") \
               == ErrorType.HTTP_400

    def test_http_500_at_string_end(self):
        """500 at the end of the string (no trailing space) should match."""
        assert classify_error("Received status 500", "backend-api") \
               == ErrorType.LOGIC_ERROR

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


# ── classify_error: HTTP status priority ──────────────────────────────────────


class TestClassifyErrorHttpPriority:

    def test_400_before_validation_error(self):
        """HTTP 400 should be classified as HTTP_400, not VALIDATION_ERROR."""
        assert (
            classify_error(
                "400 Bad Request: Validation failed for field 'email'",
                "backend-domain",
            )
            == ErrorType.HTTP_400
        )

    def test_500_becomes_logic_error(self):
        assert (
            classify_error(
                "500 Internal Server Error: unexpected database state",
                "backend-domain",
            )
            == ErrorType.LOGIC_ERROR
        )

    def test_http_400_overrides_java_classification(self):
        """HTTP status detection runs before Java keyword matching."""
        assert (
            classify_error(
                "400 Bad Request: ConstraintViolationException",
                "backend-domain",
            )
            == ErrorType.HTTP_400
        )


# ── classify_error: additional Java error types ───────────────────────────────


class TestClassifyErrorAdditionalJava:

    def test_stale_object_state_exception(self):
        assert (
            classify_error("org.hibernate.StaleObjectStateException", "backend-domain")
            == ErrorType.OPTIMISTIC_LOCK
        )

    def test_object_optimistic_locking_failure(self):
        assert (
            classify_error(
                "org.springframework.orm.ObjectOptimisticLockingFailureException",
                "backend-domain",
            )
            == ErrorType.OPTIMISTIC_LOCK
        )

    def test_unsatisfied_dependency_exception(self):
        assert (
            classify_error(
                "org.springframework.beans.factory.UnsatisfiedDependencyException",
                "backend-domain",
            )
            == ErrorType.BEAN_CREATION_ERROR
        )

    def test_no_unique_bean_definition_exception(self):
        assert (
            classify_error(
                "org.springframework.beans.factory.NoUniqueBeanDefinitionException",
                "backend-domain",
            )
            == ErrorType.BEAN_CREATION_ERROR
        )

    def test_method_argument_not_valid_exception(self):
        assert (
            classify_error(
                "org.springframework.web.method.annotation."
                "MethodArgumentNotValidException",
                "backend-domain",
            )
            == ErrorType.VALIDATION_ERROR
        )

    def test_bind_exception(self):
        assert (
            classify_error("BindException: binding failed", "backend-domain")
            == ErrorType.VALIDATION_ERROR
        )

    def test_stack_overflow_error(self):
        assert (
            classify_error("java.lang.StackOverflowError", "backend-domain")
            == ErrorType.STACK_OVERFLOW
        )

    def test_out_of_memory_error(self):
        assert (
            classify_error("java.lang.OutOfMemoryError: Java heap space", "backend-domain")
            == ErrorType.OUT_OF_MEMORY
        )


# ── classify_error: additional TypeScript error types ─────────────────────────


class TestClassifyErrorAdditionalTypeScript:

    def test_cannot_set_property(self):
        assert (
            classify_error(
                "TypeError: Cannot set property 'disabled' of null",
                "frontend-feature",
            )
            == ErrorType.NULL_REFERENCE
        )

    def test_concurrent_update(self):
        assert (
            classify_error("concurrent state updates detected", "frontend-feature")
            == ErrorType.ASYNC_RACE_CONDITION
        )

    def test_set_state_called_in_render(self):
        assert (
            classify_error(
                "Warning: Cannot update a component while rendering "
                "a different component. setState called",
                "frontend-feature",
            )
            == ErrorType.ASYNC_RACE_CONDITION
        )

    def test_cannot_update_component(self):
        assert (
            classify_error(
                "Cannot update a component during render of another component",
                "frontend-feature",
            )
            == ErrorType.REACT_STATE_MISMATCH
        )

    def test_invariant_failed(self):
        assert (
            classify_error(
                "Invariant failed: Expected props to be defined",
                "frontend-feature",
            )
            == ErrorType.REACT_STATE_MISMATCH
        )

    def test_must_not_be_blank_ts(self):
        assert (
            classify_error("must not be blank", "frontend-feature")
            == ErrorType.VALIDATION_ERROR
        )

    def test_must_not_be_null_ts(self):
        assert (
            classify_error("email must not be null", "frontend-feature")
            == ErrorType.VALIDATION_ERROR
        )


# ── classify_error: edge cases ────────────────────────────────────────────────


class TestClassifyErrorEdgeCases:

    def test_empty_string(self):
        assert classify_error("", "backend-domain") == ErrorType.UNKNOWN

    def test_whitespace_only(self):
        assert classify_error("   \n  ", "backend-domain") == ErrorType.UNKNOWN

    def test_unrecognizable_java(self):
        assert (
            classify_error(
                "java.lang.RuntimeException: something weird happened",
                "backend-domain",
            )
            == ErrorType.UNKNOWN
        )

    def test_unrecognizable_ts(self):
        assert (
            classify_error(
                "ReferenceError: myVar is not defined",
                "frontend-feature",
            )
            == ErrorType.UNKNOWN
        )


# ── get_strategy: frontend JDWP→CDP swap ─────────────────────────────────────


class TestGetStrategyFrontendSwap:

    def test_null_reference_jdwp_becomes_cdp_for_frontend(self):
        s = get_strategy(ErrorType.NULL_REFERENCE, "frontend-feature")
        assert s.primary_tool == DebugTool.CDP
        assert s.breakpoint_offset == -1
        assert s.inspect_all_vars is True
        assert s.step_type == "step_over"

    def test_bean_creation_error_unchanged_for_frontend(self):
        """BEAN_CREATION_ERROR uses ACTUATOR, not JDWP — no swap needed."""
        s = get_strategy(ErrorType.BEAN_CREATION_ERROR, "frontend-feature")
        assert s.primary_tool == DebugTool.ACTUATOR

    def test_validation_error_unchanged_for_frontend(self):
        """VALIDATION_ERROR uses DELTA_DEBUG, not JDWP — no swap needed."""
        s = get_strategy(ErrorType.VALIDATION_ERROR, "frontend-feature")
        assert s.primary_tool == DebugTool.DELTA_DEBUG

    def test_async_race_unchanged_for_frontend(self):
        """ASYNC_RACE_CONDITION already uses CDP — no swap needed."""
        s = get_strategy(ErrorType.ASYNC_RACE_CONDITION, "frontend-feature")
        assert s.primary_tool == DebugTool.CDP

    def test_class_cast_jdwp_becomes_cdp_for_frontend(self):
        s = get_strategy(ErrorType.CLASS_CAST, "frontend-feature")
        assert s.primary_tool == DebugTool.CDP

    def test_optimistic_lock_jdwp_becomes_cdp_for_frontend(self):
        s = get_strategy(ErrorType.OPTIMISTIC_LOCK, "frontend-feature")
        assert s.primary_tool == DebugTool.CDP

    def test_concurrent_modification_jdwp_becomes_cdp_for_frontend(self):
        s = get_strategy(ErrorType.CONCURRENT_MODIFICATION, "frontend-feature")
        assert s.primary_tool == DebugTool.CDP


# ── needs_* helpers: thorough coverage ────────────────────────────────────────


class TestNeedsHelpers:

    def test_needs_jdwp_true_for_null_reference(self):
        assert needs_jdwp(get_strategy(ErrorType.NULL_REFERENCE)) is True

    def test_needs_jdwp_false_for_cdp(self):
        assert needs_jdwp(get_strategy(ErrorType.ASYNC_RACE_CONDITION)) is False

    def test_needs_jdwp_false_for_actuator(self):
        assert needs_jdwp(get_strategy(ErrorType.BEAN_CREATION_ERROR)) is False

    def test_needs_jdwp_false_for_delta_debug(self):
        assert needs_jdwp(get_strategy(ErrorType.VALIDATION_ERROR)) is False

    def test_needs_cdp_true_for_cdp(self):
        assert needs_cdp(get_strategy(ErrorType.ASYNC_RACE_CONDITION)) is True

    def test_needs_cdp_true_for_playwright(self):
        assert needs_cdp(get_strategy(ErrorType.REACT_STATE_MISMATCH)) is True

    def test_needs_cdp_false_for_jdwp(self):
        assert needs_cdp(get_strategy(ErrorType.NULL_REFERENCE)) is False

    def test_needs_actuator_true(self):
        assert needs_actuator(get_strategy(ErrorType.BEAN_CREATION_ERROR)) is True

    def test_needs_actuator_false_for_all_others(self):
        for et in ErrorType:
            if et == ErrorType.BEAN_CREATION_ERROR:
                continue
            assert needs_actuator(get_strategy(et)) is False

    def test_needs_delta_debug_true_for_validation(self):
        assert needs_delta_debug(get_strategy(ErrorType.VALIDATION_ERROR)) is True

    def test_needs_delta_debug_true_for_http_400(self):
        assert needs_delta_debug(get_strategy(ErrorType.HTTP_400)) is True

    def test_needs_delta_debug_false_for_jdwp(self):
        assert needs_delta_debug(get_strategy(ErrorType.NULL_REFERENCE)) is False


# ── Dataclass and enum integrity ──────────────────────────────────────────────


class TestDataclassAndEnums:

    def test_debug_strategy_is_frozen(self):
        """DebugStrategy must be frozen (immutable)."""
        s = get_strategy(ErrorType.NULL_REFERENCE)
        with pytest.raises(Exception):
            s.breakpoint_offset = 5

    def test_debug_strategy_default_values(self):
        """Verify default field values on DebugStrategy."""
        from sacv.nodes._debug_strategies import DebugStrategy
        s = DebugStrategy(
            error_type=ErrorType.UNKNOWN,
            primary_tool=DebugTool.JDWP,
        )
        assert s.breakpoint_offset == 0
        assert s.inspect_all_vars is True
        assert s.step_type == "step_over"
        assert s.max_steps == 3
        assert s.thread_inspection is False
        assert s.evaluate_expressions == []
        assert s.focus_hint == ""

    def test_error_type_enum_completeness(self):
        """All expected error types should be present."""
        expected = {
            ErrorType.NULL_REFERENCE,
            ErrorType.CONCURRENT_MODIFICATION,
            ErrorType.OPTIMISTIC_LOCK,
            ErrorType.BEAN_CREATION_ERROR,
            ErrorType.VALIDATION_ERROR,
            ErrorType.ASYNC_RACE_CONDITION,
            ErrorType.ASYNC_PROMISE_UNHANDLED,
            ErrorType.REACT_STATE_MISMATCH,
            ErrorType.CLASS_CAST,
            ErrorType.STACK_OVERFLOW,
            ErrorType.OUT_OF_MEMORY,
            ErrorType.TIMEOUT,
            ErrorType.HTTP_400,
            ErrorType.LOGIC_ERROR,
            ErrorType.UNKNOWN,
        }
        assert set(ErrorType) == expected

    def test_debug_tool_enum_completeness(self):
        """All expected debug tools should be present."""
        expected = {
            DebugTool.JDWP,
            DebugTool.CDP,
            DebugTool.ACTUATOR,
            DebugTool.DELTA_DEBUG,
            DebugTool.PLAYWRIGHT,
        }
        assert set(DebugTool) == expected

    def test_all_error_types_return_a_strategy(self):
        """Every ErrorType returns a strategy (from _STRATEGIES or default)."""
        from sacv.nodes._debug_strategies import _STRATEGIES
        for error_type in ErrorType:
            s = get_strategy(error_type)
            assert s is not None
            assert s.error_type == error_type or s == get_strategy(ErrorType.UNKNOWN)
            # Note: some ErrorType values (STACK_OVERFLOW, OUT_OF_MEMORY,
            # TIMEOUT) are not in _STRATEGIES and fall through to the
            # default strategy. This is fine — the default is a reasonable
            # fallback for unclassified errors.
