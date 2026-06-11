"""
nodes/_debug_strategies.py
===========================
Error type classification and breakpoint strategy selection.

Pure functions — no I/O, no LLM calls.
These determine HOW the IntelligentDebuggerNode will debug a failure.

Error type → debug strategy matrix:
  NULL_REFERENCE            → break one line before, inspect all variables
  CONCURRENT_MODIFICATION   → break at collection mutation, inspect thread
  OPTIMISTIC_LOCK           → break at @Version check, compare DB vs memory
  BEAN_CREATION_ERROR       → Spring Actuator query (no breakpoint needed)
  VALIDATION_ERROR          → Delta debug (binary search on payload)
  ASYNC_RACE_CONDITION      → break at Promise chain junction (Node.js)
  ASYNC_PROMISE_UNHANDLED   → break at rejection handler
  REACT_STATE_MISMATCH      → CDP evaluate React component state
  LOGIC_ERROR               → break at first user-code frame
  TIMEOUT                   → break one line before, inspect thread pool & connection state
  UNKNOWN                   → break at first user-code frame, full variable dump
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ErrorType(str, Enum):
    NULL_REFERENCE           = "NULL_REFERENCE"
    CONCURRENT_MODIFICATION  = "CONCURRENT_MODIFICATION"
    OPTIMISTIC_LOCK          = "OPTIMISTIC_LOCK"
    BEAN_CREATION_ERROR      = "BEAN_CREATION_ERROR"
    VALIDATION_ERROR         = "VALIDATION_ERROR"
    ASYNC_RACE_CONDITION     = "ASYNC_RACE_CONDITION"
    ASYNC_PROMISE_UNHANDLED  = "ASYNC_PROMISE_UNHANDLED"
    REACT_STATE_MISMATCH     = "REACT_STATE_MISMATCH"
    CLASS_CAST               = "CLASS_CAST"
    STACK_OVERFLOW           = "STACK_OVERFLOW"
    OUT_OF_MEMORY            = "OUT_OF_MEMORY"
    TIMEOUT                  = "TIMEOUT"
    HTTP_400                 = "HTTP_400"
    LOGIC_ERROR              = "LOGIC_ERROR"
    UNKNOWN                  = "UNKNOWN"


class DebugTool(str, Enum):
    JDWP        = "jdwp"        # Java debugger via JDWP
    CDP         = "cdp"         # Node.js via Chrome DevTools Protocol
    ACTUATOR    = "actuator"    # Spring Boot Actuator endpoints
    DELTA_DEBUG = "delta_debug" # Binary search on input payload
    PLAYWRIGHT  = "playwright"  # Browser state via Playwright


@dataclass(frozen=True)
class DebugStrategy:
    error_type:          ErrorType
    primary_tool:        DebugTool
    breakpoint_offset:   int   = 0       # -1 = one line before error frame
    inspect_all_vars:    bool  = True
    step_type:           Literal["step_over", "step_into", "none"] = "step_over"
    max_steps:           int   = 3
    thread_inspection:   bool  = False   # for concurrency bugs
    evaluate_expressions: list[str] = field(default_factory=list)
    focus_hint:          str   = ""      # short note for the agent


# ── Classification rules ──────────────────────────────────────────────────────

_JAVA_CLASSIFICATION: list[tuple[list[str], ErrorType]] = [
    (["NullPointerException"],                           ErrorType.NULL_REFERENCE),
    (["ConcurrentModificationException"],                ErrorType.CONCURRENT_MODIFICATION),
    (["OptimisticLockException", "StaleObjectStateException",
      "ObjectOptimisticLockingFailureException"],        ErrorType.OPTIMISTIC_LOCK),
    (["BeanCreationException", "NoSuchBeanDefinitionException",
      "UnsatisfiedDependencyException", "NoUniqueBeanDefinitionException"],
                                                         ErrorType.BEAN_CREATION_ERROR),
    (["ConstraintViolationException", "MethodArgumentNotValidException",
      "BindException", "Validation failed", "must not be blank"],
                                                         ErrorType.VALIDATION_ERROR),
    (["ClassCastException"],                             ErrorType.CLASS_CAST),
    (["StackOverflowError"],                             ErrorType.STACK_OVERFLOW),
    (["OutOfMemoryError"],                               ErrorType.OUT_OF_MEMORY),
    (["SocketTimeoutException", "Read timed out", "Write timed out",
      "RequestTimeoutException", "Connection timed out"], ErrorType.TIMEOUT),
]

_TS_CLASSIFICATION: list[tuple[list[str], ErrorType]] = [
    (["TypeError: Cannot read propert", "TypeError: Cannot set propert",
      "TypeError: undefined is not",   "TypeError: null is not"],
                                                         ErrorType.NULL_REFERENCE),
    (["UnhandledPromiseRejection", "UnhandledPromiseRejectionWarning"],
                                                         ErrorType.ASYNC_PROMISE_UNHANDLED),
    (["race condition", "concurrent", "setState called"],
                                                         ErrorType.ASYNC_RACE_CONDITION),
    (["Cannot update a component", "Warning: Can't perform", "Invariant failed"],
                                                         ErrorType.REACT_STATE_MISMATCH),
    (["422", "Validation failed", "must not be blank", "must not be null"],
                                                         ErrorType.VALIDATION_ERROR),
    (["timed out", "ETIMEDOUT", "deadline exceeded", "socket hang up"],
                                                         ErrorType.TIMEOUT),
]

# ── Strategy catalogue ────────────────────────────────────────────────────────

_STRATEGIES: dict[ErrorType, DebugStrategy] = {
    ErrorType.NULL_REFERENCE: DebugStrategy(
        error_type=ErrorType.NULL_REFERENCE,
        primary_tool=DebugTool.JDWP,
        breakpoint_offset=-1,
        inspect_all_vars=True,
        step_type="step_over",
        max_steps=3,
        focus_hint="Identify which variable is null at the frame before the NPE.",
    ),
    ErrorType.CONCURRENT_MODIFICATION: DebugStrategy(
        error_type=ErrorType.CONCURRENT_MODIFICATION,
        primary_tool=DebugTool.JDWP,
        breakpoint_offset=0,
        inspect_all_vars=True,
        step_type="step_into",
        max_steps=5,
        thread_inspection=True,
        evaluate_expressions=["Thread.currentThread().getName()"],
        focus_hint="Identify which thread mutates the collection while iterating.",
    ),
    ErrorType.OPTIMISTIC_LOCK: DebugStrategy(
        error_type=ErrorType.OPTIMISTIC_LOCK,
        primary_tool=DebugTool.JDWP,
        breakpoint_offset=0,
        inspect_all_vars=True,
        step_type="step_over",
        max_steps=2,
        evaluate_expressions=["entity.getVersion()", "entityManager.contains(entity)"],
        focus_hint="Compare the @Version field in memory vs. what the DB expects.",
    ),
    ErrorType.BEAN_CREATION_ERROR: DebugStrategy(
        error_type=ErrorType.BEAN_CREATION_ERROR,
        primary_tool=DebugTool.ACTUATOR,  # No breakpoint needed
        breakpoint_offset=0,
        inspect_all_vars=False,
        step_type="none",
        max_steps=0,
        focus_hint="Query /actuator/beans to see the live DI graph.",
    ),
    ErrorType.VALIDATION_ERROR: DebugStrategy(
        error_type=ErrorType.VALIDATION_ERROR,
        primary_tool=DebugTool.DELTA_DEBUG,  # Binary search on input
        breakpoint_offset=0,
        inspect_all_vars=False,
        step_type="none",
        max_steps=0,
        focus_hint="Find the minimal DTO / JSON payload that still triggers the error.",
    ),
    ErrorType.ASYNC_RACE_CONDITION: DebugStrategy(
        error_type=ErrorType.ASYNC_RACE_CONDITION,
        primary_tool=DebugTool.CDP,
        breakpoint_offset=0,
        inspect_all_vars=True,
        step_type="step_into",
        max_steps=8,
        thread_inspection=True,
        evaluate_expressions=["JSON.stringify(store.getState(), null, 2)"],
        focus_hint="Step into async chain; watch for out-of-order state mutations.",
    ),
    ErrorType.ASYNC_PROMISE_UNHANDLED: DebugStrategy(
        error_type=ErrorType.ASYNC_PROMISE_UNHANDLED,
        primary_tool=DebugTool.CDP,
        breakpoint_offset=0,
        inspect_all_vars=True,
        step_type="step_over",
        max_steps=3,
        focus_hint="Find the Promise that rejects without a .catch() handler.",
    ),
    ErrorType.REACT_STATE_MISMATCH: DebugStrategy(
        error_type=ErrorType.REACT_STATE_MISMATCH,
        primary_tool=DebugTool.PLAYWRIGHT,
        breakpoint_offset=0,
        inspect_all_vars=True,
        step_type="none",
        max_steps=0,
        evaluate_expressions=[
            "window.__REACT_DEVTOOLS_GLOBAL_HOOK__?.renderers?.size",
        ],
        focus_hint="Dump React component tree and state via Playwright evaluate.",
    ),
    ErrorType.HTTP_400: DebugStrategy(
        error_type=ErrorType.HTTP_400,
        primary_tool=DebugTool.DELTA_DEBUG,
        breakpoint_offset=0,
        inspect_all_vars=False,
        step_type="none",
        max_steps=0,
        focus_hint="Binary-search the request payload to find the offending field.",
    ),
    ErrorType.TIMEOUT: DebugStrategy(
        error_type=ErrorType.TIMEOUT,
        primary_tool=DebugTool.JDWP,
        breakpoint_offset=-1,
        inspect_all_vars=True,
        step_type="step_into",
        max_steps=5,
        evaluate_expressions=["java.lang.Thread.getAllStackTraces().keySet()"],
        focus_hint="Inspect thread state at timeout: check thread pool saturation, DB connection pool, and network calls.",
    ),
    ErrorType.CLASS_CAST: DebugStrategy(
        error_type=ErrorType.CLASS_CAST,
        primary_tool=DebugTool.JDWP,
        breakpoint_offset=-1,
        inspect_all_vars=True,
        step_type="step_over",
        max_steps=2,
        evaluate_expressions=["obj.getClass().getName()"],
        focus_hint="Check runtime type of the object being cast.",
    ),
}

_DEFAULT_STRATEGY = DebugStrategy(
    error_type=ErrorType.UNKNOWN,
    primary_tool=DebugTool.JDWP,
    breakpoint_offset=0,
    inspect_all_vars=True,
    step_type="step_over",
    max_steps=5,
    focus_hint="General inspection: dump all variables at first user-code frame.",
)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_error(raw_output: str, module_type: str) -> ErrorType:
    """
    Pure function. Classify a raw error string into an ErrorType.
    Checks both Java and TypeScript patterns based on module_type.

    HTTP status codes are checked first so that messages like
    "400 Bad Request: Validation failed" are classified as HTTP_400
    rather than VALIDATION_ERROR.
    """
    # HTTP status code detection (works for both — highest priority)
    # Use \b word boundaries instead of space-padded matching so that
    # patterns like "400:", ":400", "code=400" all match correctly.
    if re.search(r"\b400\b", raw_output) or "Bad Request" in raw_output:
        return ErrorType.HTTP_400
    if re.search(r"\b500\b", raw_output) or "Internal Server Error" in raw_output:
        return ErrorType.LOGIC_ERROR

    # Language-agnostic timeout detection
    if re.search(r"timeout", raw_output, re.IGNORECASE) or \
       re.search(r"timed?\s*out", raw_output, re.IGNORECASE) or \
       re.search(r"deadline exceeded", raw_output, re.IGNORECASE) or \
       re.search(r"ETIMEDOUT", raw_output):
        return ErrorType.TIMEOUT

    rules = _TS_CLASSIFICATION if "frontend" in module_type else _JAVA_CLASSIFICATION

    for keywords, error_type in rules:
        if any(kw in raw_output for kw in keywords):
            return error_type

    return ErrorType.UNKNOWN


def get_strategy(error_type: ErrorType, module_type: str = "") -> DebugStrategy:
    """Pure function. Returns the debug strategy for a given error type.

    For frontend/TypeScript modules, Java-only tools (JDWP) are swapped
    to CDP since the JDWP protocol is not applicable.
    """
    strategy = _STRATEGIES.get(error_type, _DEFAULT_STRATEGY)

    # For frontend/TS modules, JDWP is not applicable — use CDP
    if "frontend" in module_type and strategy.primary_tool == DebugTool.JDWP:
        return DebugStrategy(
            error_type=strategy.error_type,
            primary_tool=DebugTool.CDP,
            breakpoint_offset=strategy.breakpoint_offset,
            inspect_all_vars=strategy.inspect_all_vars,
            step_type=strategy.step_type,
            max_steps=strategy.max_steps,
            thread_inspection=strategy.thread_inspection,
            evaluate_expressions=list(strategy.evaluate_expressions),
            focus_hint=strategy.focus_hint,
        )

    return strategy


def needs_jdwp(strategy: DebugStrategy) -> bool:
    return strategy.primary_tool == DebugTool.JDWP


def needs_cdp(strategy: DebugStrategy) -> bool:
    return strategy.primary_tool in (DebugTool.CDP, DebugTool.PLAYWRIGHT)


def needs_actuator(strategy: DebugStrategy) -> bool:
    return strategy.primary_tool == DebugTool.ACTUATOR


def needs_delta_debug(strategy: DebugStrategy) -> bool:
    return strategy.primary_tool == DebugTool.DELTA_DEBUG
