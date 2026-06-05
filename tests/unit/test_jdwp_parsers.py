"""
tests/unit/test_jdwp_parsers.py
================================
Unit tests for pure parser functions in jdwp_client.py.

Tests cover:
1. _parse_hit — breakpoint/step hit location parsing from JDB output
2. _parse_locals — local variable parsing from JDB 'locals' output
3. _parse_where — stack trace parsing from JDB 'where' output
"""
from __future__ import annotations

import pytest
from sacv.adapters.debug.jdwp_client import (
    _parse_hit, _parse_locals, _parse_where,
    BreakpointHitInfo, LocalVariable,
)


class TestParseHit:

    def test_parses_standard_breakpoint_hit(self):
        output = (
            'Breakpoint hit:  thread=main(t1), '
            'com.example.UserService.findById(), line=42 bci=73'
        )
        result = _parse_hit(output)
        assert result is not None
        assert result.class_name == "com.example.UserService"
        assert result.method_name == "findById"
        assert result.file == "UserService.java"
        assert result.line == 42
        assert result.thread_name == "main"

    def test_parses_inner_class_breakpoint(self):
        output = (
            'Breakpoint hit:  thread=main(t1), '
            'com.example.UserService$Builder.build(), line=100 bci=10'
        )
        result = _parse_hit(output)
        assert result is not None
        assert result.class_name == "com.example.UserService$Builder"
        assert result.method_name == "build"
        assert result.file == "UserService.java"

    def test_parses_step_hit(self):
        output = (
            'Step complete:  thread=main(t1), '
            'com.example.service.CartService.checkout(), line=55 bci=20'
        )
        result = _parse_hit(output)
        assert result is not None
        assert result.class_name == "com.example.service.CartService"
        assert result.method_name == "checkout"
        assert result.line == 55

    def test_returns_none_on_no_match(self):
        output = "Compiling UserService.java..."
        assert _parse_hit(output) is None

    def test_returns_none_on_empty_output(self):
        assert _parse_hit("") is None

    def test_returns_none_on_jdb_prompt_only(self):
        assert _parse_hit("> ") is None

    def test_parses_with_thread_name(self):
        output = (
            'Breakpoint hit:  thread=pool-1-thread-5(t12), '
            'com.example.Worker.process(), line=80 bci=5'
        )
        result = _parse_hit(output)
        assert result is not None
        assert result.thread_name == "pool-1-thread-5"


class TestParseLocals:

    def test_parses_single_variable(self):
        output = "  user = com.example.User@abc123 (User)"
        result = _parse_locals(output)
        assert len(result) == 1
        assert result[0].name == "user"
        assert result[0].value == "com.example.User@abc123"
        assert result[0].type == "User"

    def test_parses_multiple_variables(self):
        output = (
            "  user = com.example.User@abc123 (User)\n"
            "  id = 42 (Long)\n"
            "  name = \"Alice\" (String)"
        )
        result = _parse_locals(output)
        assert len(result) == 3
        assert result[0].name == "user"
        assert result[1].name == "id"
        assert result[2].name == "name"

    def test_parses_primitive_types(self):
        output = (
            "  count = 100 (int)\n"
            "  active = true (boolean)\n"
            "  rate = 3.14 (double)"
        )
        result = _parse_locals(output)
        assert len(result) == 3
        assert result[0].type == "int"
        assert result[0].value == "100"
        assert result[1].value == "true"
        assert result[2].value == "3.14"

    def test_returns_empty_for_empty_output(self):
        assert _parse_locals("") == []

    def test_returns_empty_for_non_variable_lines(self):
        output = (
            "Local variables:\n"
            "  ----\n"
            "(no local variables)"
        )
        result = _parse_locals(output)
        assert result == []

    def test_parses_value_with_spaces(self):
        output = '  message = "Hello World" (String)'
        result = _parse_locals(output)
        assert len(result) == 1
        assert result[0].value == '"Hello World"'
        assert result[0].type == "String"

    def test_defaults_type_to_unknown_when_missing(self):
        output = "  someVar = someValue"
        result = _parse_locals(output)
        assert len(result) == 1
        assert result[0].type == "unknown"


class TestParseWhere:

    def test_parses_single_stack_frame(self):
        output = "  [1] com.example.UserService.findById(UserService.java:42)"
        result = _parse_where(output)
        assert len(result) == 1
        assert result[0] == "com.example.UserService.findById(UserService.java:42)"

    def test_parses_multiple_stack_frames(self):
        output = (
            "  [1] com.example.UserService.findById(UserService.java:42)\n"
            "  [2] com.example.controller.UserController.handle(UserController.java:15)\n"
            "  [3] org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:683)"
        )
        result = _parse_where(output)
        assert len(result) == 3
        assert "UserService" in result[0]
        assert "UserController" in result[1]
        assert "FrameworkServlet" in result[2]

    def test_returns_empty_for_empty_output(self):
        assert _parse_where("") == []

    def test_returns_empty_for_non_stack_frame_lines(self):
        output = (
            "Java frames:\n"
            "  j com.example.Main.main([Ljava/lang/String;)V\n"
            "  j java.lang.reflect.Method.invoke(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;"
        )
        result = _parse_where(output)
        assert result == []

    def test_parses_jni_native_frame(self):
        output = "  [0] <JVM_ENTRY> (JVM_ENTRY.cpp:0)"
        result = _parse_where(output)
        assert len(result) == 1
        assert result[0] == "<JVM_ENTRY> (JVM_ENTRY.cpp:0)"
