"""
tests/unit/test_log_parser.py
==============================
Unit tests for stack trace pruning — pure functions, zero I/O.
"""
from __future__ import annotations
import pytest
from sacv.nodes._log_parser import (
    prune_java_stack, prune_typescript_stack, prune_stack,
    frames_to_dict, format_for_actor, ParsedFrame,
)

# ── Java stack trace fixtures ─────────────────────────────────────────────────

_JAVA_NPE = """
java.lang.NullPointerException: Cannot invoke "com.example.User.getId()"
	at com.example.service.UserService.findById(UserService.java:42)
	at com.example.controller.UserController.getUser(UserController.java:28)
	at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:897)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:764)
	at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:231)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at com.sun.proxy.$Proxy42.processRequest(Unknown Source)
	at org.hibernate.internal.SessionFactoryImpl$SessionBuilderImpl.openSession(SessionFactoryImpl.java:1281)
"""

_JAVA_BEAN = """
org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'paymentService'
	at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.instantiateBean(AbstractAutowireCapableBeanFactory.java:1331)
	at com.example.payment.PaymentService.<init>(PaymentService.java:18)
	at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
"""

_TS_ERROR = """
TypeError: Cannot read properties of undefined (reading 'balance')
    at UserWallet (src/features/wallet/UserWallet.tsx:34:15)
    at renderWithHooks (/node_modules/react-dom/cjs/react-dom.development.js:14985:18)
    at mountIndeterminateComponent (/node_modules/react-dom/cjs/react-dom.development.js:17811:13)
    at src/features/auth/AuthProvider.tsx:67:22
    at /node_modules/@tanstack/react-query/build/lib/QueryClientProvider.js:23:15
"""


class TestPruneJavaStack:

    def test_filters_spring_internals(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        for f in frames:
            assert not f.method.startswith("org.springframework")
            assert not f.method.startswith("javax.servlet")
            assert not f.method.startswith("org.apache.catalina")
            assert not f.method.startswith("sun.")
            assert not f.method.startswith("com.sun.")

    def test_keeps_user_package_frames(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        assert len(frames) == 2
        methods = [f.method for f in frames]
        assert any("UserService" in m for m in methods)
        assert any("UserController" in m for m in methods)

    def test_extracts_exception_message(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        assert frames[0].message == \
            "NullPointerException: Cannot invoke \"com.example.User.getId()\""

    def test_extracts_correct_line_numbers(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        lines = {f.file: f.line for f in frames}
        assert lines.get("UserService.java") == 42
        assert lines.get("UserController.java") == 28

    def test_empty_output_returns_empty(self):
        assert prune_java_stack("", "com.example") == []

    def test_wrong_package_returns_empty(self):
        assert prune_java_stack(_JAVA_NPE, "com.other") == []

    def test_bean_error_keeps_user_constructor(self):
        frames = prune_java_stack(_JAVA_BEAN, "com.example")
        assert any("PaymentService" in f.method for f in frames)

    def test_parse_frame_contains_required_fields(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        for f in frames:
            assert f.file
            assert f.line > 0
            assert f.method


class TestPruneTypeScriptStack:

    def test_filters_node_modules(self):
        frames = prune_typescript_stack(_TS_ERROR, "src")
        for f in frames:
            assert "node_modules" not in f.file

    def test_keeps_src_frames(self):
        frames = prune_typescript_stack(_TS_ERROR, "src")
        assert len(frames) >= 1
        files = [f.file for f in frames]
        assert any("wallet/UserWallet.tsx" in f or "auth/AuthProvider.tsx" in f
                   for f in files)

    def test_extracts_line_numbers(self):
        frames = prune_typescript_stack(_TS_ERROR, "src")
        lines = {f.file: f.line for f in frames}
        assert any(v == 34 for v in lines.values())

    def test_empty_output_returns_empty(self):
        assert prune_typescript_stack("", "src") == []


class TestDispatch:

    def test_backend_module_uses_java_parser(self):
        frames = prune_stack(_JAVA_NPE, "backend-domain", "com.example")
        assert all(".java" in f.file for f in frames)

    def test_frontend_module_uses_ts_parser(self):
        frames = prune_stack(_TS_ERROR, "frontend-feature", src_root="src")
        assert all(".tsx" in f.file or ".ts" in f.file for f in frames)


class TestFormatting:

    def test_frames_to_dict_structure(self):
        f = ParsedFrame(file="UserService.java", line=42,
                        method="com.example.UserService.findById", message="NPE")
        d = frames_to_dict([f])
        assert d[0]["file"] == "UserService.java"
        assert d[0]["line"] == 42
        assert d[0]["method"] == "com.example.UserService.findById"
        assert d[0]["message"] == "NPE"

    def test_format_for_actor_nonempty(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        output = format_for_actor(frames)
        assert "UserService" in output
        assert "42" in output

    def test_format_for_actor_empty(self):
        output = format_for_actor([])
        assert "no user-code frames" in output
