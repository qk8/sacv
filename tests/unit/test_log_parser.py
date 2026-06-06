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
            "java.lang.NullPointerException: Cannot invoke \"com.example.User.getId()\""

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

    def test_filters_hibernate_internals(self):
        raw = """
java.lang.RuntimeException
	at org.hibernate.internal.SessionImpl.fireSave(SessionImpl.java:113)
	at com.example.repo.UserRepository.save(UserRepository.java:25)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "UserRepository" in frames[0].method

    def test_filters_junit_internals(self):
        raw = """
java.lang.AssertionError
	at org.junit.Assert.fail(Assert.java:47)
	at org.junit.jupiter.api.AssertEquals.fail(AssertEquals.java:42)
	at com.example.service.UserServiceTest.testFindById(UserServiceTest.java:15)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "UserServiceTest" in frames[0].method

    def test_filters_mockito_internals(self):
        raw = """
org.mockito.exceptions.base.MockitoException
	at org.mockito.internal.runners.DefaultInternalRunner.run(DefaultInternalRunner.java:35)
	at com.example.service.UserServiceTest.testMock(UserServiceTest.java:30)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "UserServiceTest" in frames[0].method

    def test_filters_bytebuddy_internals(self):
        raw = """
java.lang.Exception
	at net.bytebuddy.asm.Advice$Token$Scope$Independent.open(Advice.java:322)
	at com.example.util.Helper.doWork(Helper.java:10)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "Helper" in frames[0].method

    def test_filters_hikaricp_internals(self):
        raw = """
java.sql.SQLException
	at com.zaxxer.hikari.pool.PoolBase.checkDeadPool(PoolBase.java:123)
	at com.example.repo.ConnectionPoolTest.testPool(ConnectionPoolTest.java:20)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "ConnectionPoolTest" in frames[0].method

    def test_filters_logback_internals(self):
        raw = """
java.lang.IllegalStateException
	at ch.qos.logback.core.spi.ContextBase.addListener(ContextBase.java:100)
	at com.example.config.LogConfig.setup(LogConfig.java:15)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "LogConfig" in frames[0].method

    def test_filters_micrometer_internals(self):
        raw = """
io.micrometer.core.instrument.internal.TimedRunnable.run(TimedRunnable.java:50)
	at com.example.monitoring.MonitorService.check(MonitorService.java:25)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "MonitorService" in frames[0].method

    def test_filters_tomcat_internals(self):
        raw = """
java.io.IOException
	at org.apache.tomcat.util.net.NioEndpoint$NioSocketWrapper.fillReadBuffer(NioEndpoint.java:123)
	at com.example.http.HttpHandler.handle(HttpHandler.java:40)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "HttpHandler" in frames[0].method

    def test_multiple_user_frames_sorted(self):
        raw = """
java.lang.RuntimeException
	at com.example.service.UserService.findById(UserService.java:42)
	at com.example.controller.UserController.getUser(UserController.java:28)
	at com.example.filter.AuthFilter.doFilter(AuthFilter.java:15)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 3
        # Innermost frame first (UserService), then outer (UserController)
        assert "UserService" in frames[0].method
        assert "UserController" in frames[1].method
        assert "AuthFilter" in frames[2].method


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

    def test_filters_dist_paths(self):
        raw = "Error: test\n    at fn (dist/bundle.js:10:5)"
        frames = prune_typescript_stack(raw, "src")
        assert all("/dist/" not in f.file for f in frames)

    def test_filters_next_build_paths(self):
        raw = "Error: test\n    at fn (.next/server.js:20:10)"
        frames = prune_typescript_stack(raw, "src")
        assert all("/.next/" not in f.file for f in frames)

    def test_custom_src_root(self):
        raw = "Error: test\n    at fn (app/components/Button.tsx:5:10)"
        frames = prune_typescript_stack(raw, "app")
        assert len(frames) >= 1
        assert any("Button.tsx" in f.file for f in frames)

    def test_mixed_valid_and_invalid_paths(self):
        raw = (
            "Error: test\n"
            "    at User (src/features/User.tsx:10:5)\n"
            "    at /node_modules/react/index.js:20:15\n"
            "    at Auth (src/features/Auth.tsx:30:10)\n"
        )
        frames = prune_typescript_stack(raw, "src")
        assert len(frames) == 2
        files = [f.file for f in frames]
        assert any("User.tsx" in f for f in files)
        assert any("Auth.tsx" in f for f in files)


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

    def test_format_for_actor_includes_message_on_first_frame_only(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        output = format_for_actor(frames)
        # First frame has message
        assert "NullPointerException" in output
        # Second frame does NOT have the message
        lines = output.split("\n")
        assert len(lines) >= 2
        # The message should only appear once (on the first line)
        assert output.count("NullPointerException") == 1

    def test_format_for_actor_multiple_frames(self):
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        output = format_for_actor(frames)
        assert "UserService" in output
        assert "UserController" in output

    def test_frames_to_dict_empty_list(self):
        assert frames_to_dict([]) == []

    def test_format_for_actor_preserves_order(self):
        """Frames are formatted in the order returned by pruner (innermost first)."""
        frames = prune_java_stack(_JAVA_NPE, "com.example")
        output = format_for_actor(frames)
        # UserService.java:42 should appear before UserController.java:28
        pos1 = output.find("UserService.java:42")
        pos2 = output.find("UserController.java:28")
        assert pos1 < pos2

    def test_filters_java_lang_thread(self):
        raw = """
java.lang.Exception
	at java.lang.Thread.run(Thread.java:748)
	at com.example.worker.WorkerThread.run(WorkerThread.java:20)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "WorkerThread" in frames[0].method

    def test_filters_java_lang_threadmxbean_not_user_code(self):
        """java.lang.ThreadMXBean is a JDK management class — must be filtered."""
        raw = """
java.lang.RuntimeException
	at java.lang.ThreadMXBean.getThreadInfo(ThreadMXBean.java:99)
	at com.example.monitoring.ThreadMonitor.check(ThreadMonitor.java:30)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 1
        assert "ThreadMonitor" in frames[0].method

    def test_inner_class_name_preserved(self):
        raw = """
java.lang.RuntimeException
	at com.example.service.UserService$Builder.build(UserService.java:55)
	at com.example.controller.UserController.create(UserController.java:12)
"""
        frames = prune_java_stack(raw, "com.example")
        assert len(frames) == 2
        assert "UserService" in frames[0].method


class TestPruneTypeScriptEdgeCases:

    def test_no_source_frames_returns_empty(self):
        raw = """
Error: test
    at fn (/node_modules/react/index.js:10:5)
    at fn (/dist/bundle.js:20:10)
"""
        frames = prune_typescript_stack(raw, "src")
        assert frames == []

    def test_components_directory_included(self):
        raw = "Error: test\n    at Button (src/components/Button.tsx:5:10)"
        frames = prune_typescript_stack(raw, "src")
        assert len(frames) == 1
        assert "Button.tsx" in frames[0].file

    def test_pages_directory_included(self):
        raw = "Error: test\n    at Page (src/pages/index.tsx:1:1)"
        frames = prune_typescript_stack(raw, "src")
        assert len(frames) == 1

    def test_lib_directory_included(self):
        raw = "Error: test\n    at util (src/lib/utils.ts:10:5)"
        frames = prune_typescript_stack(raw, "src")
        assert len(frames) == 1

    def test_app_directory_included(self):
        raw = "Error: test\n    at layout (src/app/layout.tsx:5:5)"
        frames = prune_typescript_stack(raw, "src")
        assert len(frames) == 1


class TestDispatchEdgeCases:

    def test_empty_module_type_defaults_to_java(self):
        """Empty module_type should fall through to Java parser."""
        frames = prune_stack(_JAVA_NPE, "", "com.example")
        assert all(".java" in f.file for f in frames)

    def test_module_with_frontend_in_name(self):
        """Any module_type containing 'frontend' uses TS pruner."""
        frames = prune_stack(_TS_ERROR, "frontend-data", src_root="src")
        assert all(".tsx" in f.file or ".ts" in f.file for f in frames)


class TestFormattingEdgeCases:

    def test_format_for_actor_single_frame(self):
        """Single frame formats correctly with message."""
        raw = """
java.lang.RuntimeException: single error
	at com.example.service.SingleService.doWork(SingleService.java:10)
"""
        frames = prune_java_stack(raw, "com.example")
        output = format_for_actor(frames)
        assert "SingleService" in output
        assert "RuntimeException" in output

    def test_format_for_actor_with_empty_message(self):
        """Frames without exception message still format."""
        raw = """
	at com.example.service.UserService.findById(UserService.java:42)
"""
        frames = prune_java_stack(raw, "com.example")
        output = format_for_actor(frames)
        assert "UserService" in output

    def test_frames_to_dict_preserves_all_fields(self):
        """Every dict key from ParsedFrame is present."""
        f = ParsedFrame(
            file="Test.java", line=99,
            method="com.example.Test.run", message="boom",
        )
        d = frames_to_dict([f])
        assert set(d[0].keys()) == {"file", "line", "method", "message"}
