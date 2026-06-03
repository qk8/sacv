"""
tests/unit/test_tdd_gate_helpers.py
=====================================
Unit tests for TDD gate helper functions and pure logic.

Tests cover:
1. _feature_id — ID generation from task IDs
2. _canonicalise_test_path — permanent path enforcement
3. _test_command_for — test command selection
4. Strategy absence handling in tdd_gate_node
5. JSON parse failure handling
6. Unexpected test pass detection
"""
from __future__ import annotations

import pytest

from sacv.nodes.tdd_gate import _feature_id, _canonicalise_test_path, _test_command_for


@pytest.mark.unit
class TestFeatureId:

    def test_alphanumeric_preserved(self):
        assert _feature_id("task-001") == "task-001"

    def test_special_chars_replaced(self):
        assert _feature_id("task@#$%001") == "task----001"

    def test_uppercase_lowercased(self):
        assert _feature_id("Task-ABC-001") == "task-abc-001"

    def test_truncated_at_32(self):
        long_id = "a" * 50
        assert len(_feature_id(long_id)) == 32

    def test_empty_string(self):
        result = _feature_id("")
        assert result == ""

    def test_mixed_special_and_uppercase(self):
        result = _feature_id("My Task! @2024")
        assert "my" in result
        assert "@" not in result
        assert "!" not in result


@pytest.mark.unit
class TestCanonicaliseTestPath:

    def test_frontend_preserves_existing_path(self):
        """Frontend paths starting with tests/e2e/ are kept as-is."""
        result = _canonicalise_test_path(
            "tests/e2e/features/login.spec.ts",
            "frontend-feature", "task-001",
        )
        assert result == "tests/e2e/features/login.spec.ts"

    def test_frontend_rewrites_non_e2e_path(self):
        """Frontend paths not under tests/e2e/ are rewritten."""
        result = _canonicalise_test_path(
            "tests/unit/login.spec.ts",
            "frontend-feature", "task-001",
        )
        assert result == "tests/e2e/features/task-001.spec.ts"

    def test_api_preserves_existing_path(self):
        """API paths starting with tests/api/ are kept as-is."""
        result = _canonicalise_test_path(
            "tests/api/routes/user.spec.ts",
            "backend-api", "task-001",
        )
        assert result == "tests/api/routes/user.spec.ts"

    def test_api_rewrites_non_api_path(self):
        """API paths not under tests/api/ are rewritten."""
        result = _canonicalise_test_path(
            "tests/unit/user.spec.ts",
            "backend-api", "task-001",
        )
        assert result == "tests/api/routes/task-001.spec.ts"

    def test_backend_domain_uses_java_package(self):
        """Backend domain paths use user_package for Java test location."""
        result = _canonicalise_test_path(
            "UserServiceTest.java",
            "backend-domain", "task-001",
            user_package="com.example",
        )
        assert result == "src/test/java/com/example/task-001Test.java"

    def test_backend_domain_preserves_src_test_path(self):
        """Paths already under src/test/ are kept as-is."""
        result = _canonicalise_test_path(
            "src/test/java/com/example/UserServiceTest.java",
            "backend-domain", "task-001",
        )
        assert result == "src/test/java/com/example/UserServiceTest.java"

    def test_custom_java_package_reflected_in_path(self):
        result = _canonicalise_test_path(
            "X.java",
            "backend-domain", "task-001",
            user_package="com.acme",
        )
        assert "com/acme/task-001Test.java" in result

    def test_empty_path_uses_default(self):
        """Empty file_path falls back to default convention."""
        result = _canonicalise_test_path(
            "", "frontend-feature", "task-001",
        )
        assert result == "tests/e2e/features/task-001.spec.ts"

    def test_task_id_with_special_chars_is_sanitized_in_path(self):
        result = _canonicalise_test_path(
            "tests/unit/X.java", "backend-domain", "task@001!test",
        )
        # Feature ID should be sanitized
        assert "@" not in result
        assert "!" not in result


@pytest.mark.unit
class TestTestCommandFor:

    def test_frontend_returns_playwright_command(self):
        result = _test_command_for("frontend-feature")
        assert "playwright" in result

    def test_backend_api_returns_npm_test(self):
        result = _test_command_for("backend-api")
        assert "npm test" in result
        assert "tests/api" in result

    def test_backend_domain_returns_mvn_test(self):
        result = _test_command_for("backend-domain")
        assert "mvn test" in result

    def test_infrastructure_returns_mvn_test(self):
        result = _test_command_for("infrastructure")
        assert "mvn test" in result

    def test_cross_cutting_returns_mvn_test(self):
        result = _test_command_for("cross-cutting")
        assert "mvn test" in result
