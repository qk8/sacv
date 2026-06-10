"""
tests/unit/test_cli_build_deps.py
==================================
Tests for ERR-003: _build_deps has user-friendly error surface.
Tests _safe_build helper: ImportError and generic Exception produce
clear messages and sys.exit(1) instead of raw tracebacks.
"""
from __future__ import annotations

import pytest


def _factory_succeeds():
    return "ok"


def _factory_import_error():
    raise ImportError("No module named 'docker'")


def _factory_runtime_error():
    raise RuntimeError("git repo not found")


class TestSafeBuildImportError:

    def test_exits_with_code_1(self):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit) as exc_info:
            _safe_build("DockerContainerManager", _factory_import_error)
        assert exc_info.value.code == 1

    def test_prints_error_label_and_message(self, capsys):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit):
            _safe_build("DockerContainerManager", _factory_import_error)
        stderr = capsys.readouterr().err
        assert "ERROR" in stderr
        assert "DockerContainerManager" in stderr
        assert "Missing dependency" in stderr
        assert "No module named 'docker'" in stderr

    def test_suggests_pip_install(self, capsys):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit):
            _safe_build("DockerContainerManager", _factory_import_error)
        stderr = capsys.readouterr().err
        assert "pip install" in stderr


class TestSafeBuildGenericError:

    def test_exits_with_code_1(self):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit) as exc_info:
            _safe_build("BranchManager", _factory_runtime_error)
        assert exc_info.value.code == 1

    def test_prints_error_label_type_and_message(self, capsys):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit):
            _safe_build("BranchManager", _factory_runtime_error)
        stderr = capsys.readouterr().err
        assert "ERROR" in stderr
        assert "BranchManager" in stderr
        assert "RuntimeError" in stderr
        assert "git repo not found" in stderr

    def test_suggests_check_configuration(self, capsys):
        from sacv.cli import _safe_build
        with pytest.raises(SystemExit):
            _safe_build("BranchManager", _factory_runtime_error)
        stderr = capsys.readouterr().err
        assert "configuration" in stderr


class TestSafeBuildSuccess:

    def test_returns_factory_result(self):
        from sacv.cli import _safe_build
        result = _safe_build("TestFactory", _factory_succeeds)
        assert result == "ok"
