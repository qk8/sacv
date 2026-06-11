"""
tests/unit/test_tracing_import_error.py
========================================
OTEL-003: Verify that OTel emits a warning when enabled but packages
are not installed.
"""
from __future__ import annotations

import sys
from unittest.mock import patch


class TestOtelImportErrorWarning:

    def test_warning_when_otel_enabled_but_not_installed(self, caplog):
        """When SACV_OTEL_ENABLED=true but opentelemetry is missing, emit warning."""
        # Temporarily set the env var
        env_patch = patch.dict("os.environ", {"SACV_OTEL_ENABLED": "true"})
        env_patch.start()

        # Remove opentelemetry modules from sys.modules so the import fails
        removed = {}
        for key in list(sys.modules):
            if "opentelemetry" in key:
                removed[key] = sys.modules.pop(key)

        try:
            # Force reimport of tracing.py to pick up the env var
            import importlib
            import sacv.tracing
            importlib.reload(sacv.tracing)

            # Should have _HAS_OTEL=False and a warning in logs
            assert sacv.tracing._HAS_OTEL is False
            assert any("OTel enabled" in record.message for record in caplog.records), (
                f"Expected OTel warning in logs, got: {[r.message for r in caplog.records]}"
            )
        finally:
            env_patch.stop()
            # Restore removed modules
            sys.modules.update(removed)
