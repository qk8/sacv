"""
tests/unit/test_mcp_transport.py
=================================
Unit tests for the MCP JSON-RPC transport layer.

Tests cover:
1. McpStdioTransport — initialization handshake parsing
2. JSON-RPC request/response parsing
3. Error handling — malformed JSON, missing result, RPC errors
4. _on_failure hook
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sacv.adapters.mcp_transport import McpStdioTransport


@pytest.fixture
def transport():
    """Create a transport with mocked subprocess."""
    class TestTransport(McpStdioTransport):
        _log_prefix = "test"
        _default_cmd = ["test-server"]
        _TIMEOUT_SEC = 2

    t = TestTransport()
    # Mock the subprocess
    mock_proc = AsyncMock()
    mock_proc.stdin = AsyncMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.returncode = None
    t._proc = mock_proc
    return t


class TestMcpStdioTransportInitialization:

    async def test_initialization_request_format(self, transport):
        """The initialize request has the correct JSON-RPC format."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        })

        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())
        transport._proc.stdin.drain = AsyncMock()

        # Manually call _initialize since __aenter__ would need a real subprocess
        await transport._initialize()

        # Check that stdin was written to
        assert transport._proc.stdin.write.called

    async def test_initialization_response_parsing(self, transport):
        """Valid initialize response is parsed correctly."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        })

        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        await transport._initialize()

        # initialized notification should have been sent
        assert transport._proc.stdin.write.called

    async def test_invalid_json_raises_runtime_error(self, transport):
        """Non-JSON initialize response raises RuntimeError."""
        transport._proc.stdout.readline = AsyncMock(return_value=b"not json")

        with pytest.raises(RuntimeError, match="not JSON"):
            await transport._initialize()

    async def test_error_in_response_raises_runtime_error(self, transport):
        """Initialize response with 'error' key raises RuntimeError."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
            "error": {"code": -32600, "message": "Invalid request"},
        })

        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        with pytest.raises(RuntimeError, match="Invalid request"):
            await transport._initialize()

    async def test_missing_result_raises_runtime_error(self, transport):
        """Initialize response without 'result' key raises RuntimeError."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
        })

        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        with pytest.raises(RuntimeError, match="missing 'result'"):
            await transport._initialize()

    async def test_server_info_logged(self, transport):
        """Server name and version are extracted from initialize response."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "my-server", "version": "2.0.0"},
            },
        })

        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        await transport._initialize()

        # The initialized notification should be written
        writes = transport._proc.stdin.write.call_args_list
        assert len(writes) == 2  # initialize request + initialized notification


class TestMcpStdioTransportCall:

    async def test_call_increments_request_id(self, transport):
        """Each _call increments the request ID."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": "[]"}]},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("test_tool", {"arg": "value"})

        assert transport._req_id == 1

    async def test_call_sends_tools_call_method(self, transport):
        """_call sends a tools/call JSON-RPC request."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": "[]"}]},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        await transport._call("my_tool", {"key": "val"})

        # Check the written payload
        write_call = transport._proc.stdin.write.call_args_list[0]
        payload = json.loads(write_call[0][0].decode())
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "my_tool"
        assert payload["params"]["arguments"] == {"key": "val"}

    async def test_call_parses_text_result(self, transport):
        """Text result from MCP tool is parsed from JSON string."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": '{"key": "value"}'}]},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("tool", {})
        assert result == {"key": "value"}

    async def test_call_returns_raw_text_on_json_decode_error(self, transport):
        """Non-JSON text result is returned as-is."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": "plain text response"}]},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("tool", {})
        assert result == "plain text response"

    async def test_call_handles_error_response(self, transport):
        """RPC error response returns _on_failure() result."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Tool not found"},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("missing_tool", {})
        assert result is None  # default _on_failure returns None

    async def test_call_handles_empty_content(self, transport):
        """Empty content list returns _on_failure() result."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": []},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("tool", {})
        assert result is None

    async def test_call_handles_non_list_content(self, transport):
        """Non-list content returns _on_failure() result."""
        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": "not a list"},
        })
        transport._proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        result = await transport._call("tool", {})
        assert result is None

    async def test_on_failure_returns_none_by_default(self):
        """Base _on_failure returns None."""
        t = McpStdioTransport()
        assert t._on_failure() is None

    async def test_custom_on_failure_override(self):
        """Subclasses can override _on_failure."""
        class CustomTransport(McpStdioTransport):
            _log_prefix = "custom"
            _default_cmd = ["custom"]
            _TIMEOUT_SEC = 2

            def _on_failure(self):
                return {"fallback": True}

        t = CustomTransport()
        assert t._on_failure() == {"fallback": True}


class TestMcpStdioTransportReconnect:

    async def test_ensure_running_returns_true_when_alive(self, transport):
        """Returns True when subprocess is running."""
        transport._proc.returncode = None
        result = await transport._ensure_running()
        assert result is True

    async def test_ensure_running_attempts_reconnect_when_dead(self, transport):
        """Attempts reconnect when subprocess has exited."""
        transport._proc.returncode = 1

        mock_start = AsyncMock(return_value=None)
        transport.start = mock_start
        transport.start.__name__ = "start"

        # Mock the start method to succeed
        original_start = transport.start
        async def fake_start():
            transport._proc.returncode = None
        transport.start = fake_start

        result = await transport._ensure_running()
        # Should have attempted start (returns True if start succeeds)
        assert result is True


class TestMcpStdioTransportLifecycle:

    async def test_start_creates_subprocess(self):
        """start() creates a subprocess with correct arguments."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["test-server", "--arg"]
            _TIMEOUT_SEC = 2

        t = TestTransport()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = AsyncMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.pid = 1234
            mock_exec.return_value = mock_proc

            # Mock _initialize to avoid actual protocol
            t._initialize = AsyncMock()

            await t.start()

            mock_exec.assert_called_once_with(
                "test-server", "--arg",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert t._proc == mock_proc

    async def test_stop_terminates_subprocess(self):
        """stop() terminates and waits for subprocess."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["test-server"]
            _TIMEOUT_SEC = 2

        t = TestTransport()
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock()
        t._proc = mock_proc

        await t.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    async def test_init_sets_lock(self):
        """__init__ creates an asyncio.Lock."""
        t = McpStdioTransport()
        assert isinstance(t._lock, asyncio.Lock)

    async def test_init_sets_req_id_to_zero(self):
        """__init__ sets _req_id to 0."""
        t = McpStdioTransport()
        assert t._req_id == 0

    async def test_custom_server_cmd(self):
        """Custom server command overrides default."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["default-cmd"]
            _TIMEOUT_SEC = 2

        t = TestTransport(server_cmd=["custom", "args"])
        assert t._cmd == ["custom", "args"]

    async def test_default_cmd_copied_from_class(self):
        """Default command is copied from class attribute."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["default", "value"]
            _TIMEOUT_SEC = 2

        t = TestTransport()
        assert t._cmd == ["default", "value"]


class TestMcpStdioTransportTOCTOU:

    async def test_reconnects_on_write_failure(self):
        """When subprocess dies between _ensure_running check and write,
        the transport should reconnect and retry."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["test-server"]
            _TIMEOUT_SEC = 2

            def _on_failure(self):
                return {"fallback": True}

        t = TestTransport()

        # Simulate TOCTOU: process dies between liveness check and write.
        # First _ensure_running sees returncode=None (appears alive).
        # Write fails with OSError. Second _ensure_running sees returncode=1
        # (OS updated it) and reconnects.
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdin.write.side_effect = OSError("Broken pipe")
        mock_proc.stdout = AsyncMock()
        t._proc = mock_proc

        mock_new_proc = AsyncMock()
        mock_new_proc.stdin = AsyncMock()
        mock_new_proc.stdin.drain = AsyncMock()
        mock_new_proc.stdout = AsyncMock()
        mock_new_proc.returncode = None

        mock_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": '{"ok": true}'}]},
        })
        mock_new_proc.stdout.readline = AsyncMock(return_value=(mock_response + "\n").encode())

        reconnect_count = 0

        async def fake_start():
            nonlocal reconnect_count
            reconnect_count += 1
            t._proc = mock_new_proc

        t.start = fake_start

        # Override _ensure_running to simulate TOCTOU: first call returns True
        # (process appears alive), second call triggers reconnect.
        _ensure_call_count = 0

        async def custom_ensure_running(self_param: "TestTransport") -> bool:
            nonlocal _ensure_call_count
            _ensure_call_count += 1
            if _ensure_call_count == 1:
                # First call: TOCTOU window, process appears alive
                return True
            # Second call (from OSError handler): trigger reconnect
            await t.start()
            return True

        t._ensure_running = custom_ensure_running.__get__(t, TestTransport)

        result = await t._call("test_tool", {})

        assert reconnect_count == 1
        assert result == {"ok": True}

    async def test_returns_failure_after_reconnect_fails(self):
        """When subprocess dies and reconnect also fails, return _on_failure()."""
        class TestTransport(McpStdioTransport):
            _log_prefix = "test"
            _default_cmd = ["test-server"]
            _TIMEOUT_SEC = 2

            def _on_failure(self):
                return {"fallback": True}

        t = TestTransport()

        mock_dead_proc = AsyncMock()
        mock_dead_proc.returncode = 1
        t._proc = mock_dead_proc

        # Reconnect also fails
        async def fake_start():
            raise RuntimeError("reconnect failed")

        t.start = fake_start

        result = await t._call("test_tool", {})

        assert result == {"fallback": True}
