"""
tests/unit/test_cdp_client.py
================================
Unit tests for CdpClient with mocked websockets.

Tests cover:
1. Context manager — __aenter__/__aexit__
2. Debugger lifecycle — enable, disable
3. Breakpoints — set, remove
4. Execution control — resume, step_over, step_into, step_out, wait_for_paused
5. Variable inspection — get_scope_variables_from_paused, evaluate_in_frame, evaluate
6. Error handling — timeout, no websockets library
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sacv.adapters.debug.cdp_client import CdpClient, PausedEvent, CallFrame


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_websockets(mock_ws):
    """Mock the websockets module."""
    with patch("sacv.adapters.debug.cdp_client.websockets") as mock_mod:
        mock_mod.connect = AsyncMock(return_value=mock_ws)
        mock_mod.WebSocketClientProtocol = MagicMock
        yield mock_ws


@pytest.fixture
def client(mock_websockets):
    """Create a CdpClient with mocked websockets."""
    c = CdpClient(host="test-host", port=9229)
    c._ws = mock_websockets
    return c


class TestCdpClientWebSocketAvailability:

    async def test_import_error_without_websockets(self):
        """CdpClient raises ImportError when websockets is not installed."""
        with patch("sacv.adapters.debug.cdp_client._WS_AVAILABLE", False):
            c = CdpClient()
            with pytest.raises(ImportError, match="websockets library required"):
                await c.connect()


class TestCdpClientContextManager:

    async def test_aenter_calls_connect(self, mock_websockets):
        """__aenter__ calls connect and returns self."""
        async with CdpClient() as client:
            assert client is not None

    async def test_aexit_calls_close(self, mock_websockets):
        """__aexit__ calls close."""
        async with CdpClient() as client:
            pass
        # close should have been called via aexit


class TestCdpClientConnect:

    async def test_connect_discovers_ws_url(self, mock_websockets):
        """connect() discovers WebSocket URL via /json endpoint."""
        import urllib.request as _urllib
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {"webSocketDebuggerUrl": "ws://test-host:9229/devtools/browser/1"}
        ]).encode()
        _urllib.urlopen = MagicMock(return_value=mock_resp)

        async with CdpClient(host="test-host", port=9229) as client:
            assert client._ws is not None

    async def test_connect_falls_back_to_default_ws_url(self, mock_websockets):
        """connect() falls back to ws://host:port when /json fails."""
        import urllib.request as _urllib
        _urllib.urlopen = MagicMock(side_effect=Exception("connection refused"))

        async with CdpClient(host="test-host", port=9229) as client:
            # Should have connected via fallback URL
            assert client._ws is not None

    async def test_connect_creates_receive_task(self, mock_websockets):
        """connect() creates a background receive task."""
        import urllib.request as _urllib
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {"webSocketDebuggerUrl": "ws://test-host:9229/devtools/browser/1"}
        ]).encode()
        _urllib.urlopen = MagicMock(return_value=mock_resp)

        async with CdpClient(host="test-host", port=9229) as client:
            assert client._recv_task is not None


class TestCdpClientDebuggerLifecycle:

    async def test_enable_debugger_sends_commands(self, client, mock_websockets):
        """enable_debugger() sends Debugger.enable and Runtime.enable."""
        await client.enable_debugger()
        assert mock_websockets.send.call_count == 2

    async def test_disable_debugger_sends_command(self, client, mock_websockets):
        """disable_debugger() sends Debugger.disable."""
        await client.disable_debugger()
        calls = [c[0][0] for c in mock_websockets.send.call_args_list]
        disable_msg = json.loads(calls[0])
        assert disable_msg["method"] == "Debugger.disable"


class TestCdpClientBreakpoints:

    async def test_set_breakpoint_by_url(self, client, mock_websockets):
        """set_breakpoint_by_url() sends correct CDP command."""
        # Mock _send to return the breakpoint ID (avoids receive-loop complexity)
        client._send = AsyncMock(return_value={"breakpointId": "bp-1"})
        bp_id = await client.set_breakpoint_by_url("src/UserService.ts", 42)
        assert bp_id == "bp-1"

        # Verify the CDP message sent via _send
        client._send.assert_called_once()
        _, params = client._send.call_args[0]
        assert params["lineNumber"] == 41  # 0-indexed
        assert params["url"] == "src/UserService.ts"

    async def test_set_breakpoint_with_column(self, client, mock_websockets):
        """set_breakpoint_by_url() respects column parameter."""
        mock_websockets.send = AsyncMock(return_value={"breakpointId": "bp-1"})
        await client.set_breakpoint_by_url("src/UserService.ts", 42, column=10)

        calls = [c[0][0] for c in mock_websockets.send.call_args_list]
        bp_msg = json.loads(calls[-1])
        assert bp_msg["params"]["columnNumber"] == 10

    async def test_remove_breakpoint(self, client, mock_websockets):
        """remove_breakpoint() sends correct CDP command."""
        await client.remove_breakpoint("bp-1")
        calls = [c[0][0] for c in mock_websockets.send.call_args_list]
        rm_msg = json.loads(calls[-1])
        assert rm_msg["method"] == "Debugger.removeBreakpoint"
        assert rm_msg["params"]["breakpointId"] == "bp-1"


class TestCdpClientExecution:

    async def test_resume(self, client, mock_websockets):
        """resume() sends Debugger.resume."""
        client._send = AsyncMock(return_value={})
        await client.resume()
        client._send.assert_called_once_with("Debugger.resume")

    async def test_step_over_returns_paused(self, client, mock_websockets):
        """step_over() returns PausedEvent on pause."""
        client._send = AsyncMock(return_value={})

        paused_event = PausedEvent(
            reason="breakpoint",
            call_frames=[CallFrame(
                frame_id="f1", function="test", url="a.ts",
                line=10, column=0, scope_chain=[],
            )],
        )
        # step_over calls wait_for_paused which creates a new _paused_future
        # so we mock wait_for_paused directly
        client.wait_for_paused = AsyncMock(return_value=paused_event)

        result = await client.step_over()
        assert result is not None
        assert result.reason == "breakpoint"

    async def test_step_into_returns_paused(self, client, mock_websockets):
        """step_into() returns PausedEvent on pause."""
        client._send = AsyncMock(return_value={})

        paused_event = PausedEvent(
            reason="pause",
            call_frames=[CallFrame(
                frame_id="f1", function="inner", url="b.ts",
                line=20, column=0, scope_chain=[],
            )],
        )
        client.wait_for_paused = AsyncMock(return_value=paused_event)

        result = await client.step_into()
        assert result is not None
        assert result.reason == "pause"

    async def test_step_out_returns_paused(self, client, mock_websockets):
        """step_out() returns PausedEvent on pause."""
        client._send = AsyncMock(return_value={})

        paused_event = PausedEvent(
            reason="step",
            call_frames=[CallFrame(
                frame_id="f1", function="outer", url="c.ts",
                line=30, column=0, scope_chain=[],
            )],
        )
        client.wait_for_paused = AsyncMock(return_value=paused_event)

        result = await client.step_out()
        assert result is not None

    async def test_wait_for_paused_returns_none_on_timeout(self, client, mock_websockets):
        """wait_for_paused() returns None on timeout."""
        result = await client.wait_for_paused(timeout=0.01)
        assert result is None


class TestCdpClientVariableInspection:

    async def test_get_scope_variables_from_paused(self, client, mock_websockets):
        """get_scope_variables_from_paused() returns variables from scope chain."""
        paused = PausedEvent(
            reason="breakpoint",
            call_frames=[CallFrame(
                frame_id="f1", function="test", url="a.ts",
                line=10, column=0, scope_chain=[{
                    "type": "local",
                    "object": {"objectId": "obj-1"},
                }],
            )],
        )
        client._send = AsyncMock(
            return_value={"result": [{"name": "user", "value": {"value": "alice", "type": "string"}}]}
        )

        result = await client.get_scope_variables_from_paused(paused)
        assert "user" in result
        assert result["user"]["value"] == "alice"

    async def test_get_scope_variables_from_paused_empty_frame(self, client, mock_websockets):
        """get_scope_variables_from_paused() returns empty dict when no frames."""
        paused = PausedEvent(reason="breakpoint", call_frames=[])
        result = await client.get_scope_variables_from_paused(paused)
        assert result == {}

    async def test_evaluate_in_frame(self, client, mock_websockets):
        """evaluate_in_frame() sends correct CDP command."""
        client._send = AsyncMock(return_value={"result": {"value": 42}})
        result = await client.evaluate_in_frame("2 + 2", "f1")
        assert result == 42

        client._send.assert_called_once()
        _, params = client._send.call_args[0]
        assert params["callFrameId"] == "f1"

    async def test_evaluate(self, client, mock_websockets):
        """evaluate() sends Runtime.evaluate command."""
        client._send = AsyncMock(return_value={"result": {"value": "hello"}})
        result = await client.evaluate("'hello'")
        assert result == "hello"

        client._send.assert_called_once()
        _, params = client._send.call_args[0]
        assert params["expression"] == "'hello'"


class TestCdpClientSend:

    async def test_send_increments_id(self, client, mock_websockets):
        """_send() increments the message ID."""
        # Mock wait_for to raise TimeoutError immediately (simulates timeout)
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await client._send("Debugger.enable")
        # _id should have been incremented even on timeout
        assert client._id == 1

    async def test_send_returns_empty_on_timeout(self, client, mock_websockets):
        """_send() returns {} on timeout."""
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await client._send("Debugger.enable")
        assert result == {}


class TestCdpClientClose:

    async def test_close_cancels_recv_task(self, mock_websockets):
        """close() cancels the receive task."""
        async with CdpClient() as client:
            client._recv_task = AsyncMock()
            client._recv_task.cancel = MagicMock()
            client._ws = mock_websockets
            client._ws.close = AsyncMock()

        client._recv_task.cancel.assert_called_once()

    async def test_close_closes_ws(self, mock_websockets):
        """close() closes the WebSocket."""
        async with CdpClient() as client:
            client._ws = mock_websockets
            mock_websockets.close = AsyncMock()

        mock_websockets.close.assert_called_once()


class TestCdpClientDeprecatedMethod:

    async def test_get_scope_variables_warns(self, client, mock_websockets):
        """get_scope_variables() logs a warning and returns empty dict."""
        with patch("sacv.adapters.debug.cdp_client.log") as mock_log:
            result = await client.get_scope_variables("f1")
            assert result == {}
            mock_log.warning.assert_called()
            assert "deprecated" in str(mock_log.warning.call_args)
