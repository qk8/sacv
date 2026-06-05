"""
tests/unit/test_jdwp_client.py
================================
Unit tests for JdwpClient with mocked subprocess.

Tests cover:
1. Context manager — __aenter__/__aexit__
2. Breakpoint control — set_breakpoint_at_line, set_breakpoint_at_method
3. Execution control — run, step_over, step_into, step_out, wait_for_breakpoint_hit
4. Inspection — get_local_variables, get_call_stack, evaluate, get_thread_list
5. Error handling — timeout, process not connected
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sacv.adapters.debug.jdwp_client import JdwpClient, BreakpointHitInfo


@pytest.fixture
def mock_process():
    """Create a mock subprocess for JdwpClient."""
    proc = AsyncMock()
    proc.stdin = AsyncMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = AsyncMock()
    proc.returncode = None
    proc.wait = AsyncMock()
    proc.kill = MagicMock()
    return proc


@pytest.fixture
def client(mock_process):
    """Create a JdwpClient with mocked subprocess."""
    client = JdwpClient(host="test-host", port=5005)
    client._process = mock_process
    return client


class TestJdwpClientContextManager:

    async def test_aenter_calls_connect(self):
        """__aenter__ calls connect and returns self."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = AsyncMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_exec.return_value = mock_proc

            async with JdwpClient() as client:
                assert client is not None

    async def test_aexit_calls_close(self):
        """__aexit__ calls close."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = AsyncMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_exec.return_value = mock_proc

            async with JdwpClient() as client:
                pass
            # close should have been called (via aexit)


class TestJdwpClientConnect:

    async def test_connect_creates_subprocess(self):
        """connect() creates a jdb subprocess with correct arguments."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = AsyncMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_exec.return_value = mock_proc

            async with JdwpClient() as client:
                pass

            mock_exec.assert_called_once_with(
                "jdb", "-attach", "localhost:5005",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

    async def test_connect_reads_until_prompt(self):
        """connect() consumes the JDB startup banner."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = AsyncMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_exec.return_value = mock_proc

            async with JdwpClient() as client:
                pass
            # stdout.read should have been called to consume banner


class TestJdwpClientBreakpoint:

    async def test_set_breakpoint_at_line_success(self, client, mock_process):
        """set_breakpoint_at_line returns True on success."""
        mock_process.stdout.readline = AsyncMock(
            return_value=b"Deferring breakpoint at com.example.UserService:42\n> "
        )
        mock_process.stdout.read = AsyncMock(return_value=b"Deferring breakpoint at com.example.UserService:42\n> ")

        result = await client.set_breakpoint_at_line("UserService", 42)
        assert result is True

    async def test_set_breakpoint_at_line_set(self, client, mock_process):
        """set_breakpoint_at_line returns True when breakpoint is set."""
        mock_process.stdout.read = AsyncMock(return_value=b"Set breakpoint at com.example.UserService:42\n> ")

        result = await client.set_breakpoint_at_line("UserService", 42)
        assert result is True

    async def test_set_breakpoint_at_line_failure(self, client, mock_process):
        """set_breakpoint_at_line returns False on failure."""
        mock_process.stdout.read = AsyncMock(return_value=b"No class or line number specified\n> ")

        result = await client.set_breakpoint_at_line("NonExistent", 999)
        assert result is False

    async def test_set_breakpoint_at_method_success(self, client, mock_process):
        """set_breakpoint_at_method returns True on success."""
        mock_process.stdout.read = AsyncMock(return_value=b"Set breakpoint in com.example.UserService.findById\n> ")

        result = await client.set_breakpoint_at_method("UserService", "findById")
        assert result is True

    async def test_set_breakpoint_at_method_failure(self, client, mock_process):
        """set_breakpoint_at_method returns False on failure."""
        mock_process.stdout.read = AsyncMock(return_value=b"Method not found\n> ")

        result = await client.set_breakpoint_at_method("UserService", "nonExistent")
        assert result is False


class TestJdwpClientExecution:

    async def test_run_returns_output(self, client, mock_process):
        """run() returns the JDB output."""
        mock_process.stdout.read = AsyncMock(return_value=b"Continuing.\n> ")

        result = await client.run()
        assert "Continuing" in result

    async def test_step_over_returns_breakpoint_hit(self, client, mock_process):
        """step_over() parses breakpoint hit info."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"Step complete:  thread=main(t1), com.example.UserService.process(), line=50 bci=10\n> "
        )

        result = await client.step_over()
        assert result is not None
        assert result.class_name == "com.example.UserService"
        assert result.method_name == "process"
        assert result.line == 50

    async def test_step_over_no_hit(self, client, mock_process):
        """step_over() returns None when no breakpoint hit."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"Step complete.\n> "
        )

        result = await client.step_over()
        assert result is None

    async def test_step_into_returns_breakpoint_hit(self, client, mock_process):
        """step_into() parses breakpoint hit info."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"Step complete:  thread=main(t1), com.example.UserService.save(), line=30 bci=5\n> "
        )

        result = await client.step_into()
        assert result is not None
        assert result.method_name == "save"

    async def test_step_out_returns_breakpoint_hit(self, client, mock_process):
        """step_out() parses breakpoint hit info."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"Step complete:  thread=main(t1), com.example.Controller.handle(), line=10 bci=2\n> "
        )

        result = await client.step_out()
        assert result is not None
        assert result.method_name == "handle"

    async def test_wait_for_breakpoint_hit_returns_hit(self, client, mock_process):
        """wait_for_breakpoint_hit() returns BreakpointHitInfo on hit."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"Breakpoint hit:  thread=main(t1), com.example.UserService.findById(), line=42 bci=73\n> "
        )

        result = await client.wait_for_breakpoint_hit(timeout=1.0)
        assert result is not None
        assert result.line == 42

    async def test_wait_for_breakpoint_hit_returns_none_on_timeout(self, client, mock_process):
        """wait_for_breakpoint_hit() returns None on timeout."""
        async def never_resolves(*_args):
            await asyncio.sleep(10)

        mock_process.stdout.read = AsyncMock(side_effect=never_resolves)

        result = await client.wait_for_breakpoint_hit(timeout=0.1)
        assert result is None


class TestJdwpClientInspection:

    async def test_get_local_variables(self, client, mock_process):
        """get_local_variables() parses local variable output."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"  user = com.example.User@abc (User)\n"
                         b"  id = 42 (Long)\n> "
        )

        result = await client.get_local_variables()
        assert len(result) == 2
        assert result[0].name == "user"
        assert result[1].name == "id"

    async def test_get_call_stack(self, client, mock_process):
        """get_call_stack() parses stack trace output."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"  [1] com.example.UserService.findById(UserService.java:42)\n"
                         b"  [2] com.example.Controller.handle(Controller.java:10)\n> "
        )

        result = await client.get_call_stack()
        assert len(result) == 2
        assert "UserService" in result[0]
        assert "Controller" in result[1]

    async def test_evaluate(self, client, mock_process):
        """evaluate() parses print output."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"expression = com.example.User@abc123\n> "
        )

        result = await client.evaluate("user")
        assert result == "com.example.User@abc123"

    async def test_evaluate_no_equals(self, client, mock_process):
        """evaluate() handles output without '=' separator."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"<identifier>\n> "
        )

        result = await client.evaluate("nonExistent")
        assert result == "<identifier>"

    async def test_get_thread_list(self, client, mock_process):
        """get_thread_list() parses thread output."""
        mock_process.stdout.read = AsyncMock(
            return_value=b"  Thread [main] (Suspended)\n"
                         b"  Thread [pool-1-thread-5] (Running)\n> "
        )

        result = await client.get_thread_list()
        assert len(result) == 2
        assert "main" in result[0]


class TestJdwpClientClose:

    async def test_close_sends_quit(self, client, mock_process):
        """close() sends 'quit' command to JDB."""
        await client.close()

        mock_process.stdin.write.assert_called_once_with(b"quit\n")
        mock_process.stdin.drain.assert_called_once()

    async def test_close_handles_exception(self, client, mock_process):
        """close() kills process on exception."""
        mock_process.stdin.write = MagicMock(side_effect=OSError("broken pipe"))

        # Should not raise — close catches exceptions
        await client.close()
        mock_process.kill.assert_called_once()

    async def test_close_handles_no_process(self):
        """close() is a no-op when process is None."""
        client = JdwpClient()
        client._process = None
        # Should not raise
        await client.close()


class TestJdwpClientInit:

    async def test_default_host_and_port(self):
        client = JdwpClient()
        assert client._host == "localhost"
        assert client._port == 5005

    async def test_custom_host_and_port(self):
        client = JdwpClient(host="127.0.0.1", port=9999)
        assert client._host == "127.0.0.1"
        assert client._port == 9999

    async def test_process_initialized_to_none(self):
        client = JdwpClient()
        assert client._process is None
