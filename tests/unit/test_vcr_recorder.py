"""
tests/unit/test_vcr_recorder.py
================================
Unit tests for VCRAgentProvider — VCR cassette recorder/replayer.

Tests cover:
1. Record mode appends recordings
2. Replay mode returns stored results in order
3. Exhausted replay raises IndexError
4. save_cassette writes JSON to disk
5. create_subagent shares cassette and recordings
6. Replay mode with missing cassette raises FileNotFoundError
7. Prompt hash is stored (SHA-256, first 12 chars)
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
import tempfile

from sacv.testing.vcr_recorder import VCRAgentProvider, FIXTURES_DIR
from sacv.interfaces.agent_provider import AgentProvider, AgentConfig, AgentResult


class _RealAgent(AgentProvider):
    """Minimal agent that returns predictable results."""

    def __init__(self, results: list[AgentResult]) -> None:
        self._results = iter(results)

    async def run_task(self, prompt: str, context: dict, config: AgentConfig) -> AgentResult:
        return next(self._results)

    async def create_subagent(self, config: AgentConfig) -> AgentProvider:
        return self


def _make_result(content: str, tokens: int = 10) -> AgentResult:
    return AgentResult(
        content=content, tool_calls=[], finish_reason="stop",
        input_tokens=tokens, output_tokens=tokens,
    )


class TestVCRRecordMode:

    async def test_record_mode_appends_results(self):
        provider = _RealAgent([
            _make_result("hello"),
            _make_result("world"),
        ])
        vcr = VCRAgentProvider(provider, "test_record", mode="record")
        cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

        r1 = await vcr.run_task("p1", {}, cfg)
        assert r1.content == "hello"

        r2 = await vcr.run_task("p2", {}, cfg)
        assert r2.content == "world"

        assert len(vcr._recordings) == 2

    async def test_record_mode_stores_prompt_hash(self):
        provider = _RealAgent([_make_result("ok")])
        vcr = VCRAgentProvider(provider, "test_hash", mode="record")
        cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

        await vcr.run_task("my_prompt", {}, cfg)
        assert len(vcr._recordings[0]["prompt_hash"]) == 12

    async def test_record_mode_stores_result_fields(self):
        provider = _RealAgent([_make_result("resp", tokens=5)])
        vcr = VCRAgentProvider(provider, "test_fields", mode="record")
        cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

        await vcr.run_task("p", {}, cfg)
        entry = vcr._recordings[0]["result"]
        assert entry["content"] == "resp"
        assert entry["tool_calls"] == []
        assert entry["finish_reason"] == "stop"
        assert entry["input_tokens"] == 5
        assert entry["output_tokens"] == 5

    async def test_record_mode_with_tool_calls(self):
        provider = _RealAgent([
            AgentResult(
                content="", tool_calls=[{"name": "read_file", "args": {"path": "x.py"}}],
                finish_reason="tool_calls", input_tokens=10, output_tokens=5,
            ),
        ])
        vcr = VCRAgentProvider(provider, "test_tools", mode="record")
        cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

        await vcr.run_task("p", {}, cfg)
        entry = vcr._recordings[0]["result"]
        assert len(entry["tool_calls"]) == 1
        assert entry["tool_calls"][0]["name"] == "read_file"


class TestVCRReplayMode:

    async def test_replay_returns_stored_results_in_order(self, tmp_path):
        cassette_data = [
            {"prompt_hash": "abc", "result": {
                "content": "first", "tool_calls": [], "finish_reason": "stop",
                "input_tokens": 5, "output_tokens": 5,
            }},
            {"prompt_hash": "def", "result": {
                "content": "second", "tool_calls": [], "finish_reason": "stop",
                "input_tokens": 10, "output_tokens": 10,
            }},
        ]
        cassette_file = tmp_path / "test_replay.json"
        cassette_file.write_text(json.dumps(cassette_data))

        # Temporarily override FIXTURES_DIR for isolation
        import sacv.testing.vcr_recorder as vcr_mod
        original = vcr_mod.FIXTURES_DIR
        try:
            vcr_mod.FIXTURES_DIR = tmp_path
            vcr = VCRAgentProvider(None, "test_replay", mode="replay")
            cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

            r1 = await vcr.run_task("p1", {}, cfg)
            assert r1.content == "first"

            r2 = await vcr.run_task("p2", {}, cfg)
            assert r2.content == "second"
        finally:
            vcr_mod.FIXTURES_DIR = original

    async def test_replay_exhausted_raises_index_error(self, tmp_path):
        cassette_data = [
            {"prompt_hash": "x", "result": {
                "content": "only_one", "tool_calls": [], "finish_reason": "stop",
                "input_tokens": 1, "output_tokens": 1,
            }},
        ]
        cassette_file = tmp_path / "test_exhausted.json"
        cassette_file.write_text(json.dumps(cassette_data))

        import sacv.testing.vcr_recorder as vcr_mod
        original = vcr_mod.FIXTURES_DIR
        try:
            vcr_mod.FIXTURES_DIR = tmp_path
            vcr = VCRAgentProvider(None, "test_exhausted", mode="replay")
            cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

            await vcr.run_task("p", {}, cfg)  # first call OK
            with pytest.raises(IndexError, match="exhausted"):
                await vcr.run_task("p", {}, cfg)  # second call fails
        finally:
            vcr_mod.FIXTURES_DIR = original

    async def test_replay_missing_cassette_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Cassette not found"):
            VCRAgentProvider(None, "nonexistent_cassette", mode="replay")


class TestVCRSaveCassette:

    async def test_save_cassette_writes_json(self, tmp_path):
        provider = _RealAgent([_make_result("save_test")])
        import sacv.testing.vcr_recorder as vcr_mod
        original = vcr_mod.FIXTURES_DIR
        try:
            vcr_mod.FIXTURES_DIR = tmp_path
            vcr = VCRAgentProvider(provider, "test_save", mode="record")
            cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

            await vcr.run_task("p", {}, cfg)
            vcr.save_cassette()

            cassette = tmp_path / "test_save.json"
            assert cassette.exists()
            data = json.loads(cassette.read_text())
            assert len(data) == 1
            assert data[0]["result"]["content"] == "save_test"
        finally:
            vcr_mod.FIXTURES_DIR = original


class TestVCRSubagent:

    async def test_subagent_shares_cassette(self):
        provider = _RealAgent([_make_result("child_result")])
        vcr = VCRAgentProvider(provider, "test_sub", mode="record")
        cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

        child = await vcr.create_subagent(cfg)

        assert child._cassette == vcr._cassette
        assert child._recordings is vcr._recordings
        assert child._mode == vcr._mode
        # Child has no independent provider
        assert child._provider is None

    async def test_subagent_record_mode_no_crash(self, tmp_path):
        """Subagent in replay mode shares cassette and state."""
        cassette_data: list = []
        cassette_file = tmp_path / "test_sub2.json"
        cassette_file.write_text(json.dumps(cassette_data))

        import sacv.testing.vcr_recorder as vcr_mod
        original = vcr_mod.FIXTURES_DIR
        try:
            vcr_mod.FIXTURES_DIR = tmp_path
            vcr = VCRAgentProvider(None, "test_sub2", mode="replay")
            cfg = AgentConfig(role="test", system_prompt="", max_turns=1, allowed_tools=[])

            child = await vcr.create_subagent(cfg)
            assert child._replay_index == vcr._replay_index
            assert child._cassette == vcr._cassette
            assert child._recordings is vcr._recordings
        finally:
            vcr_mod.FIXTURES_DIR = original
