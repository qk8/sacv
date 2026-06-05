"""
tests/unit/test_cdp_parsers.py
================================
Unit tests for pure parser functions in cdp_client.py.

Tests cover:
1. _parse_paused — CDP Debugger.paused event parsing
"""
from __future__ import annotations

import pytest
from sacv.adapters.debug.cdp_client import _parse_paused, PausedEvent, CallFrame


class TestParsePaused:

    def test_parses_breakpoint_pause(self):
        params = {
            "reason": "breakpoint",
            "hitBreakpoints": ["bp-1"],
            "callFrames": [
                {
                    "callFrameId": "frame-1",
                    "functionName": "findById",
                    "url": "file:///workspace/src/UserService.ts",
                    "location": {"lineNumber": 41, "columnNumber": 10},
                    "scopeChain": [],
                },
            ],
        }
        result = _parse_paused(params)
        assert isinstance(result, PausedEvent)
        assert result.reason == "breakpoint"
        assert result.hit_breakpoints == ["bp-1"]
        assert len(result.call_frames) == 1
        frame = result.call_frames[0]
        assert frame.frame_id == "frame-1"
        assert frame.function == "findById"
        assert frame.url == "file:///workspace/src/UserService.ts"
        assert frame.line == 42  # converted from 0-indexed
        assert frame.column == 10

    def test_parses_pause_without_breakpoints(self):
        params = {
            "reason": "other",
            "callFrames": [
                {
                    "callFrameId": "f1",
                    "functionName": "<anonymous>",
                    "url": "file:///workspace/app.js",
                    "location": {"lineNumber": 0, "columnNumber": 0},
                    "scopeChain": [],
                },
            ],
        }
        result = _parse_paused(params)
        assert result.reason == "other"
        assert result.hit_breakpoints == []
        assert result.call_frames[0].line == 1  # 0-indexed + 1

    def test_parses_multiple_call_frames(self):
        params = {
            "reason": "breakpoint",
            "callFrames": [
                {
                    "callFrameId": "f1",
                    "functionName": "outer",
                    "url": "file:///workspace/a.ts",
                    "location": {"lineNumber": 10, "columnNumber": 0},
                    "scopeChain": [],
                },
                {
                    "callFrameId": "f2",
                    "functionName": "inner",
                    "url": "file:///workspace/b.ts",
                    "location": {"lineNumber": 20, "columnNumber": 5},
                    "scopeChain": [],
                },
            ],
        }
        result = _parse_paused(params)
        assert len(result.call_frames) == 2
        assert result.call_frames[0].function == "outer"
        assert result.call_frames[1].function == "inner"

    def test_parses_scope_chain(self):
        params = {
            "reason": "breakpoint",
            "callFrames": [
                {
                    "callFrameId": "f1",
                    "functionName": "test",
                    "url": "file:///workspace/test.ts",
                    "location": {"lineNumber": 5, "columnNumber": 0},
                    "scopeChain": [
                        {
                            "type": "local",
                            "object": {"objectId": "obj-1"},
                            "name": "user",
                        },
                        {
                            "type": "global",
                            "object": {},
                            "name": "global",
                        },
                    ],
                },
            ],
        }
        result = _parse_paused(params)
        assert len(result.call_frames[0].scope_chain) == 2
        assert result.call_frames[0].scope_chain[0]["type"] == "local"

    def test_handles_missing_call_frames(self):
        params = {"reason": "exception", "callFrames": []}
        result = _parse_paused(params)
        assert result.reason == "exception"
        assert result.call_frames == []

    def test_handles_empty_params(self):
        result = _parse_paused({})
        assert result.reason == "breakpoint"  # default
        assert result.call_frames == []

    def test_handles_missing_location(self):
        params = {
            "reason": "breakpoint",
            "callFrames": [
                {
                    "callFrameId": "f1",
                    "functionName": "mystery",
                    "url": "file:///workspace/unknown.js",
                    "scopeChain": [],
                },
            ],
        }
        result = _parse_paused(params)
        frame = result.call_frames[0]
        assert frame.line == 1  # default lineNumber 0 + 1
        assert frame.column == 0  # default

    def test_handles_missing_function_name(self):
        params = {
            "reason": "breakpoint",
            "callFrames": [
                {
                    "callFrameId": "f1",
                    "url": "file:///workspace/index.ts",
                    "location": {"lineNumber": 0, "columnNumber": 0},
                    "scopeChain": [],
                },
            ],
        }
        result = _parse_paused(params)
        assert result.call_frames[0].function == "<anonymous>"

    def test_call_frame_id_property(self):
        params = {
            "reason": "breakpoint",
            "callFrames": [{"callFrameId": "unique-frame-id", "functionName": "x",
                            "url": "a.ts", "location": {"lineNumber": 0, "columnNumber": 0},
                            "scopeChain": []}],
        }
        result = _parse_paused(params)
        assert result.call_frame_id == "unique-frame-id"

    def test_call_frame_id_empty_when_no_frames(self):
        result = _parse_paused({"callFrames": []})
        assert result.call_frame_id == ""
