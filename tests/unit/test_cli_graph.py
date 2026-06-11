"""
tests/unit/test_cli_graph.py
=============================

Tests for the `sacv graph` CLI command (ARC-002).

Tests verify that graph topology export produces valid Mermaid, ASCII,
and JSON output containing all expected workflow nodes.
"""
from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.deps import NodeDeps
from sacv.orchestration.graph import build_graph
from sacv.testing.stub_providers import (
    StubAgentProvider,
    StubCodeGraphProvider,
    StubCrossDomainProvider,
    StubDiffProvider,
    StubGitProvider,
    StubMemoryProvider,
    StubSandboxProvider,
)


@pytest.fixture
def stub_graph():
    """Build the compiled workflow graph using stub providers."""
    stub_deps = NodeDeps(
        agent=StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )
    return build_graph(stub_deps, checkpointer=MemorySaver())


EXPECTED_NODES = [
    "bootstrap",
    "mode_router",
    "scout",
    "value_node",
    "tdd_gate",
    "actor",
    "preflight_node",
    "all_critics_router",
    "critic_security",
    "critic_style",
    "critic_consistency",
    "all_critics_merge",
    "verifier",
    "intelligent_debugger",
    "replan",
    "speculative_branch",
    "hitl_escalation",
    "memory_consolidation",
]


class TestGraphMermaidExport:

    def test_mermaid_output_contains_all_nodes(self, stub_graph):
        """Mermaid diagram includes every workflow node."""
        mermaid = stub_graph.get_graph().draw_mermaid()
        for node in EXPECTED_NODES:
            assert node in mermaid, f"Node '{node}' missing from Mermaid output"

    def test_mermaid_output_contains_edges(self, stub_graph):
        """Mermaid diagram contains edge definitions (-->)."""
        mermaid = stub_graph.get_graph().draw_mermaid()
        assert "-->" in mermaid


class TestGraphAsciiExport:

    @pytest.mark.skip(reason="grandalf not installed")
    def test_ascii_output_contains_all_nodes(self, stub_graph):
        """ASCII diagram includes every workflow node."""
        ascii_out = stub_graph.get_graph().draw_ascii()
        for node in EXPECTED_NODES:
            assert node in ascii_out, f"Node '{node}' missing from ASCII output"

    @pytest.mark.skip(reason="grandalf not installed")
    def test_ascii_output_is_non_empty(self, stub_graph):
        """ASCII output is a non-empty string."""
        ascii_out = stub_graph.get_graph().draw_ascii()
        assert len(ascii_out) > 50


class TestCmdGraph:
    """Tests for the cmd_graph CLI function (ARC-002)."""

    def _make_args(self, fmt: str):
        import argparse
        return argparse.Namespace(format=fmt)

    async def test_cmd_graph_mermaid_format(self):
        """cmd_graph with mermaid format outputs Mermaid diagram."""
        from sacv.cli import cmd_graph
        args = self._make_args("mermaid")
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            await cmd_graph(args)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "-->" in output
        for node in EXPECTED_NODES:
            assert node in output, f"Node '{node}' missing from Mermaid output"

    async def test_cmd_graph_json_format(self):
        """cmd_graph with json format outputs valid JSON with nodes and edges."""
        from sacv.cli import cmd_graph
        args = self._make_args("json")
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            await cmd_graph(args)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        data = json.loads(output)
        assert "nodes" in data
        assert "edges" in data
        node_names = {n["id"] for n in data["nodes"]}
        for node in EXPECTED_NODES:
            assert node in node_names, f"Node '{node}' missing from JSON nodes"

    async def test_cmd_graph_ascii_format(self):
        """cmd_graph with ascii format outputs ASCII diagram or helpful error."""
        from sacv.cli import cmd_graph
        args = self._make_args("ascii")
        import io, sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        exit_code = None
        try:
            await cmd_graph(args)
        except SystemExit as e:
            exit_code = e.code
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        stdout_output = stdout_buf.getvalue()
        stderr_output = stderr_buf.getvalue()
        # Either ASCII output was produced, or grandalf is missing
        if exit_code == 1:
            assert "grandalf" in stderr_output.lower()
        else:
            for node in EXPECTED_NODES:
                assert node in stdout_output, f"Node '{node}' missing from ASCII output"


class TestGraphJsonExport:

    def test_json_output_has_nodes_and_edges(self, stub_graph):
        """JSON output contains 'nodes' and 'edges' keys."""
        data = json.loads(json.dumps(stub_graph.get_graph().to_json()))
        assert "nodes" in data
        assert "edges" in data

    def test_json_nodes_match_expected_count(self, stub_graph):
        """JSON nodes list has one entry per workflow node."""
        data = json.loads(json.dumps(stub_graph.get_graph().to_json()))
        node_names = {n["id"] for n in data["nodes"]}
        for node in EXPECTED_NODES:
            assert node in node_names, f"Node '{node}' missing from JSON nodes"

    def test_json_edges_are_valid(self, stub_graph):
        """Every edge references valid source and target nodes."""
        data = json.loads(json.dumps(stub_graph.get_graph().to_json()))
        node_names = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert edge["source"] in node_names
            assert edge["target"] in node_names
