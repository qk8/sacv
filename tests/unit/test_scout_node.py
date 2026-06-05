"""
tests/unit/test_scout_node.py
================================
Unit tests for the Scout node.

Tests cover:
1. Context skeleton built from code graph + cross-domain
2. Blast radius only in brownfield mode
3. AGENTS.md loaded when present
4. AGENTS.md truncated at max chars
5. File hints extracted from task description
6. Phase advances to VALUE_NODE
7. Empty file hints handled gracefully
"""
from __future__ import annotations

import pytest
from pathlib import Path

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, ProjectMode
from sacv.nodes.scout import make_scout_node, _FILE_PATTERN
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)
from sacv.interfaces.code_graph_provider import CallGraph, BlastRadiusMap


def _deps(
    code_graph=None, cross_domain=None, **kw
):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=code_graph or StubCodeGraphProvider(),
        cross_domain=cross_domain or StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(**kw),
    )


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-sc-001",
        "task_description": "Add UserService.findById method",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.SCOUT.value,
        "context_skeleton": None, "blast_radius_map": None,
        "agents_md_context": None, "strategy_candidates": [],
        "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
        "check_profile": "standard", "cumulative_cost_dollars": 0.0,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.unit
class TestScoutNode:

    async def test_context_skeleton_built(self, tmp_path, monkeypatch):
        """Scout builds context_skeleton from code graph + cross-domain."""
        monkeypatch.chdir(tmp_path)
        call_graph = StubCodeGraphProvider(
            graph=CallGraph(".", ["node1", "node2"], ["edge1"]),
        )
        cross_domain = StubCrossDomainProvider()
        deps = _deps(code_graph=call_graph, cross_domain=cross_domain)
        node = make_scout_node(deps)

        out = await node(_state())

        assert out["current_phase"] == WorkflowPhase.VALUE_NODE.value
        skeleton = out["context_skeleton"]
        assert "call_graph" in skeleton
        assert "dependencies" in skeleton
        assert "schema_map" in skeleton
        assert "arch_align" in skeleton
        # Call graph should be truncated to limits
        assert len(skeleton["call_graph"]["nodes"]) <= 30
        assert len(skeleton["call_graph"]["edges"]) <= 50

    async def test_blast_radius_only_in_brownfield(self, tmp_path, monkeypatch):
        """Blast radius is computed only in brownfield mode."""
        monkeypatch.chdir(tmp_path)
        call_graph = StubCodeGraphProvider()
        blast = BlastRadiusMap(
            entry_files=["User.java"], affected_files=["User.java", "Repo.java"],
            dependency_depth=2, cross_service_impact=[],
            schema_impact=["users_table"], risk_score=0.6,
        )
        code_graph = StubCodeGraphProvider(blast=blast)
        deps = _deps(code_graph=code_graph)
        node = make_scout_node(deps)

        # Greenfield — no blast radius
        out_gf = await node(_state(
            project_mode=ProjectMode.GREENFIELD.value,
            task_description="Add UserService.findById method to UserService.java",
        ))
        assert out_gf["blast_radius_map"] is None

        # Brownfield — blast radius computed
        out_bf = await node(_state(
            project_mode=ProjectMode.BROWNFIELD.value,
            task_description="Add UserService.findById method to UserService.java",
        ))
        assert out_bf["blast_radius_map"] is not None
        assert out_bf["blast_radius_map"]["risk_score"] == 0.6

    async def test_agents_md_loaded_when_present(self, tmp_path, monkeypatch):
        """When AGENTS.md exists, it is loaded as agents_md_context."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# AGENTS.md\n\n## Common Mistakes\nTest content.")
        deps = _deps()
        node = make_scout_node(deps)

        out = await node(_state())

        assert out["agents_md_context"] is not None
        assert "# AGENTS.md" in out["agents_md_context"]
        assert "## Common Mistakes" in out["agents_md_context"]

    async def test_agents_md_not_loaded_when_absent(self, tmp_path, monkeypatch):
        """When AGENTS.md doesn't exist, agents_md_context is None."""
        monkeypatch.chdir(tmp_path)
        deps = _deps()
        node = make_scout_node(deps)

        out = await node(_state())

        assert out["agents_md_context"] is None

    async def test_agents_md_truncated_at_max_chars(self, tmp_path, monkeypatch):
        """AGENTS.md content is truncated at config.agents_md_prompt_chars."""
        monkeypatch.chdir(tmp_path)
        max_chars = 2000  # default from WorkflowConfig
        long_content = "x" * (max_chars + 1000)
        (tmp_path / "AGENTS.md").write_text(long_content)
        deps = _deps(agents_md_prompt_chars=max_chars)
        node = make_scout_node(deps)

        out = await node(_state())

        context = out["agents_md_context"]
        assert context is not None
        assert len(context) <= max_chars + 50  # +truncation message
        assert "[...truncated" in context

    async def test_file_hints_extracted_from_description(self, tmp_path, monkeypatch):
        """File names in task description are extracted as hints."""
        monkeypatch.chdir(tmp_path)
        received_entry_points = []

        class _TrackingCodeGraph(StubCodeGraphProvider):
            async def get_call_graph(self, entry_points):
                received_entry_points.extend(entry_points)
                return await super().get_call_graph(entry_points)

        code_graph = _TrackingCodeGraph()
        deps = _deps(code_graph=code_graph)
        node = make_scout_node(deps)

        state = _state(task_description="Add UserService.findById and UserRepository.findByEmail")
        # Add .java extensions so _FILE_PATTERN can extract them
        state["task_description"] += " in UserService.java and UserRepository.java"
        await node(state)

        # "UserService.java" and "UserRepository.java" should be extracted as file hints
        assert "UserService.java" in received_entry_points
        assert "UserRepository.java" in received_entry_points

    async def test_empty_file_hints_uses_default(self, tmp_path, monkeypatch):
        """When no file hints extracted, uses ['.'] as default."""
        monkeypatch.chdir(tmp_path)
        received_entry_points = []

        class _TrackingCodeGraph(StubCodeGraphProvider):
            async def get_call_graph(self, entry_points):
                received_entry_points.extend(entry_points)
                return await super().get_call_graph(entry_points)

        code_graph = _TrackingCodeGraph()
        deps = _deps(code_graph=code_graph)
        node = make_scout_node(deps)

        state = _state(task_description="Add a new feature with no file names")
        await node(state)

        assert "." in received_entry_points

    async def test_phase_advances_to_value_node(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deps = _deps()
        node = make_scout_node(deps)
        out = await node(_state())
        assert out["current_phase"] == WorkflowPhase.VALUE_NODE.value

    async def test_cross_domain_called_with_entity_names(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        received_entities = []

        class _TrackingCrossDomain(StubCrossDomainProvider):
            async def map_code_to_schema(self, entity_names):
                received_entities.extend(entity_names)
                return await super().map_code_to_schema(entity_names)

        deps = _deps(cross_domain=_TrackingCrossDomain())
        node = make_scout_node(deps)

        state = _state(task_description="Add UserService.findById and UserRepository in UserService.java UserRepository.java")
        await node(state)

        assert "UserService" in received_entities
        assert "UserRepository" in received_entities


@pytest.mark.unit
class TestFilePattern:

    def test_matches_java_files(self):
        assert "UserService.java" in _FILE_PATTERN.findall("UserService.java")

    def test_matches_typescript_files(self):
        assert "Login.tsx" in _FILE_PATTERN.findall("Login.tsx")

    def test_matches_sql_files(self):
        assert "schema.sql" in _FILE_PATTERN.findall("schema.sql")

    def test_matches_yaml_files(self):
        assert "docker-compose.yaml" in _FILE_PATTERN.findall("docker-compose.yaml")
        assert "config.yml" in _FILE_PATTERN.findall("config.yml")

    def test_matches_json_files(self):
        assert "package.json" in _FILE_PATTERN.findall("package.json")

    def test_matches_xml_files(self):
        assert "pom.xml" in _FILE_PATTERN.findall("pom.xml")

    def test_does_not_match_md_files(self):
        result = _FILE_PATTERN.findall("README.md")
        assert result == []

    def test_does_not_match_txt_files(self):
        result = _FILE_PATTERN.findall("notes.txt")
        assert result == []

    def test_returns_unique_files(self):
        result = list(set(_FILE_PATTERN.findall("UserService.java UserRepository.java UserService.java")))
        assert len(result) == 2  # unique

    def test_handles_paths_with_dashes(self):
        result = _FILE_PATTERN.findall("src/main/java/com/example/User-Service.java")
        # Regex matches full path including directories
        assert any("User-Service.java" in f for f in result)
