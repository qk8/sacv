"""Unit tests for WorkflowConfig.from_json() field propagation."""
import json
import tempfile
import pathlib

from sacv.orchestration.config import WorkflowConfig


def test_from_json_respects_agents_md_prompt_chars(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"agents_md_prompt_chars": 500}))
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.agents_md_prompt_chars == 500


def test_from_json_default_agents_md_prompt_chars(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = WorkflowConfig.from_json(cfg_file)
    assert cfg.agents_md_prompt_chars == 2_000
