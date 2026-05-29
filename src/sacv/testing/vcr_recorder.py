"""VCR cassette recorder/replayer for AgentProvider."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Literal
from sacv.interfaces.agent_provider import AgentProvider, AgentConfig, AgentResult

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "responses"

class VCRAgentProvider(AgentProvider):
    def __init__(self, provider: AgentProvider | None, cassette_name: str,
                 mode: Literal["record", "replay"]) -> None:
        self._provider     = provider
        self._cassette     = FIXTURES_DIR / f"{cassette_name}.json"
        self._mode         = mode
        self._recordings:  list[dict] = []
        self._replay_index = 0
        if mode == "replay":
            if not self._cassette.exists():
                raise FileNotFoundError(f"Cassette not found: {self._cassette}")
            self._recordings = json.loads(self._cassette.read_text())

    async def run_task(self, prompt: str, context: dict, config: AgentConfig) -> AgentResult:
        if self._mode == "replay":
            try:
                entry = self._recordings[self._replay_index]
            except IndexError:
                raise IndexError(f"VCR exhausted at call {self._replay_index}")
            self._replay_index += 1
            return AgentResult(**entry["result"])
        assert self._provider is not None
        result = await self._provider.run_task(prompt, context, config)
        self._recordings.append({
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12],
            "result": {"content": result.content, "tool_calls": result.tool_calls,
                       "finish_reason": result.finish_reason,
                       "input_tokens": result.input_tokens,
                       "output_tokens": result.output_tokens},
        })
        return result

    async def create_subagent(self, config: AgentConfig) -> "VCRAgentProvider":
        return VCRAgentProvider(None, f"{self._cassette.stem}_sub", self._mode)

    def save_cassette(self) -> None:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        self._cassette.write_text(json.dumps(self._recordings, indent=2))
