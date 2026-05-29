from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    role: str
    system_prompt: str
    max_turns: int
    allowed_tools: list[str] = field(default_factory=list)
    temperature: float = 0.0

@dataclass
class AgentResult:
    content: str
    tool_calls: list[dict]
    finish_reason: str
    input_tokens: int
    output_tokens: int

class AgentProvider(ABC):
    @abstractmethod
    async def run_task(self, prompt: str, context: dict, config: AgentConfig) -> AgentResult: ...
    @abstractmethod
    async def create_subagent(self, config: AgentConfig) -> "AgentProvider": ...
