from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int

@dataclass
class SandboxHandle:
    container_id: str
    working_dir: str
    warm: bool

class SandboxProvider(ABC):
    @abstractmethod
    async def warm_container(self) -> SandboxHandle: ...
    @abstractmethod
    async def exec_in_container(self, handle: SandboxHandle, command: str,
                                 env: dict[str, str] | None = None, timeout: int = 120) -> ExecResult: ...
    @abstractmethod
    async def destroy_container(self, handle: SandboxHandle) -> None: ...
    @abstractmethod
    def get_host_jdwp_port(self) -> int: ...
    @abstractmethod
    def get_host_cdp_port(self) -> int: ...
