from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModeConfig(ABC):
    @abstractmethod
    def check_overrides(self) -> dict: ...
    @abstractmethod
    def critic_weights(self) -> dict: ...
