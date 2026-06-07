# NOT YET WIRED — see ARCH-003. Mode-specific critic weights and check
# overrides are defined but not applied to the live workflow.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ModeConfig(ABC):
    @abstractmethod
    def check_overrides(self) -> dict[str, Any]: ...
    @abstractmethod
    def critic_weights(self) -> dict[str, Any]: ...
