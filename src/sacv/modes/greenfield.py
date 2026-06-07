# NOT YET WIRED — see ARCH-003.
from sacv.modes.base import ModeConfig
from dataclasses import dataclass
from typing import Any

@dataclass
class GreenfieldConfig(ModeConfig):
    def check_overrides(self) -> dict[str, Any]:
        return {"enforce_ddd": True, "enforce_solid": True, "allow_legacy_patterns": False}
    def critic_weights(self) -> dict[str, Any]:
        return {"style": 1.2, "consistency": 0.8, "security": 1.0}
