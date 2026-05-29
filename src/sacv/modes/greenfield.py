from sacv.modes.base import ModeConfig
from dataclasses import dataclass

@dataclass
class GreenfieldConfig(ModeConfig):
    def check_overrides(self) -> dict:
        return {"enforce_ddd": True, "enforce_solid": True, "allow_legacy_patterns": False}
    def critic_weights(self) -> dict:
        return {"style": 1.2, "consistency": 0.8, "security": 1.0}
