# NOT YET WIRED — see ARCH-003.
from sacv.modes.base import ModeConfig
from dataclasses import dataclass
from typing import Any

@dataclass
class BrownfieldConfig(ModeConfig):
    def check_overrides(self) -> dict[str, Any]:
        return {"enforce_ddd": False, "enforce_solid": False, "allow_legacy_patterns": True,
                "require_blast_radius": True, "backward_compat_guard": True}
    def critic_weights(self) -> dict[str, Any]:
        return {"style": 0.7, "consistency": 1.5, "security": 1.0}
