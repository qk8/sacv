from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class IterationLimits:
    implement_loop: int = 100
    clarify_round:  int = 5
    spec_audit:     int = 3
    plan_review:    int = 3


@dataclass(frozen=True)
class StagnationConfig:
    total_abort_force:             int   = 3
    drift_revision_limit:          int   = 2
    semantic_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class TokenBudget:
    cost_per_m_input:  float = 5.0
    cost_per_m_output: float = 30.0
    critical_dollar:   float = 80.0
    warning_dollar:    float = 50.0


@dataclass(frozen=True)
class CadenceConfig:
    cleanup_interval:     int = 25
    llm_quality_interval: int = 10
    drift_check_interval: dict[str, int] = field(
        default_factory=lambda: {"simple": 20, "medium": 15, "complex": 10}
    )


@dataclass(frozen=True)
class DebugConfig:
    """
    Configuration for the IntelligentDebugger node.

    user_java_package:  Base package of user code (e.g. 'com.example').
                        Used to filter framework noise from stack traces.
    user_ts_src_root:   Source root for TypeScript files (e.g. 'src').
                        Used to filter node_modules from stack traces.
    jdwp_port:          JDWP debug port exposed by the Docker sandbox.
    cdp_port:           Chrome DevTools Protocol port for Node.js debugging.
    debug_timeout_sec:  Max seconds to wait for a breakpoint hit.
    max_debug_steps:    Max step-over/step-into operations per session.
    actuator_base_url:  Spring Boot Actuator base URL inside the sandbox.
    openapi_spec_path:  Path to generated OpenAPI spec (for cross-stack type check).
    """
    user_java_package:  str  = "com.example"
    user_ts_src_root:   str  = "src"
    jdwp_port:          int  = 5005
    cdp_port:           int  = 9229
    debug_timeout_sec:  int  = 30
    max_debug_steps:    int  = 10
    actuator_base_url:  str  = "http://localhost:8080/actuator"
    openapi_spec_path:  str  = "contracts/openapi/api.yaml"


@dataclass(frozen=True)
class WorkflowConfig:
    # Circuit breaker
    max_self_correction_cycles:      int   = 3
    # Confidence score threshold
    confidence_escalation_threshold: float = 0.25
    # Replan budget
    max_replan_attempts:             int   = 1
    # Resource throttles
    max_parallel_branches:           int   = 2
    max_parallel_critics:            int   = 2
    # Value Node
    min_strategy_score:              float = 0.3
    max_strategies:                  int   = 3
    max_blast_files:                 int   = 50
    # Monorepo mode (approach 1 from architecture session)
    monorepo_mode:                   bool  = False
    # Sub-configs
    iteration_limits: IterationLimits  = field(default_factory=IterationLimits)
    stagnation:       StagnationConfig = field(default_factory=StagnationConfig)
    token_budget:     TokenBudget      = field(default_factory=TokenBudget)
    cadence:          CadenceConfig    = field(default_factory=CadenceConfig)
    debug:            DebugConfig      = field(default_factory=DebugConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkflowConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            max_self_correction_cycles=raw.get("stagnation", {}).get("total_abort_force", 3),
            monorepo_mode=raw.get("monorepo_mode", False),
            debug=DebugConfig(
                user_java_package=raw.get("debug", {}).get("user_java_package", "com.example"),
                user_ts_src_root=raw.get("debug", {}).get("user_ts_src_root", "src"),
            ),
            iteration_limits=IterationLimits(
                implement_loop=raw.get("iteration_limits", {}).get("implement_loop", 100),
            ),
            stagnation=StagnationConfig(
                total_abort_force=raw.get("stagnation", {}).get("total_abort_force", 3),
            ),
            token_budget=TokenBudget(
                cost_per_m_input=raw.get("token_budget", {}).get("cost_per_m_input", 5.0),
                cost_per_m_output=raw.get("token_budget", {}).get("cost_per_m_output", 30.0),
                critical_dollar=raw.get("token_budget", {}).get("critical_dollar", 80.0),
                warning_dollar=raw.get("token_budget", {}).get("warning_dollar", 50.0),
            ),
        )
