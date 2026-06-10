from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "max_self_correction_cycles", "confidence_escalation_threshold",
    "max_replan_attempts", "max_tdd_gate_attempts", "max_empty_diff_retries",
    "max_parallel_branches", "max_parallel_critics", "min_strategy_score",
    "max_strategies", "max_blast_files", "monorepo_mode", "agents_md_prompt_chars",
    "max_workflow_duration", "confidence_weights",
    "stagnation", "token_budget", "cadence", "debug",
})


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
    """
    Reserved for future implementation.

    cleanup_interval:     purge stale LangGraph checkpoints every N nodes
    llm_quality_interval: run a self-evaluation on output quality every N LLM calls
    drift_check_interval: check semantic drift in error history every N iterations
                          (by complexity)
    """
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
    debug_timeout_sec:   int  = 30
    debug_port_wait_sec: float = 30.0
    max_debug_steps:     int  = 10
    actuator_base_url:   str  = "http://localhost:8080/actuator"
    openapi_spec_path:   str  = "contracts/openapi/api.yaml"
    otel_query_url:      str  = "http://localhost:16686/api/traces"


@dataclass(frozen=True)
class ConfidenceWeights:
    """Coefficients for compute_confidence_score penalties.

    Defaults match the original hardcoded values in edges.py.
    """
    stagnation_penalty:        float = 0.40
    blast_penalty_scale:       float = 0.30
    critic_penalty_per_crit:   float = 0.10
    max_critic_penalty:        float = 0.30


@dataclass(frozen=True)
class WorkflowConfig:
    # Circuit breaker
    max_self_correction_cycles:      int   = 3
    # Confidence score threshold
    confidence_escalation_threshold: float = 0.25
    # Replan budget
    max_replan_attempts:             int   = 1
    # TDD gate
    max_tdd_gate_attempts:           int   = 3
    # Actor empty-diff retry limit (separate from correction cycles)
    max_empty_diff_retries:          int   = 3
    # Resource throttles
    max_parallel_branches:           int   = 2
    max_parallel_critics:            int   = 2
    # Value Node
    min_strategy_score:              float = 0.3
    max_strategies:                  int   = 3
    max_blast_files:                 int   = 50
    # Monorepo mode (approach 1 from architecture session)
    monorepo_mode:                   bool  = False
    # Agents.md prompt chars (chars of AGENTS.md to include in Actor prompt)
    agents_md_prompt_chars:          int   = 2_000
    # Workflow-level session timeout (seconds; 0 = no timeout)
    max_workflow_duration:           int   = 3600
    # Sub-configs
    stagnation:         StagnationConfig  = field(default_factory=StagnationConfig)
    token_budget:       TokenBudget       = field(default_factory=TokenBudget)
    cadence:            CadenceConfig     = field(default_factory=CadenceConfig)
    debug:              DebugConfig       = field(default_factory=DebugConfig)
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkflowConfig":
        raw = json.loads(Path(path).read_text())

        # ── Warn on unknown top-level keys ────────────────────────────────
        unknown = set(raw.keys()) - _KNOWN_TOP_LEVEL_KEYS
        if unknown:
            _log.warning(
                "config.unknown_keys",
                path=str(path),
                unknown_keys=sorted(unknown),
                hint="These keys are unrecognised and will have no effect. Check for typos.",
            )

        dbg = raw.get("debug", {})
        cad = raw.get("cadence", {})
        stg = raw.get("stagnation", {})
        tok = raw.get("token_budget", {})
        return cls(
            max_self_correction_cycles=raw.get(
                "max_self_correction_cycles",
                stg.get("total_abort_force", 3),
            ),
            confidence_escalation_threshold=raw.get("confidence_escalation_threshold", 0.25),
            max_replan_attempts=raw.get("max_replan_attempts", 1),
            max_tdd_gate_attempts=raw.get("max_tdd_gate_attempts", 3),
            max_empty_diff_retries=raw.get("max_empty_diff_retries", 3),
            max_parallel_branches=raw.get("max_parallel_branches", 2),
            max_parallel_critics=raw.get("max_parallel_critics", 2),
            min_strategy_score=raw.get("min_strategy_score", 0.3),
            max_strategies=raw.get("max_strategies", 3),
            max_blast_files=raw.get("max_blast_files", 50),
            monorepo_mode=raw.get("monorepo_mode", False),
            agents_md_prompt_chars=raw.get("agents_md_prompt_chars", 2_000),
            max_workflow_duration=raw.get("max_workflow_duration", 3600),
            debug=DebugConfig(
                user_java_package=dbg.get("user_java_package", "com.example"),
                user_ts_src_root=dbg.get("user_ts_src_root", "src"),
                jdwp_port=dbg.get("jdwp_port", 5005),
                cdp_port=dbg.get("cdp_port", 9229),
                debug_timeout_sec=dbg.get("debug_timeout_sec", 30),
                debug_port_wait_sec=dbg.get("debug_port_wait_sec", 30.0),
                max_debug_steps=dbg.get("max_debug_steps", 10),
                actuator_base_url=dbg.get("actuator_base_url", "http://localhost:8080/actuator"),
                openapi_spec_path=dbg.get("openapi_spec_path", "contracts/openapi/api.yaml"),
                otel_query_url=dbg.get("otel_query_url", "http://localhost:16686/api/traces"),
            ),
            stagnation=StagnationConfig(
                total_abort_force=stg.get("total_abort_force", 3),
                drift_revision_limit=stg.get("drift_revision_limit", 2),
                semantic_similarity_threshold=stg.get(
                    "semantic_similarity_threshold", 0.85
                ),
            ),
            token_budget=TokenBudget(
                cost_per_m_input=tok.get("cost_per_m_input", 5.0),
                cost_per_m_output=tok.get("cost_per_m_output", 30.0),
                critical_dollar=tok.get("critical_dollar", 80.0),
                warning_dollar=tok.get("warning_dollar", 50.0),
            ),
            cadence=CadenceConfig(
                cleanup_interval=cad.get("cleanup_interval", 25),
                llm_quality_interval=cad.get("llm_quality_interval", 10),
                drift_check_interval=cad.get(
                    "drift_check_interval", {"simple": 20, "medium": 15, "complex": 10}
                ),
            ),
            confidence_weights=ConfidenceWeights(
                stagnation_penalty=raw.get("confidence_weights", {}).get(
                    "stagnation_penalty", 0.40
                ),
                blast_penalty_scale=raw.get("confidence_weights", {}).get(
                    "blast_penalty_scale", 0.30
                ),
                critic_penalty_per_crit=raw.get("confidence_weights", {}).get(
                    "critic_penalty_per_crit", 0.10
                ),
                max_critic_penalty=raw.get("confidence_weights", {}).get(
                    "max_critic_penalty", 0.30
                ),
            ),
        )
