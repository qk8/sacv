"""
sacv/cli.py
===========
Minimal CLI for the SACV workflow.

Usage:
    python -m sacv.cli run --task-id T1 --description "Add findById" \\
                           --mode brownfield --module backend-domain

    python -m sacv.cli resume --escalation-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import structlog

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps

log = structlog.get_logger(__name__)


def _build_deps() -> "NodeDeps":
    """Build NodeDeps with production adapters. Adjust paths as needed."""
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.config import WorkflowConfig
    from sacv.adapters.claude.claude_agent_adapter import ClaudeAgentAdapter
    from sacv.adapters.memory.agentmemory_adapter import AgentMemoryAdapter
    from sacv.adapters.graph.codegraph_adapter import CodeGraphAdapter
    from sacv.adapters.graph.graphify_adapter import GraphifyAdapter
    from sacv.adapters.sandbox import DockerContainerManager
    from sacv.git.branch_manager import BranchManager
    from sacv.git.diff_engine import DiffEngine

    return NodeDeps(
        agent=ClaudeAgentAdapter(),
        memory=AgentMemoryAdapter(),
        code_graph=CodeGraphAdapter(),
        cross_domain=GraphifyAdapter(),
        git=BranchManager(),
        sandbox=DockerContainerManager(),
        diff=DiffEngine(),
        config=WorkflowConfig(),
    )


async def _start_deps(deps: "NodeDeps") -> None:
    """Start all MCP subprocess adapters (agentmemory, codegraph, graphify)."""
    from sacv.adapters.memory.agentmemory_adapter import AgentMemoryAdapter
    from sacv.adapters.graph.codegraph_adapter import CodeGraphAdapter
    from sacv.adapters.graph.graphify_adapter import GraphifyAdapter

    if isinstance(deps.memory, AgentMemoryAdapter):
        await deps.memory.start()
    if isinstance(deps.code_graph, CodeGraphAdapter):
        await deps.code_graph.start()
    if isinstance(deps.cross_domain, GraphifyAdapter):
        await deps.cross_domain.start()


async def _stop_deps(deps: "NodeDeps") -> None:
    """Stop all MCP subprocess adapters gracefully."""
    from sacv.adapters.memory.agentmemory_adapter import AgentMemoryAdapter
    from sacv.adapters.graph.codegraph_adapter import CodeGraphAdapter
    from sacv.adapters.graph.graphify_adapter import GraphifyAdapter

    if isinstance(deps.memory, AgentMemoryAdapter):
        await deps.memory.stop()
    if isinstance(deps.code_graph, CodeGraphAdapter):
        await deps.code_graph.stop()
    if isinstance(deps.cross_domain, GraphifyAdapter):
        await deps.cross_domain.stop()


async def cmd_run(args: argparse.Namespace) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    from sacv.orchestration.graph import build_graph
    from sacv.orchestration.state import WorkflowPhase
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from sacv.adapters.sandbox import DockerContainerManager

    deps = _build_deps()
    await _start_deps(deps)
    try:
        # Validate Docker image exists before starting the graph
        await DockerContainerManager.validate_image()

        # ── Validate MCP server connectivity ──────────────────────────────
        from sacv.adapters.memory.agentmemory_adapter import AgentMemoryAdapter
        from sacv.adapters.graph.codegraph_adapter import CodeGraphAdapter
        from sacv.adapters.graph.graphify_adapter import GraphifyAdapter

        for adapter, name in [
            (deps.memory, "agentmemory"),
            (deps.code_graph, "codegraph"),
            (deps.cross_domain, "graphify"),
        ]:
            if hasattr(adapter, "validate"):
                try:
                    await adapter.validate()
                except RuntimeError as exc:
                    print(f"ERROR: {name} MCP server is not available: {exc}",
                          file=sys.stderr)
                    sys.exit(1)

        log.info("workflow.all_deps_validated")

        # ── Dump effective configuration at DEBUG level ───────────────────
        import dataclasses
        log.debug(
            "workflow.config",
            config=dataclasses.asdict(deps.config),
            task_id=args.task_id,
            mode=args.mode,
            module=args.module,
            check_profile=args.check_profile,
        )

        db_path = Path(".workflow/sacv.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        initial_state = {
            "task_id":          args.task_id,
            "task_description": args.description,
            "project_mode":     args.mode,
            "module_type":      args.module,
            "session_id":       "",
            "session_start_ms": None,   # set by bootstrap
            "current_phase":    WorkflowPhase.BOOTSTRAP.value,
            "check_profile":    args.check_profile,
            # All remaining fields initialised to None/[] by bootstrap
            "context_skeleton":       None, "blast_radius_map": None,
            "agents_md_context":      None, "strategy_candidates": [],
            "selected_strategy":      None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "tdd_gate_attempts":      0, "skip_tdd_gate": False,
            "diff_proposal":          None,
            "preflight_result":       None, "critic_findings": [],
            "verifier_verdict":       None, "debug_observations": None,
            "correction_state": {
                "attempt_count": 0, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score":        1.0, "replan_count": 0,
            "active_branches":         [], "exhausted_branches": [],
            "speculative_stash_ref":   None, "escalation_payload": None,
            "procedural_constraints":  [], "lesson_learned": None,
            "arch_rules_updated":      False,
            "cumulative_cost_dollars": 0.0,    # BUG-008: token budget tracking
        }

        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(deps, checkpointer=checkpointer)
            config = {"configurable": {"thread_id": args.task_id}}
            try:
                result = await graph.ainvoke(initial_state, config=config)
            except Exception:
                try:
                    state_snapshot = await graph.get_state(config)
                    if state_snapshot and state_snapshot.values:
                        sv = state_snapshot.values
                        log.error(
                            "workflow.fatal_exception",
                            task_id=args.task_id,
                            last_phase=sv.get("current_phase", "unknown"),
                            attempt=sv.get("correction_state", {}).get("attempt_count"),
                            last_verdict=(sv.get("verifier_verdict") or {}).get("test_result")
                            if sv.get("verifier_verdict") else None,
                            cost=sv.get("cumulative_cost_dollars"),
                            exc_info=True,
                        )
                except Exception:
                    log.error("workflow.fatal_exception_no_state", exc_info=True)
                raise

        print(json.dumps({
            "phase": result.get("current_phase"),
            "task": args.task_id,
        }))
    finally:
        await _stop_deps(deps)  # always runs, even on Ctrl+C or exception


async def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a graph that was paused at HITL escalation."""
    from sacv.orchestration.graph import build_graph
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    esc_path = Path(f".workflow/escalations/{args.escalation_id}.json")
    if not esc_path.exists():
        print(f"ERROR: escalation file not found: {esc_path}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(esc_path.read_text())
    task_id = payload["task_id"]

    deps = _build_deps()
    await _start_deps(deps)
    try:
        # Use AsyncSqliteSaver for persistence (must match the run checkpointer)
        db_path = Path(".workflow/sacv.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(deps, checkpointer=checkpointer)
            # Resume the interrupted graph with the human's decision
            config = {"configurable": {"thread_id": task_id}}
            # Provide None as the resume input (human reviewed; no automated fix)
            from langgraph.types import Command
            result = await graph.ainvoke(Command(resume=None), config=config)

        print(json.dumps({
            "resumed": task_id,
            "phase": result.get("current_phase"),
        }))
    finally:
        await _stop_deps(deps)  # always runs, even on Ctrl+C or exception


def main() -> None:
    parser = argparse.ArgumentParser(prog="sacv")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a new SACV workflow task")
    run_p.add_argument("--task-id", required=True, help="Unique task identifier")
    run_p.add_argument("--description", required=True, help="Task description")
    run_p.add_argument(
        "--mode", choices=["greenfield", "brownfield"], default="greenfield",
    )
    run_p.add_argument(
        "--module",
        choices=[
            "backend-domain", "backend-api",
            "frontend-feature", "frontend-data",
            "infrastructure", "cross-cutting",
        ],
        default="backend-domain",
    )
    run_p.add_argument(
        "--check-profile",
        choices=["standard", "full"],
        default="standard",
        help=(
            "Preflight check profile. 'standard' runs LSP + architecture "
            "checks. 'full' additionally runs cross-stack type safety, "
            "performance delta, and visual diff checks."
        ),
    )
    run_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Set log level to DEBUG (default: INFO)",
    )
    run_p.add_argument(
        "--log-format",
        choices=["json", "console"],
        default=None,
        help="Override LOG_FORMAT env var",
    )

    res_p = sub.add_parser("resume", help="Resume a paused HITL escalation")
    res_p.add_argument(
        "--escalation-id", required=True,
        help="Escalation ID from the .workflow/escalations/<id>.json file",
    )
    res_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Set log level to DEBUG (default: INFO)",
    )
    res_p.add_argument(
        "--log-format",
        choices=["json", "console"],
        default=None,
        help="Override LOG_FORMAT env var",
    )

    args = parser.parse_args()
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    if args.log_format:
        os.environ["LOG_FORMAT"] = args.log_format
    from sacv.logging_config import configure_logging
    configure_logging()

    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "resume":
        asyncio.run(cmd_resume(args))


if __name__ == "__main__":
    main()
