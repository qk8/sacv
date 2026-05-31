"""
docker/container_manager.py
============================
Manages the warm Docker sandbox container.

Design:
- On first use, ``warm_container()`` starts a long-lived container
  from the ``sacv-sandbox`` image and returns its handle.
- All subsequent verification calls use ``docker exec`` on this handle,
  eliminating container spin-up latency on every invocation.
- ``destroy_container()`` stops and removes the container cleanly.
- The container is bind-mounted to the host workspace (read-write)
  so the Actor's applied diffs are visible inside the sandbox immediately.

The concrete ``SandboxProvider`` implementation used by ``NodeDeps``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from sacv.interfaces.sandbox_provider import SandboxProvider, SandboxHandle, ExecResult

log = structlog.get_logger(__name__)

_SANDBOX_IMAGE      = "sacv-sandbox:latest"
_CONTAINER_PREFIX   = "sacv-sandbox"
_DEFAULT_WORKDIR    = "/workspace"
_EXEC_TIMEOUT_SEC   = 300


class DockerContainerManager(SandboxProvider):
    """
    Uses the Docker CLI (``docker`` command on PATH) to manage the sandbox.
    Async wrappers around subprocess calls so node code stays non-blocking.
    """

    def __init__(
        self,
        image:      str       = _SANDBOX_IMAGE,
        host_mount: str | Path = ".",
        network:    str       = "bridge",  # changed from "none" — needed for OTel/Jaeger
        jdwp_port:  int       = 5005,
        cdp_port:   int       = 9229,
    ) -> None:
        self._image      = image
        self._host_mount = str(Path(host_mount).resolve())
        self._network    = network
        self._jdwp_port  = jdwp_port
        self._cdp_port   = cdp_port
        self._handle:    SandboxHandle | None = None

    async def warm_container(self) -> SandboxHandle:
        """
        Always creates a new isolated container.
        Callers are responsible for calling destroy_container() when done.
        The singleton self._handle is kept only for optional long-lived
        background container reuse (not used in the main workflow).
        """
        container_id = await self._start_container()
        # Give sandbox-start.sh time to start background services (Jaeger etc.)
        await asyncio.sleep(2)
        handle = SandboxHandle(
            container_id=container_id,
            working_dir=_DEFAULT_WORKDIR,
            warm=True,
        )
        log.info("docker.warm_started", id=container_id[:12])
        return handle

    async def exec_in_container(
        self,
        handle:  SandboxHandle,
        command: str,
        env:     dict[str, str] | None = None,
        timeout: int = _EXEC_TIMEOUT_SEC,
    ) -> ExecResult:
        """
        Execute a shell command inside the warm container via ``docker exec``.
        """
        env_flags: list[str] = []
        for k, v in (env or {}).items():
            env_flags += ["-e", f"{k}={v}"]

        cmd = [
            "docker", "exec",
            *env_flags,
            "-w", handle.working_dir,
            handle.container_id,
            "sh", "-c", command,
        ]

        import time
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            log.error("docker.exec_timeout", command=command[:80], timeout=timeout)
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr=f"Timed out after {timeout}s",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

        duration = int((time.monotonic() - t0) * 1000)
        result   = ExecResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            duration_ms=duration,
        )
        log.debug(
            "docker.exec",
            command=command[:60],
            exit_code=result.exit_code,
            duration_ms=duration,
        )
        return result

    async def destroy_container(self, handle: SandboxHandle) -> None:
        await _run_docker(["docker", "stop",   handle.container_id])
        await _run_docker(["docker", "rm", "-f", handle.container_id])
        log.info("docker.destroyed", id=handle.container_id[:12])

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _start_container(self) -> str:
        import uuid
        name = f"{_CONTAINER_PREFIX}-{uuid.uuid4().hex[:8]}"

        cmd = [
            "docker", "run",
            "--name",    name,
            "--detach",
            "--network", self._network,
            "--mount",   f"type=bind,source={self._host_mount},target={_DEFAULT_WORKDIR}",
            "--memory",  "2g",
            "--cpus",    "2",
            # Publish all debug + service ports so host-side clients can connect
            "-p", f"{self._jdwp_port}:{self._jdwp_port}",
            "-p", f"{self._cdp_port}:{self._cdp_port}",
            "-p", "8080:8080",
            "-p", "16686:16686",
            "-p", "4317:4317",
            "-p", "4318:4318",
            self._image,
            # Do NOT override CMD — let sandbox-start.sh run (starts Jaeger etc.)
        ]
        proc_result = await _run_docker(cmd)
        return proc_result.strip()   # docker outputs the container ID

    async def _container_alive(self, container_id: str) -> bool:
        try:
            out = await _run_docker(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id]
            )
            return out.strip() == "true"
        except Exception:
            return False


async def _run_docker(cmd: list[str]) -> str:
    """Run a docker command and return stdout. Raises RuntimeError on failure."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker command failed: {' '.join(cmd[:4])}\n"
            f"{stderr.decode(errors='replace')[:300]}"
        )
    return stdout.decode(errors="replace").strip()
