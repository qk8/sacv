"""
adapters/sandbox/docker_sandbox_adapter.py
==========================================
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
_HEALTH_CHECK_INTERVAL = 0.25   # seconds between polls
_HEALTH_CHECK_TIMEOUT  = 15.0   # give up after this many seconds


class DockerContainerManager(SandboxProvider):
    """
    Uses the Docker CLI (``docker`` command on PATH) to manage the sandbox.
    Async wrappers around subprocess calls so node code stays non-blocking.
    """

    @staticmethod
    async def validate_image(image: str = _SANDBOX_IMAGE) -> None:
        """Raise RuntimeError if the sandbox Docker image is not available locally."""
        try:
            await _run_docker(["docker", "image", "inspect", image])
        except RuntimeError:
            raise RuntimeError(
                f"Docker image '{image}' not found. "
                f"Build it first: docker build -f Dockerfile.sandbox -t {image} ."
            )

    def __init__(
        self,
        image:      str       = _SANDBOX_IMAGE,
        host_mount: str | Path = ".",
        network:    str       = "none",
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
        container_id, host_jdwp, host_cdp = await self._start_container()
        handle = SandboxHandle(
            container_id=container_id,
            working_dir=_DEFAULT_WORKDIR,
            warm=True,
            host_jdwp_port=host_jdwp,
            host_cdp_port=host_cdp,
        )
        await self._wait_for_ready(handle)
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

    def create_isolated_instance(self, host_mount: str) -> "DockerContainerManager":
        """
        Create a new DockerContainerManager with a different host mount
        but the same port configuration. Used by speculative branching
        to create isolated sandbox instances for parallel evaluation.
        """
        return DockerContainerManager(
            image=self._image,
            host_mount=host_mount,
            network=self._network,
            jdwp_port=self._jdwp_port,
            cdp_port=self._cdp_port,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _start_container(self) -> tuple[str, int, int]:
        """
        Start a new container and return (container_id, host_jdwp_port, host_cdp_port).

        The ephemeral host ports are returned directly rather than stored as
        instance-side-effects, making concurrent warm_container calls safe.
        """
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
            # Use 0 (ephemeral) host ports — avoids conflicts when multiple
            # containers run concurrently (e.g. speculative branches).
            # The actual host ports are resolved after start via docker inspect.
            "-p", f"0:{self._jdwp_port}",
            "-p", f"0:{self._cdp_port}",
            "-p", "0:8080",
            "-p", "0:16686",
            "-p", "0:4317",
            "-p", "0:4318",
            self._image,
            # Do NOT override CMD — let sandbox-start.sh run (starts Jaeger etc.)
        ]
        proc_result = await _run_docker(cmd, timeout=300)
        container_id = proc_result.strip()

        # Resolve actual ephemeral host ports from container port bindings
        # Run blocking subprocess.run in a thread pool so the event loop
        # is not blocked (BUG-002 fix).
        host_jdwp, host_cdp = await asyncio.to_thread(
            self._resolve_host_ports, container_id
        )

        return container_id, host_jdwp, host_cdp

    def _resolve_host_ports(self, container_id: str) -> tuple[int, int]:
        """
        Look up the actual host ports assigned by Docker for the JDWP and CDP
        container ports.  Returns (host_jdwp_port, host_cdp_port).

        Falls back to the configured default ports if inspection fails.
        """
        import json
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return self._jdwp_port, self._cdp_port
            inspect_data = json.loads(result.stdout)
            if not inspect_data:
                return self._jdwp_port, self._cdp_port
            host_map = inspect_data[0].get("NetworkSettings", {}).get("Ports", {})
            jdwp_host = self._find_host_port(host_map, f"{self._jdwp_port}/tcp")
            cdp_host  = self._find_host_port(host_map, f"{self._cdp_port}/tcp")
            return jdwp_host or self._jdwp_port, cdp_host or self._cdp_port
        except Exception:
            pass
        # Fallback to configured defaults
        return self._jdwp_port, self._cdp_port

    @staticmethod
    def _find_host_port(host_map: dict, container_port: str) -> int | None:
        """Extract host port from docker inspect port mapping."""
        mappings = host_map.get(container_port)
        if mappings:
            first = mappings[0] if isinstance(mappings[0], dict) else {}
            return int(first.get("HostPort", 0))
        return None

    async def _container_alive(self, container_id: str) -> bool:
        try:
            out = await _run_docker(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id]
            )
            return out.strip() == "true"
        except Exception:
            return False

    async def _wait_for_ready(self, handle: SandboxHandle) -> None:
        """
        Poll the container until the sandbox-start.sh script signals readiness.
        We check for the existence of /tmp/sacv-ready (written by sandbox-start.sh).
        Falls back gracefully after _HEALTH_CHECK_TIMEOUT seconds.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _HEALTH_CHECK_TIMEOUT
        while loop.time() < deadline:
            result = await self.exec_in_container(
                handle,
                "test -f /tmp/sacv-ready && echo OK || echo WAIT",
                timeout=2,
            )
            if result.stdout.strip() == "OK":
                return
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
        log.warning("docker.ready_timeout", timeout=_HEALTH_CHECK_TIMEOUT)
        # Do not raise — container may still be usable; services just started slowly


async def _run_docker(cmd: list[str], timeout: int = 60) -> str:
    """Run a docker command and return stdout. Raises RuntimeError on failure.

    Args:
        cmd: Docker CLI command.
        timeout: Max seconds to wait. Default 60s. Use 300s for container
                 start (first run may need to pull the image).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=float(timeout),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(
            f"docker command timed out after {timeout}s: {' '.join(cmd[:4])}"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker command failed: {' '.join(cmd[:4])}\n"
            f"{stderr.decode(errors='replace')[:300]}"
        )
    return stdout.decode(errors="replace").strip()
