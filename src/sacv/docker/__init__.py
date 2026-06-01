"""docker — backward-compat re-export; prefer sacv.adapters.sandbox."""
from sacv.adapters.sandbox.docker_sandbox_adapter import DockerContainerManager

__all__ = ["DockerContainerManager"]
