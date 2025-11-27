import subprocess
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DockerWrapper:
    """Base wrapper for executing commands in Docker containers."""

    def __init__(self, image: str, host_data_dir: Path, container_data_dir: str = "/data"):
        """Initialize the Docker wrapper.

        Args:
            image: Docker image name.
            host_data_dir: Path to the data directory on the host.
            container_data_dir: Path to the data directory inside the container.
        """
        self.image = image
        self.host_data_dir = host_data_dir.resolve()
        self.container_data_dir = container_data_dir

    def run(self, command: List[str], gpu: bool = False, env: Dict[str, str] = None) -> subprocess.CompletedProcess:
        """Run a command inside the Docker container.

        Args:
            command: List of command arguments (e.g., ["python", "main.py", ...]).
            gpu: Whether to enable GPU support (--gpus all).
            env: Dictionary of environment variables to set in the container.

        Returns:
            subprocess.CompletedProcess: The result of the execution.
        """
        docker_cmd = ["docker", "run", "--rm"]

        # Volume mount
        docker_cmd.extend(["-v", f"{self.host_data_dir}:{self.container_data_dir}"])

        # GPU support
        if gpu:
            docker_cmd.extend(["--gpus", "all"])

        # Environment variables
        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])

        # Image and Command
        docker_cmd.append(self.image)
        docker_cmd.extend(command)

        logger.info(f"Executing Docker command: {' '.join(docker_cmd)}")

        # We run the command.
        # Note: We do NOT pass env=env here because that sets env for the docker client, not the container.
        return subprocess.run(docker_cmd, check=True, text=True)
