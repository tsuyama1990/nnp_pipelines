import subprocess
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DockerWrapper:
    """Base wrapper for executing commands in Docker containers.

    Handles Docker-in-Docker volume mounting by mapping container paths back to host paths
    using HOST_WORK_DIR environment variable.
    """

    def __init__(self, image: str, host_data_dir: Path, container_data_dir: str = "/data"):
        """Initialize the Docker wrapper.

        Args:
            image: Docker image name.
            host_data_dir: Path to the data directory (as seen by THIS process).
            container_data_dir: Path to the data directory inside the sibling container.
        """
        self.image = image
        # Local path (inside al_md_kmc_worker container)
        self.local_data_dir = host_data_dir.resolve()
        self.container_data_dir = container_data_dir

        # Determine Host Path for mounting
        # If running in Docker (DinD), we need the path on the physical host, not the container path.
        self.host_work_dir = os.environ.get("HOST_WORK_DIR")

        if self.host_work_dir:
            # We assume the local_data_dir is a subdirectory of the work dir mounted at /app/work
            # Typically:
            #   Host: /path/to/exp/work
            #   Container (al_md_kmc_worker): /app/work
            #   We want to mount: /path/to/exp/work/iteration_1 -> /data (in sibling)

            # Strategy: Replace /app/work prefix with HOST_WORK_DIR
            # First, normalize paths
            local_str = str(self.local_data_dir)

            # Check where we are mounted.
            # setup_experiment.py maps $(pwd):/app/work.
            # So if local path starts with /app/work, we replace it.
            # But the user might be running locally (dev mode).

            # Common convention:
            # internal_base = "/app/work"
            # But wait, main.py adds /app to path.
            # WORKDIR is likely /app or /app/work.

            # Safe approach: Just assume HOST_WORK_DIR corresponds to current working directory (pwd)
            # or a specific base.

            # Let's assume HOST_WORK_DIR points to the root of the 'work' folder on host.
            # And 'work' folder is mounted to $(pwd) inside container if we follow setup_experiment.py:
            # -v $(pwd):/app/work

            # So if local_data_dir is /app/work/data, and HOST_WORK_DIR is /home/user/exp/work
            # We want to mount /home/user/exp/work/data.

            # We need to find the relative path from the mount point.
            # If we assume CWD is the mount point (often true):
            cwd = Path.cwd()
            try:
                rel_path = self.local_data_dir.relative_to(cwd)
                self.mount_source = str(Path(self.host_work_dir) / rel_path)
                logger.info(f"DinD: Mapped local {self.local_data_dir} to host {self.mount_source}")
            except ValueError:
                # local_data_dir is not relative to CWD.
                # Fallback: Just use local path (maybe running natively)
                logger.warning(f"Could not resolve relative path for {self.local_data_dir} from {cwd}. using raw path.")
                self.mount_source = str(self.local_data_dir)
        else:
            # Running on host directly
            self.mount_source = str(self.local_data_dir)


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

        # Volume mount using calculated host source
        docker_cmd.extend(["-v", f"{self.mount_source}:{self.container_data_dir}"])

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

        return subprocess.run(docker_cmd, check=True, text=True)
