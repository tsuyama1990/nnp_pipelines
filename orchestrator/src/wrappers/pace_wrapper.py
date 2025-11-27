from pathlib import Path
from typing import Optional
from .base_wrapper import DockerWrapper

class PaceWorker(DockerWrapper):
    """Wrapper for the Pacemaker Worker."""

    def __init__(self, host_data_dir: Path, image: str = "pace_worker:latest"):
        super().__init__(image, host_data_dir)

    def train(self, config_filename: str, meta_config_filename: str, dataset_filename: str,
              initial_potential: Optional[str] = None, potential_yaml: Optional[str] = None,
              asi: Optional[str] = None, iteration: int = 0) -> str:
        """Run Pacemaker training."""
        cmd = [
            "python", "/app/src/main.py", "train",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--dataset", f"{self.container_data_dir}/{dataset_filename}",
            "--iteration", str(iteration)
        ]

        if initial_potential:
            cmd.extend(["--initial-potential", f"{self.container_data_dir}/{initial_potential}"])

        if potential_yaml:
            cmd.extend(["--potential-yaml", f"{self.container_data_dir}/{potential_yaml}"])

        if asi:
            cmd.extend(["--asi", f"{self.container_data_dir}/{asi}"])

        self.run(cmd, gpu=True)
        return "output_potential.yace"

    def sample(self, config_filename: str, meta_config_filename: str,
               candidates_filename: str, n_samples: int, output_filename: str) -> None:
        """Run Active Learning Sampling."""
        cmd = [
            "python", "/app/src/main.py", "sample",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--candidates", f"{self.container_data_dir}/{candidates_filename}",
            "--n_samples", str(n_samples),
            "--output", f"{self.container_data_dir}/{output_filename}"
        ]

        self.run(cmd, gpu=True)

    def direct_sample(self, input_filename: str, output_filename: str, n_clusters: int) -> None:
        """Run Direct Sampling (ACE+BIRCH)."""
        cmd = [
            "python", "/app/src/main.py", "direct_sample",
            "--input", f"{self.container_data_dir}/{input_filename}",
            "--output", f"{self.container_data_dir}/{output_filename}",
            "--n_clusters", str(n_clusters)
        ]
        self.run(cmd, gpu=True)

    def validate(self, potential_filename: str, output_filename: str) -> None:
        """Run Validation."""
        cmd = [
            "python", "/app/src/main.py", "validate",
            "--potential", f"{self.container_data_dir}/{potential_filename}",
            "--output", f"{self.container_data_dir}/{output_filename}"
        ]
        self.run(cmd, gpu=True)
