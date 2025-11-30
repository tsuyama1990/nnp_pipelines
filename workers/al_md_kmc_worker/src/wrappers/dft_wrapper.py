from pathlib import Path
from .base_wrapper import DockerWrapper

class DftWorker(DockerWrapper):
    """Wrapper for the DFT Worker (Quantum Espresso)."""

    def __init__(self, host_data_dir: Path, image: str = "dft_worker:latest"):
        super().__init__(image, host_data_dir)

    def label(self, config_filename: str, meta_config_filename: str,
              structure_filename: str, output_filename: str) -> None:
        """Run DFT labeling."""
        cmd = [
            "python", "/app/src/main.py",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--structure", f"{self.container_data_dir}/{structure_filename}",
            "--output", f"{self.container_data_dir}/{output_filename}"
        ]

        self.run(cmd, gpu=False) # QE typically CPU/MPI
