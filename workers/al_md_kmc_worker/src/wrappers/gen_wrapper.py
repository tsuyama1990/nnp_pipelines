from pathlib import Path
from typing import List, Optional
from .base_wrapper import DockerWrapper

class GenWorker(DockerWrapper):
    """Wrapper for the Generation Worker (MACE)."""

    def __init__(self, host_data_dir: Path, image: str = "gen_worker:latest"):
        super().__init__(image, host_data_dir)

    def generate(self, config_filename: str, output_filename: str) -> None:
        """Run structure generation.

        Args:
            config_filename: Name of the config file (must be in host_data_dir).
            output_filename: Name of the output file (will be created in host_data_dir).
        """
        # Paths inside container
        container_config = f"{self.container_data_dir}/{config_filename}"
        container_output = f"{self.container_data_dir}/{output_filename}"

        cmd = [
            "python", "/app/src/main.py", "generate",
            "--config", container_config,
            "--output", container_output
        ]

        self.run(cmd, gpu=True) # MACE uses GPU

    def filter(self, input_filename: str, output_filename: str, model: str = "medium", fmax: float = 100.0) -> None:
        """Run MACE filtering.

        Args:
            input_filename: Input XYZ.
            output_filename: Output XYZ.
            model: Model size.
            fmax: Force cutoff.
        """
        container_input = f"{self.container_data_dir}/{input_filename}"
        container_output = f"{self.container_data_dir}/{output_filename}"

        cmd = [
            "python", "/app/src/main.py", "filter",
            "--input", container_input,
            "--output", container_output,
            "--model", model,
            "--fmax", str(fmax)
        ]

        self.run(cmd, gpu=True)
