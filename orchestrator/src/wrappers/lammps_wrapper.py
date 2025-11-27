from pathlib import Path
from .base_wrapper import DockerWrapper

class LammpsWorker(DockerWrapper):
    """Wrapper for the LAMMPS Worker."""

    def __init__(self, host_data_dir: Path, image: str = "lammps_worker:latest"):
        super().__init__(image, host_data_dir)

    def run_md(self, config_filename: str, meta_config_filename: str, potential_filename: str,
               structure_filename: str, steps: int, gamma: float, restart: bool = False) -> None:
        """Run MD simulation."""
        cmd = [
            "python", "/app/src/main.py", "md",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--potential", f"{self.container_data_dir}/{potential_filename}",
            "--structure", f"{self.container_data_dir}/{structure_filename}",
            "--steps", str(steps),
            "--gamma", str(gamma)
        ]

        if restart:
            cmd.append("--restart")

        self.run(cmd, gpu=False)

    def generate_cell(self, config_filename: str, meta_config_filename: str,
                      structure_filename: str, center: int, potential_filename: str,
                      output_filename: str) -> None:
        """Run Small Cell Generation (Relaxation)."""
        cmd = [
            "python", "/app/src/main.py", "small_cell",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--structure", f"{self.container_data_dir}/{structure_filename}",
            "--center", str(center),
            "--potential", f"{self.container_data_dir}/{potential_filename}",
            "--output", f"{self.container_data_dir}/{output_filename}"
        ]

        self.run(cmd, gpu=False)

    def run_kmc(self, config_filename: str, meta_config_filename: str,
                structure_filename: str, potential_filename: str, output_filename: str) -> None:
        """Run KMC Step."""
        cmd = [
            "python", "/app/src/main.py", "kmc",
            "--config", f"{self.container_data_dir}/{config_filename}",
            "--meta-config", f"{self.container_data_dir}/{meta_config_filename}",
            "--structure", f"{self.container_data_dir}/{structure_filename}",
            "--potential", f"{self.container_data_dir}/{potential_filename}",
            "--output", f"{self.container_data_dir}/{output_filename}"
        ]

        self.run(cmd, gpu=False)
