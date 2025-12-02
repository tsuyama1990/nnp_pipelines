"""Logger module for recording training progress."""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def setup_logging(name: str = "ace_active_carver") -> logging.Logger:
    """Sets up the global logging configuration.

    Reads LOG_LEVEL from environment variables (default: INFO).
    Sets external noisy libraries to WARNING.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True # Ensure we override previous configs
    )

    # Silence noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    return logging.getLogger(name)


class CSVLogger:
    """Logger for recording active learning metrics to a CSV file."""

    def __init__(self, filepath: str = "training_log.csv"):
        """Initialize the CSV logger.

        Args:
            filepath: Path to the CSV log file.
        """
        self.filepath = Path(filepath)
        self._initialize_log()

    def _initialize_log(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.filepath.exists():
            try:
                with self.filepath.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "iteration",
                        "max_gamma",
                        "n_added_structures",
                        "active_set_size",
                        "rmse_energy",
                        "rmse_forces",
                        "timestamp"
                    ])
            except (OSError, IOError) as e:
                logger.error(f"Failed to initialize log file {self.filepath}: {e}")

    def log_metrics(
        self,
        iteration: int,
        max_gamma: float,
        n_added: int,
        active_set_size: int,
        rmse_energy: Optional[float] = None,
        rmse_forces: Optional[float] = None
    ):
        """Log metrics for the current iteration.

        Args:
            iteration: Current iteration number.
            max_gamma: Maximum extrapolation grade encountered.
            n_added: Number of structures added to the dataset.
            active_set_size: Current size of the active set.
            rmse_energy: RMSE of energy (on test/validation set if available).
            rmse_forces: RMSE of forces (on test/validation set if available).
        """
        import datetime
        timestamp = datetime.datetime.now().isoformat()

        try:
            with self.filepath.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration,
                    max_gamma,
                    n_added,
                    active_set_size,
                    rmse_energy if rmse_energy is not None else "",
                    rmse_forces if rmse_forces is not None else "",
                    timestamp
                ])
        except (OSError, IOError) as e:
            logger.error(f"Failed to write to log file {self.filepath}: {e}")
