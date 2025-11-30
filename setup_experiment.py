#!/usr/bin/env python3
"""
Setup Experiment Script

This script initializes a new experiment directory structure by splitting a monolithic
config.yaml into modular, step-specific configuration files. It also generates a
run_pipeline.sh script with commented-out Docker commands for each step.

Usage:
    python setup_experiment.py [--config config.yaml]
"""

import argparse
import pathlib
import sys
import yaml

# Add the parent directory of orchestrator to path so we can import modules
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

# Updated import path for Unified Worker
from workers.al_md_kmc_worker.src.setup.experiment_setup import ExperimentSetup

def ensure_meta_config(path: pathlib.Path):
    """
    Ensures that the meta configuration file exists in the project root.
    If it doesn't exist, it creates it with robust default values including
    Docker image tags and execution commands.
    """
    if not path.exists():
        print(f"Meta config not found at {path}. Creating default template with Docker tags.")
        default_meta = {
            "execution": {
                "mpirun_command": "mpirun -np 4"
            },
            "lammps": {
                "command": "lmp_mpi"
            },
            "dft": {
                "command": "pw.x",
                "pseudo_dir": "./pseudos"
            },
            "docker": {
                "pace_image": "pace_worker:latest",
                "lammps_image": "lammps_worker:latest",
                "dft_image": "dft_worker:latest",
                "gen_image": "gen_worker:latest"
            }
        }
        with open(path, 'w') as f:
            yaml.dump(default_meta, f, default_flow_style=False, sort_keys=False)
    else:
        print(f"Meta config found at {path}. Using existing configuration.")

def main():
    parser = argparse.ArgumentParser(description="Initialize a new experiment.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"),
                        help="Path to the monolithic config file.")
    args = parser.parse_args()

    # Step 1: Ensure config_meta.yaml exists in the project root with the correct structure
    root_meta_path = pathlib.Path("config_meta.yaml")
    ensure_meta_config(root_meta_path)

    # Step 2: Run the experiment setup
    # The ExperimentSetup class will look for config_meta.yaml in the root (current working directory)
    # by default if not specified otherwise in its constructor defaults.
    setup = ExperimentSetup(config_path=args.config)

    # setup.run() will internally call generate_step_configs(), which copies the loaded
    # meta configuration to the experiment's configs/ directory, effectively snapshotting it.
    setup.run()

if __name__ == "__main__":
    main()
