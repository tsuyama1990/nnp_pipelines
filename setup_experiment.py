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

# Add the parent directory of orchestrator to path so we can import modules
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from orchestrator.src.setup.experiment_setup import ExperimentSetup

def main():
    parser = argparse.ArgumentParser(description="Initialize a new experiment.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"),
                        help="Path to the monolithic config file.")
    args = parser.parse_args()

    setup = ExperimentSetup(config_path=args.config)
    setup.run()

if __name__ == "__main__":
    main()
