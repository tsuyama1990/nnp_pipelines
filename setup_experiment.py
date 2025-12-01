#!/usr/bin/env python3
"""
Setup Experiment Script

This script initializes a new experiment directory structure by splitting a monolithic
config.yaml into modular, step-specific configuration files. It also generates a
run_pipeline.sh script with commented-out Docker commands for each step.

Usage:
    python setup_experiment.py [--config config.yaml] [--name NAME] [--resume PATH]
"""

import argparse
import pathlib
import sys
import yaml
import os
import logging
from typing import Optional

# Add the parent directory of orchestrator to path so we can import modules
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

# Updated import path for Unified Worker
from workers.al_md_kmc_worker.src.setup.experiment_setup import ExperimentSetup
from workers.al_md_kmc_worker.src.main import run_active_learning

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PipelineArgs:
    """Helper class to mimic argparse arguments needed by run_active_learning."""
    def __init__(self, config: str, meta_config: str, resume_iteration: Optional[int] = None):
        self.config = config
        self.meta_config = meta_config
        self.resume_iteration = resume_iteration

def ensure_meta_config(path: pathlib.Path):
    """
    Ensures that the meta configuration file exists in the project root.
    If it doesn't exist, it creates it with robust default values including
    Docker image tags and execution commands.
    """
    if not path.exists():
        logger.warning(f"Meta config not found at {path}. Creating default template with Docker tags. PLEASE REVIEW THIS FILE.")
        default_meta = {
            "execution": {
                "mpirun_command": "mpirun -np 4"
            },
            "lammps": {
                "command": "lmp_mpi"
            },
            "dft": {
                "command": "pw.x",
                "pseudo_dir": "./pseudos",
                "sssp_json_path": "./sssp.json"
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
        logger.info(f"Meta config found at {path}. Using existing configuration.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="NNP Pipeline Entry Point. Initializes and runs the active learning loop.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment configuration file."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Optional experiment name tag for directory naming."
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to an existing experiment directory to resume."
    )
    # Note: --iteration is mentioned in README for resume but currently not handled
    # explicitly by orchestrator resume logic which might auto-detect.
    # Adding it to parser to avoid error if user provides it.
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration number to resume from (optional)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Step 0: Ensure config_meta.yaml exists in the project root
    root_meta_path = pathlib.Path("config_meta.yaml")
    ensure_meta_config(root_meta_path)

    if args.resume:
        # Resume mode
        resume_path = pathlib.Path(args.resume).resolve()
        logger.info(f"Resuming experiment from: {resume_path}")

        # Strict validation of experiment root
        # 1. Check if the path itself is the experiment root (should contain work/ and configs/)
        # 2. Or if it's a subdirectory, warn the user to provide the root.

        # We expect the resume path to be the experiment root (e.g., experiment_name/)
        if not (resume_path / "work" / "07_active_learning").exists() and not (resume_path / "configs").exists():
             logger.error(f"Invalid resume path: {resume_path}")
             logger.error("Please provide the root directory of the experiment (containing 'work/' and 'configs/').")
             sys.exit(1)

        experiment_root = resume_path

        # Validate State File
        # State file is typically in work/07_active_learning/experiment_state.json OR work/experiment_state.json
        # The orchestrator logic uses work/07_active_learning as the root for state.

        # NOTE: State location depends on orchestrator implementation.
        # Based on test_pipeline_e2e.py, it seems we expect it in `work/experiment_state.json` or `work/07_active_learning`.
        # ActiveLearningOrchestrator.run sets work_root = Path("work/07_active_learning").
        # StateManager default is "experiment_state.json" in CWD? No, it takes a path.

        # Let's check the standard location
        state_path = experiment_root / "work" / "07_active_learning" / "experiment_state.json"

        if not state_path.exists():
            # Try alternative location
             state_path_alt = experiment_root / "work" / "experiment_state.json"
             if state_path_alt.exists():
                 state_path = state_path_alt
             else:
                 logger.error(f"Experiment state file not found at {state_path} or {state_path_alt}.")
                 logger.error("Cannot resume without a valid state file.")
                 sys.exit(1)

        logger.info(f"Found experiment state at: {state_path}")
        logger.info(f"Confirmed experiment root: {experiment_root}")

        # Locate configs
        config_path = experiment_root / "configs" / "07_active_learning.yaml"
        meta_config_path = experiment_root / "configs" / "config_meta.yaml"

        if not config_path.exists():
            logger.error(f"Could not find configuration at {config_path}")
            sys.exit(1)

        # Switch to experiment root so relative paths work (e.g. data/)
        os.chdir(experiment_root)

        # Run Pipeline
        pipeline_args = PipelineArgs(
            config=str(config_path.relative_to(experiment_root)),
            meta_config=str(meta_config_path.relative_to(experiment_root)),
            resume_iteration=args.iteration
        )
        run_active_learning(pipeline_args)

    else:
        # New Experiment Setup
        logger.info("Initializing new experiment...")

        config_path = pathlib.Path(args.config)
        setup = ExperimentSetup(config_path=config_path)

        # Load configurations first to modify name if needed
        setup.load_configurations()

        if args.name:
            logger.info(f"Overriding experiment name to: {args.name}")
            setup.exp_name = args.name
            if "experiment" not in setup.root_config:
                setup.root_config["experiment"] = {}
            setup.root_config["experiment"]["name"] = args.name

        setup.create_directory_structure()
        setup.generate_step_configs()
        setup.generate_pipeline_script()

        logger.info("Setup complete. Starting orchestrator...")

        # Determine paths for execution
        experiment_root = setup.exp_dir.resolve()
        config_path = experiment_root / "configs" / "07_active_learning.yaml"
        meta_config_path = experiment_root / "configs" / "config_meta.yaml"

        # Switch to experiment root
        os.chdir(experiment_root)

        # Run Pipeline
        pipeline_args = PipelineArgs(
            config="configs/07_active_learning.yaml",
            meta_config="configs/config_meta.yaml"
        )
        run_active_learning(pipeline_args)

if __name__ == "__main__":
    main()
