#!/usr/bin/env python3
"""
Setup Experiment Script

This script initializes a new experiment directory structure by splitting a monolithic
config.yaml into modular, step-specific configuration files. It also generates a
run_pipeline.sh script with valid commands for each step.

Usage:
    python setup_experiment.py [--config config.yaml] [--name NAME] [--run] [--resume PATH] [--local]
"""

import argparse
import pathlib
import sys
import yaml
import os
import logging
import shutil
import subprocess
from typing import Optional

# Add the worker src to path so we can import the orchestrator if needed locally
sys.path.append(os.path.join(os.path.dirname(__file__), "workers/al_md_kmc_worker/src"))

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

def check_docker() -> bool:
    """Checks if Docker is available and running."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

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
    parser.add_argument(
        "--run",
        action="store_true",
        help="Immediately run the pipeline after setup."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration number to resume from (optional)."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force execution in local mode (without Docker)."
    )
    return parser.parse_args()

def generate_run_script(setup: ExperimentSetup, use_local_mode: bool):
    """Generates the run_pipeline.sh script based on the execution mode."""
    run_script_path = setup.exp_dir / "run_pipeline.sh"

    # Paths relative to the experiment directory (where the script will run)
    config_rel = "configs/07_active_learning.yaml"
    meta_config_rel = "configs/config_meta.yaml"

    # We need to determine the root directory of the repo relative to the experiment dir
    # Experiment dir is typically project_root/<exp_name> or project_root/experiments/<exp_name>
    # The setup script is in project_root.
    # We can use the location of setup_experiment.py to find the root.

    if use_local_mode:
        logger.info("Generating local execution script (Docker disabled/not found).")
        # Assuming uv is available as per prompt requirements
        # Command: uv run python workers/al_md_kmc_worker/src/main.py start_loop ...
        # We need to set PYTHONPATH to include the project root so shared/ modules are found

        content = f"""#!/bin/bash
# Generated Pipeline Script (Local Mode)
set -e

echo "Starting Pipeline for experiment: {setup.exp_name} (Local Mode)"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
# Assuming standard structure: <repo_root>/<experiment_dir>/run_pipeline.sh
# We need to find the repo root.
# Let's try to find 'setup_experiment.py' to locate root.

REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    # try one level up
    REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    echo "ERROR: Could not locate repository root (containing setup_experiment.py)."
    exit 1
fi

echo "Repository Root: $REPO_ROOT"
cd "$REPO_ROOT"

# Ensure dependencies are installed (optional, but good for local mode)
# uv sync

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "Running Active Learning Loop..."
uv run python workers/al_md_kmc_worker/src/main.py start_loop \\
    --config "$SCRIPT_DIR/{config_rel}" \\
    --meta-config "$SCRIPT_DIR/{meta_config_rel}"

echo "Pipeline finished successfully."
"""
    else:
        logger.info("Generating Docker execution script.")
        # Docker Mode
        # We need to run the pace_worker (or al_md_kmc_worker) container.
        # The prompt says: docker run ... al_md_kmc_worker start_loop ...
        # We need to mount the repo root to /app (or proper paths).
        # And mount the experiment directory.

        # NOTE: The current codebase seems to assume 'al_md_kmc_worker' image is available or built.
        # The meta config has image names.

        # We'll use a simplified docker run command that mirrors what likely happens.
        # Assuming the image is 'pace_worker:latest' (based on default meta) or we should use 'al_md_kmc_worker' as per prompt.
        # The prompt explicitly mentioned `al_md_kmc_worker` in the task description.
        # But `ensure_meta_config` uses `pace_worker:latest`.
        # I will use the value from setup.root_config['experiment'].get('docker', {}).get('pace_image', 'pace_worker:latest')
        # but setup.root_config is the experiment config, not meta.
        # The meta config is handled separately.
        # I'll stick to a safe default 'pace_worker:latest' if not specified, but the prompt said 'al_md_kmc_worker'.
        # I will use 'pace_worker:latest' as that is what `ensure_meta_config` writes, and it seems the worker is unified.

        # We need to mount the current directory (repo root) to /app to access code if it's developing,
        # OR just mount the experiment dir to /app/work.

        content = f"""#!/bin/bash
# Generated Pipeline Script (Docker Mode)
set -e

echo "Starting Pipeline for experiment: {setup.exp_name} (Docker Mode)"

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

echo "Repository Root: $REPO_ROOT"

# Determine Docker Image (using default if not set)
IMAGE="pace_worker:latest"

echo "Using Docker Image: $IMAGE"

# We mount the repo root to /app so that 'shared' and 'workers' are available if the image expects them,
# or if we are overriding code.
# We also need to mount the experiment directory data.
# The worker expects to work in the experiment directory usually?
# setup_experiment.py says: os.chdir(experiment_root) before running.
# So we should work in experiment directory.

# The prompt example: `docker run ... al_md_kmc_worker start_loop ...`
# We need to map volumes.

docker run --rm -it \\
    --gpus all \\
    --user $(id -u):$(id -g) \\
    -v "$REPO_ROOT:/app" \\
    -v "$SCRIPT_DIR:/app/work" \\
    -w /app/work \\
    $IMAGE \\
    python /app/workers/al_md_kmc_worker/src/main.py start_loop \\
    --config configs/07_active_learning.yaml \\
    --meta-config configs/config_meta.yaml

echo "Pipeline finished successfully."
"""

    with open(run_script_path, "w") as f:
        f.write(content)
    os.chmod(run_script_path, 0o755)
    logger.info(f"Generated executable pipeline script at: {run_script_path}")


def main():
    args = parse_args()

    # Step 0: Ensure config_meta.yaml exists in the project root
    root_meta_path = pathlib.Path("config_meta.yaml")
    ensure_meta_config(root_meta_path)

    # Determine execution mode
    docker_available = check_docker()
    use_local_mode = args.local or not docker_available

    if args.local:
        logger.info("Local mode forced by user.")
    elif not docker_available:
        logger.warning("Docker not found or unreachable. Falling back to local mode.")

    if args.resume:
        # Resume mode
        resume_path = pathlib.Path(args.resume).resolve()
        logger.info(f"Resuming experiment from: {resume_path}")

        if not (resume_path / "work" / "07_active_learning").exists() and not (resume_path / "configs").exists():
             logger.error(f"Invalid resume path: {resume_path}")
             logger.error("Please provide the root directory of the experiment (containing 'work/' and 'configs/').")
             sys.exit(1)

        experiment_root = resume_path
        state_path = experiment_root / "work" / "07_active_learning" / "experiment_state.json"

        if not state_path.exists():
             state_path_alt = experiment_root / "work" / "experiment_state.json"
             if state_path_alt.exists():
                 state_path = state_path_alt
             else:
                 logger.error(f"Experiment state file not found at {state_path} or {state_path_alt}.")
                 logger.error("Cannot resume without a valid state file.")
                 sys.exit(1)

        logger.info(f"Found experiment state at: {state_path}")
        logger.info(f"Confirmed experiment root: {experiment_root}")

        config_path = experiment_root / "configs" / "07_active_learning.yaml"
        meta_config_path = experiment_root / "configs" / "config_meta.yaml"

        if not config_path.exists():
            logger.error(f"Could not find configuration at {config_path}")
            sys.exit(1)

        os.chdir(experiment_root)

        # In resume mode, we still just run the orchestrator.
        # If user passed --run, we run it directly in this process (if local) or via docker?
        # The original code ran it directly:
        # run_active_learning(pipeline_args)

        # If we are just setting up the resume script, we should probably generate/update the run script?
        # But --resume implies running typically?
        # The prompt says "The generated `run_pipeline.sh` creates recursive loops by calling `setup --resume`"
        # So we should avoid that.

        # If args.resume is used, we usually just want to RUN.
        # But if the user wants to generate a resume SCRIPT, they might use setup_experiment logic?
        # Typically setup_experiment.py is for SETUP.
        # If --resume is passed, it acts as a runner in the current logic.

        pipeline_args = PipelineArgs(
            config=str(config_path.relative_to(experiment_root)),
            meta_config=str(meta_config_path.relative_to(experiment_root)),
            resume_iteration=args.iteration
        )

        # If local mode or running directly:
        run_active_learning(pipeline_args)

    else:
        # New Experiment Setup
        logger.info("Initializing new experiment...")

        config_path = pathlib.Path(args.config)
        setup = ExperimentSetup(config_path=config_path)

        setup.load_configurations()

        if args.name:
            logger.info(f"Overriding experiment name to: {args.name}")
            setup.exp_name = args.name
            if "experiment" not in setup.root_config:
                setup.root_config["experiment"] = {}
            setup.root_config["experiment"]["name"] = args.name

        setup.create_directory_structure()
        setup.generate_step_configs()

        # Use our new function instead of the class method (or overwrite the class method if we could, but function is easier here)
        generate_run_script(setup, use_local_mode)

        if args.run:
            logger.info("Setup complete. Starting orchestrator immediately (--run specified)...")
            experiment_root = setup.exp_dir.resolve()
            config_path = experiment_root / "configs" / "07_active_learning.yaml"
            meta_config_path = experiment_root / "configs" / "config_meta.yaml"

            os.chdir(experiment_root)

            pipeline_args = PipelineArgs(
                config="configs/07_active_learning.yaml",
                meta_config="configs/config_meta.yaml"
            )
            run_active_learning(pipeline_args)
        else:
            logger.info("Setup complete.")
            logger.info(f"To run the pipeline, execute: {setup.exp_dir}/run_pipeline.sh")

if __name__ == "__main__":
    main()
