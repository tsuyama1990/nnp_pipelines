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
import re
from typing import Optional

# Updated import path for Unified Worker
# We only need ExperimentSetup for setup logic.
# Execution is handled via subprocess now.
sys.path.append(os.path.join(os.path.dirname(__file__), "workers/al_md_kmc_worker/src"))
from workers.al_md_kmc_worker.src.setup.experiment_setup import ExperimentSetup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).parent.resolve()

def check_secure_path(path_str: str, base_dir: pathlib.Path = ROOT_DIR) -> pathlib.Path:
    """
    Resolves a path and ensures it is safely within the expected base directory.
    Prevents directory traversal attacks.
    """
    path = pathlib.Path(path_str).resolve()
    # Check if path is relative to base_dir
    # For safety, we allow paths inside the repo root.
    if not path.is_relative_to(base_dir):
        # Exception: system paths like /usr/bin/docker are allowed if we were checking executables,
        # but here we check config/resume paths.
        raise PermissionError(f"Path traversal detected! {path} is outside allowed root {base_dir}")
    return path

def ensure_meta_config(path: pathlib.Path):
    """
    Ensures that the meta configuration file exists in the project root.
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

def sanitize_exp_name(name: str) -> str:
    """Sanitizes experiment name to prevent shell injection in scripts."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Invalid experiment name: {name}. Only alphanumeric, underscore, and hyphen allowed.")
    return name

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

    # Sanitize name just in case
    safe_name = sanitize_exp_name(setup.exp_name)

    if use_local_mode:
        logger.info("Generating local execution script (Docker disabled/not found).")

        content = f"""#!/bin/bash
# Generated Pipeline Script (Local Mode)
set -e

echo "Starting Pipeline for experiment: {safe_name} (Local Mode)"

# Get the directory of this script (experiment root)
EXP_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
SCRIPT_DIR="$EXP_DIR"

# Locate Repo Root
# We assume setup_experiment.py is in Repo Root.
REPO_ROOT="$(dirname "$EXP_DIR")"
# Adjust if experiment dir is deeper (e.g. experiments/name)
if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    echo "ERROR: Could not locate repository root (containing setup_experiment.py)."
    exit 1
fi

echo "Repository Root: $REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Relative paths for config inside the experiment dir
CONFIG_REL="configs/07_active_learning.yaml"
META_REL="configs/config_meta.yaml"

echo "Running Active Learning Loop..."
# Pass absolute paths to the worker, running from Experiment Directory as CWD
cd "$EXP_DIR"

uv run python "$REPO_ROOT/workers/al_md_kmc_worker/src/main.py" start_loop \\
    --config "$EXP_DIR/$CONFIG_REL" \\
    --meta-config "$EXP_DIR/$META_REL"

echo "Pipeline finished successfully."
"""
    else:
        logger.info("Generating Docker execution script.")
        image = "pace_worker:latest" # Default, or fetch from meta if parsed

        content = f"""#!/bin/bash
# Generated Pipeline Script (Docker Mode)
set -e

echo "Starting Pipeline for experiment: {safe_name} (Docker Mode)"

EXP_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
REPO_ROOT="$(dirname "$EXP_DIR")"
if [ ! -f "$REPO_ROOT/setup_experiment.py" ]; then
    REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

echo "Repository Root: $REPO_ROOT"
IMAGE="{image}"
echo "Using Docker Image: $IMAGE"

# Mount repo to /app, experiment to /app/work (conceptually)
# But worker expects to run inside /app/work?
# Actually, the worker script path in docker is /app/workers/al_md_kmc_worker/src/main.py if we mount repo to /app.

docker run --rm -it \\
    --gpus all \\
    --user $(id -u):$(id -g) \\
    -v "$REPO_ROOT:/app" \\
    -v "$EXP_DIR:/app/work" \\
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
    root_meta_path = ROOT_DIR / "config_meta.yaml"
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
        # Fix Path Traversal
        try:
            resume_path = check_secure_path(args.resume)
        except PermissionError as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"Resuming experiment from: {resume_path}")

        if not (resume_path / "work" / "07_active_learning").exists() and not (resume_path / "configs").exists():
             logger.error(f"Invalid resume path: {resume_path}")
             logger.error("Please provide the root directory of the experiment (containing 'work/' and 'configs/').")
             sys.exit(1)

        experiment_root = resume_path
        # Fix: Do not chdir. Use absolute paths.

        config_path = experiment_root / "configs" / "07_active_learning.yaml"
        meta_config_path = experiment_root / "configs" / "config_meta.yaml"

        if not config_path.exists():
            logger.error(f"Could not find configuration at {config_path}")
            sys.exit(1)

        # Run logic (same as --run)
        logger.info("Starting active learning loop (RESUME)...")

        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            str(ROOT_DIR / "workers/al_md_kmc_worker/src/main.py"),
            "start_loop",
            "--config", str(config_path),
            "--meta-config", str(meta_config_path)
        ]

        try:
            # Pass cwd=experiment_root to subprocess
            subprocess.run(cmd, cwd=experiment_root, check=True, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)

    else:
        # New Experiment Setup
        logger.info("Initializing new experiment...")

        try:
            config_path = check_secure_path(args.config)
        except PermissionError as e:
            logger.error(str(e))
            sys.exit(1)

        setup = ExperimentSetup(config_path=config_path)
        setup.load_configurations()

        if args.name:
            # Sanitize name
            try:
                safe_name = sanitize_exp_name(args.name)
            except ValueError as e:
                logger.error(str(e))
                sys.exit(1)

            logger.info(f"Overriding experiment name to: {safe_name}")
            setup.exp_name = safe_name
            if "experiment" not in setup.root_config:
                setup.root_config["experiment"] = {}
            setup.root_config["experiment"]["name"] = safe_name

        setup.create_directory_structure()
        setup.generate_step_configs()

        generate_run_script(setup, use_local_mode)

        if args.run:
            logger.info("Setup complete. Starting orchestrator immediately (--run specified)...")
            experiment_root = setup.exp_dir.resolve()

            config_path = experiment_root / "configs" / "07_active_learning.yaml"
            meta_config_path = experiment_root / "configs" / "config_meta.yaml"

            # Prepare environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [
                sys.executable,
                str(ROOT_DIR / "workers/al_md_kmc_worker/src/main.py"),
                "start_loop",
                "--config", str(config_path),
                "--meta-config", str(meta_config_path)
            ]

            try:
                # Pass cwd=experiment_root to subprocess
                subprocess.run(cmd, cwd=experiment_root, check=True, env=env)
            except subprocess.CalledProcessError as e:
                logger.error(f"Pipeline execution failed with exit code {e.returncode}")
                sys.exit(e.returncode)
        else:
            logger.info("Setup complete.")
            logger.info(f"To execute: {setup.exp_dir}/run_pipeline.sh")

if __name__ == "__main__":
    main()
