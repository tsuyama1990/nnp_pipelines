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
    safe_name = sanitize_exp_name(setup.exp_name)

    # Get image names from meta_config, with fallbacks
    docker_conf = setup.meta_config.get("docker", {})
    gen_image = docker_conf.get("gen_image", "gen_worker:latest")
    dft_image = docker_conf.get("dft_image", "dft_worker:latest")
    pace_image = docker_conf.get("pace_image", "pace_worker:latest")
    al_image = docker_conf.get("lammps_image", "al_md_kmc_worker:latest")

    # Define the monolithic config path to be used by all steps
    monolithic_config = "configs/monolithic.yaml"
    meta_config = "configs/config_meta.yaml"

    content = f"""#!/bin/bash
# Generated Full Pipeline Script (Docker Mode)
set -e

# --- Configuration ---
EXP_NAME="{safe_name}"
GEN_IMAGE="{gen_image}"
DFT_IMAGE="{dft_image}"
PACE_IMAGE="{pace_image}"
AL_IMAGE="{al_image}"
MONOLITHIC_CONFIG="{monolithic_config}"
META_CONFIG="{meta_config}"

# --- Helper Functions ---
step_header() {{
    echo "========================================================================"
    echo "  STEP: $1"
    echo "========================================================================"
}}

# --- Environment Setup ---
EXP_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
# The script is in the experiment dir, so REPO_ROOT is one level up
REPO_ROOT="$(dirname "$EXP_DIR")"
LOG_DIR="$EXP_DIR/logs"

echo "Starting Full Pipeline for experiment: $EXP_NAME"
echo "Experiment Directory: $EXP_DIR"
echo "Repository Root: $REPO_ROOT"
echo "---"

# Create working and log directories
mkdir -p "$EXP_DIR/work/01_generation"
mkdir -p "$EXP_DIR/work/04_dft"
mkdir -p "$EXP_DIR/work/05_training"
mkdir -p "$LOG_DIR"

# --- STEP 1: Initial Structure Generation ---
step_header "1. Initial Structure Generation"
echo "Using Docker Image: $GEN_IMAGE"
echo "Log file: $LOG_DIR/01_generation.log"
docker run --rm -i --gpus all \\
    -v "$EXP_DIR:/app/work" \\
    -w /app/work \\
    "$GEN_IMAGE" \\
    python /app/src/main.py generate \\
    --config "$MONOLITHIC_CONFIG" \\
    --output "work/01_generation/generated_structures.xyz" > "$LOG_DIR/01_generation.log" 2>&1

echo "Generated initial structures."
echo "---"

# Check if structures were generated
if [ ! -s "$EXP_DIR/work/01_generation/generated_structures.xyz" ]; then
    echo "ERROR: Step 1 failed to generate any structures. See log for details: $LOG_DIR/01_generation.log"
    exit 1
fi

# --- STEP 2: DFT Labeling of Seed Structures ---
step_header "2. DFT Labeling for Seed Potential"
echo "Using Docker Image: $DFT_IMAGE"
echo "Log file: $LOG_DIR/02_dft_labeling.log"
# Mount qe_calc directory for pseudopotentials and SSSP files
QE_CALC_DIR="$HOME/qe_calc"
docker run --rm -i \\
    --user $(id -u):$(id -g) \\
    -v "$EXP_DIR:/app/work" \\
    -v "$QE_CALC_DIR:/qe_calc" \\
    -w /app/work \\
    "$DFT_IMAGE" \\
    python3 /app/src/main.py \\
    --config "$MONOLITHIC_CONFIG" \\
    --meta-config "$META_CONFIG" \\
    --structure "work/01_generation/generated_structures.xyz" \\
    --output "work/04_dft/labeled_seed_structures.xyz" > "$LOG_DIR/02_dft_labeling.log" 2>&1

echo "DFT labeling complete."
echo "---"

# --- STEP 3: Train Seed Potential ---
step_header "3. Training Seed Potential"
echo "Using Docker Image: $PACE_IMAGE"
echo "Log file: $LOG_DIR/03_seed_training.log"
docker run --rm -i --gpus all \\
    --user $(id -u):$(id -g) \\
    -v "$EXP_DIR:/app/work" \\
    -w /app/work \\
    "$PACE_IMAGE" \\
    python /app/src/main.py train \\
    --config "$MONOLITHIC_CONFIG" \\
    --meta-config "$META_CONFIG" \\
    --dataset "work/04_dft/labeled_seed_structures.xyz" > "$LOG_DIR/03_seed_training.log" 2>&1

# The trainer should create 'fept_potential.yace' in the workdir.
# Let's find it and copy it to the root of the experiment dir for the next step.
SEED_POTENTIAL_NAME="fept_potential.yace" # As defined in config_fept.yaml
GENERATED_POTENTIAL=$(find "$EXP_DIR/work" -name "*.yace" | head -n 1)

if [ -n "$GENERATED_POTENTIAL" ]; then
    echo "Found potential at $GENERATED_POTENTIAL. Copying to $EXP_DIR/$SEED_POTENTIAL_NAME"
    cp "$GENERATED_POTENTIAL" "$EXP_DIR/$SEED_POTENTIAL_NAME"
else
    echo "ERROR: Seed potential not found after training step. See log for details: $LOG_DIR/03_seed_training.log"
    exit 1
fi
echo "Seed potential created successfully."
echo "---"

# --- STEP 4: Start Active Learning Loop ---
step_header "4. Active Learning Loop"
echo "Using Docker Image: $AL_IMAGE"
echo "Log file: $LOG_DIR/04_active_learning.log"
docker run --rm -it \\
    --gpus all \\
    --user $(id -u):$(id -g) \\
    -v "$EXP_DIR:/app/work" \\
    -w /app/work \\
    "$AL_IMAGE" \\
    python /app/src/main.py start_loop \\
    --config "$MONOLITHIC_CONFIG" \\
    --meta-config "$META_CONFIG" > "$LOG_DIR/04_active_learning.log" 2>&1

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

            # --- Quickstart Hotfix: Ensure initial potential file exists ---
            if setup.root_config:
                al_params = setup.root_config.get("al_params", {})
                initial_potential_name = al_params.get("initial_potential")
                if initial_potential_name:
                    potential_path = experiment_root / initial_potential_name
                    if not potential_path.exists():
                        logger.info(f"Initial potential '{initial_potential_name}' not found. Creating empty placeholder for quickstart.")
                        potential_path.touch()
            # -----------------------------------------------------------

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
