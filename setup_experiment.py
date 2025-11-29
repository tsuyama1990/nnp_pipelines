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
import yaml
import os
import sys
import stat

def load_yaml(path):
    """Load a YAML file."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    """Save data to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

def ensure_meta_config(path):
    """Ensure config_meta.yaml exists, creating it with defaults if not."""
    if not path.exists():
        print(f"Meta config not found at {path}. Creating default template.")
        default_meta = {
            "dft_cmd": "pw.x",
            "lammps_cmd": "lmp_serial",
            "mpirun_cmd": "mpirun -np 4"
        }
        save_yaml(default_meta, path)
    return load_yaml(path)

def main():
    parser = argparse.ArgumentParser(description="Initialize a new experiment.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"),
                        help="Path to the monolithic config file.")
    args = parser.parse_args()

    # 1. Load Configurations
    root_config_path = args.config
    constant_config_path = pathlib.Path("constant.yaml")
    meta_config_path = pathlib.Path("config_meta.yaml")

    if not root_config_path.exists():
        print(f"Error: Config file {root_config_path} not found.")
        sys.exit(1)

    root_config = load_yaml(root_config_path)
    constant_config = load_yaml(constant_config_path)
    meta_config = ensure_meta_config(meta_config_path)

    # 2. Create Directory Structure
    exp_name = root_config.get("experiment", {}).get("name", "default_experiment")
    exp_dir = pathlib.Path(f"experiment_{exp_name}")

    configs_dir = exp_dir / "configs"
    data_dir = exp_dir / "data"
    work_dir = exp_dir / "work"

    for d in [configs_dir, data_dir, work_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Initialized experiment directory: {exp_dir}")

    # 3. Mapping Logic

    # configs/core.yaml
    core_config = {
        "experiment": root_config.get("experiment", {}),
        "constants": constant_config if constant_config else {},
        "meta": meta_config,
        "global_physics": {
            "elements": root_config.get("md_params", {}).get("elements", []),
            "masses": root_config.get("md_params", {}).get("masses", {})
        }
    }
    save_yaml(core_config, configs_dir / "core.yaml")

    # configs/01_generation.yaml
    gen_config = {
        "generation": root_config.get("generation", {}),
        "seed_generation": {
            "n_random_structures": root_config.get("seed_generation", {}).get("n_random_structures"),
            "crystal_type": root_config.get("seed_generation", {}).get("crystal_type")
        }
    }
    save_yaml(gen_config, configs_dir / "01_generation.yaml")

    # configs/02_pretraining.yaml
    pretrain_config = {
        "seed_generation": {
            "exploration_temperatures": root_config.get("seed_generation", {}).get("exploration_temperatures"),
            "n_md_steps": root_config.get("seed_generation", {}).get("n_md_steps")
        },
        "model": {
            "type": "mace_medium",
            "fmax": 100.0
        }
    }
    save_yaml(pretrain_config, configs_dir / "02_pretraining.yaml")

    # configs/03_sampling.yaml
    sampling_config = {
        "seed_generation": {
            "n_samples_for_dft": root_config.get("seed_generation", {}).get("n_samples_for_dft")
        },
        "strategy": "direct_selection"
    }
    save_yaml(sampling_config, configs_dir / "03_sampling.yaml")

    # configs/04_dft.yaml
    dft_config = {
        "dft_params": root_config.get("dft_params", {}),
        "seed_generation": {
            "types": root_config.get("seed_generation", {}).get("types")
        }
    }
    save_yaml(dft_config, configs_dir / "04_dft.yaml")

    # configs/05_training.yaml
    training_config = {
        "ace_model": root_config.get("ace_model", {}),
        "training_params": root_config.get("training_params", {})
    }
    save_yaml(training_config, configs_dir / "05_training.yaml")

    # configs/06_production.yaml
    prod_config = {
        "md_params": root_config.get("md_params", {})
    }
    save_yaml(prod_config, configs_dir / "06_production.yaml")

    # configs/07_active_learning.yaml
    al_config = {
        "exploration": root_config.get("exploration", {}),
        "al_params": root_config.get("al_params", {}),
        "kmc_params": root_config.get("kmc_params", {})
    }
    save_yaml(al_config, configs_dir / "07_active_learning.yaml")

    # 4. Generate run_pipeline.sh
    script_path = configs_dir / "run_pipeline.sh"

    script_content = f"""#!/bin/bash
# Pipeline Execution Script for Experiment: {exp_name}
# This script contains commented-out commands to run each step of the pipeline.
# Uncomment lines to execute steps.

# Define Directories
WORK_DIR=$(pwd)/../work
CONFIGS_DIR=$(pwd)
DATA_DIR=$(pwd)/../data
ROOT_DIR=$(pwd)/../..

# Ensure directories exist (in case running from elsewhere)
mkdir -p $WORK_DIR $DATA_DIR

echo "Starting Pipeline for {exp_name}..."

# -----------------------------------------------------------------------------
# Step 1: Generation
# -----------------------------------------------------------------------------
echo "Step 1: Generation"
# docker run --rm --gpus all -v $ROOT_DIR:/app -v $DATA_DIR:/data gen_worker:latest \\
#     python /app/src/main.py generate \\
#     --config $CONFIGS_DIR/01_generation.yaml \\
#     --output $DATA_DIR/01_raw_structures.xyz

# -----------------------------------------------------------------------------
# Step 2: Pre-training (Seed Exploration)
# -----------------------------------------------------------------------------
echo "Step 2: Pre-training"
# Note: This usually involves running MD on the generated structures to create diversity.
# Often performed by lammps_worker or a dedicated exploration routine using a baseline potential.
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data lammps_worker:latest \\
#     python /app/src/main.py md \\
#     --config $CONFIGS_DIR/02_pretraining.yaml \\
#     --structure $DATA_DIR/01_raw_structures.xyz \\
#     --output $DATA_DIR/02_explored_structures.xyz

# -----------------------------------------------------------------------------
# Step 3: Sampling (Direct Selection)
# -----------------------------------------------------------------------------
echo "Step 3: Sampling"
# docker run --rm --gpus all -v $ROOT_DIR:/app -v $DATA_DIR:/data pace_worker:latest \\
#     python /app/src/main.py direct_sample \\
#     --input $DATA_DIR/02_explored_structures.xyz \\
#     --output $DATA_DIR/03_selected_structures.xyz \\
#     --n_clusters 20

# -----------------------------------------------------------------------------
# Step 4: DFT Labeling
# -----------------------------------------------------------------------------
echo "Step 4: DFT Labeling"
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data dft_worker:latest \\
#     python /app/src/main.py \\
#     --config $CONFIGS_DIR/04_dft.yaml \\
#     --meta-config $CONFIGS_DIR/core.yaml \\
#     --structure $DATA_DIR/03_selected_structures.xyz \\
#     --output $DATA_DIR/04_labeled_dataset.xyz

# -----------------------------------------------------------------------------
# Step 5: Training
# -----------------------------------------------------------------------------
echo "Step 5: Training"
# docker run --rm --gpus all -v $ROOT_DIR:/app -v $DATA_DIR:/data pace_worker:latest \\
#     python /app/src/main.py train \\
#     --config $CONFIGS_DIR/05_training.yaml \\
#     --dataset $DATA_DIR/04_labeled_dataset.xyz \\
#     --iteration 1

# -----------------------------------------------------------------------------
# Step 6: Production MD
# -----------------------------------------------------------------------------
echo "Step 6: Production MD"
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data lammps_worker:latest \\
#     python /app/src/main.py md \\
#     --config $CONFIGS_DIR/06_production.yaml \\
#     --potential $DATA_DIR/potential.yace \\
#     --structure $DATA_DIR/initial_state.xyz \\
#     --steps 10000

# -----------------------------------------------------------------------------
# Step 7: Active Learning Loop
# -----------------------------------------------------------------------------
echo "Step 7: Active Learning"
# This step might involve an orchestrator loop or specific AL worker calls.
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data lammps_worker:latest \\
#     python /app/src/main.py md \\
#     --config $CONFIGS_DIR/07_active_learning.yaml \\
#     --potential $DATA_DIR/potential.yace \\
#     --gamma 0.1

echo "Pipeline setup complete. Uncomment commands in $script_path to run."
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)

    print(f"Generated pipeline script: {script_path}")
    print("Setup completed successfully.")

if __name__ == "__main__":
    main()
