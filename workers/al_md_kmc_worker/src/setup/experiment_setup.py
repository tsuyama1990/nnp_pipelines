import yaml
import pathlib
import os
import stat
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentSetup:
    """
    Handles the initialization of an experiment by creating directories,
    splitting configurations, and generating execution scripts.
    """

    def __init__(self, config_path: pathlib.Path, meta_config_path: pathlib.Path = pathlib.Path("config_meta.yaml"), constant_config_path: pathlib.Path = pathlib.Path("constant.yaml")):
        self.config_path = config_path
        self.meta_config_path = meta_config_path
        self.constant_config_path = constant_config_path
        self.root_config: Dict[str, Any] = {}
        self.constant_config: Dict[str, Any] = {}
        self.meta_config: Dict[str, Any] = {}
        self.exp_name: str = "default_experiment"
        self.exp_dir: Optional[pathlib.Path] = None
        self.configs_dir: Optional[pathlib.Path] = None
        self.data_dir: Optional[pathlib.Path] = None
        self.work_dir: Optional[pathlib.Path] = None

    def load_configurations(self):
        """Loads all necessary configuration files."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} not found.")

        self.root_config = self._load_yaml(self.config_path)
        self.constant_config = self._load_yaml(self.constant_config_path) if self.constant_config_path.exists() else {}
        self.meta_config = self._ensure_meta_config(self.meta_config_path)
        self.exp_name = self.root_config.get("experiment", {}).get("name", "default_experiment")

    def _load_yaml(self, path: pathlib.Path) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, data: Dict[str, Any], path: pathlib.Path):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)

    def _ensure_meta_config(self, path: pathlib.Path) -> Dict[str, Any]:
        if not path.exists():
            print(f"Meta config not found at {path}. Creating default template.")
            default_meta = {
                "dft_cmd": "pw.x",
                "lammps_cmd": "lmp_serial",
                "mpirun_cmd": "mpirun -np 4"
            }
            self._save_yaml(default_meta, path)
        return self._load_yaml(path)

    def create_directory_structure(self):
        """Creates the experiment directory structure."""
        self.exp_dir = pathlib.Path(f"experiment_{self.exp_name}")
        self.configs_dir = self.exp_dir / "configs"
        self.data_dir = self.exp_dir / "data"
        self.work_dir = self.exp_dir / "work"

        for d in [self.configs_dir, self.data_dir, self.work_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"Initialized experiment directory: {self.exp_dir}")

    def generate_step_configs(self):
        """Generates modular configuration files for each step."""
        if not self.configs_dir:
            raise RuntimeError("Directories not created. Call create_directory_structure first.")

        # configs/config_meta.yaml - Snapshotting the meta config
        self._save_yaml(self.meta_config, self.configs_dir / "config_meta.yaml")

        # configs/core.yaml
        core_config = {
            "experiment": self.root_config.get("experiment", {}),
            "constants": self.constant_config,
            "global_physics": {
                "elements": self.root_config.get("md_params", {}).get("elements", []),
                "masses": self.root_config.get("md_params", {}).get("masses", {})
            }
        }
        self._save_yaml(core_config, self.configs_dir / "core.yaml")

        self._generate_gen_config()
        self._generate_pretrain_config()
        self._generate_sampling_config()
        self._generate_dft_config()
        self._generate_training_config()
        self._generate_prod_config()
        self._generate_al_config()

    def _generate_gen_config(self):
        gen_config = {
            "generation": self.root_config.get("generation", {}),
            "seed_generation": {
                "n_random_structures": self.root_config.get("seed_generation", {}).get("n_random_structures"),
                "crystal_type": self.root_config.get("seed_generation", {}).get("crystal_type")
            }
        }
        self._save_yaml(gen_config, self.configs_dir / "01_generation.yaml")

    def _generate_pretrain_config(self):
        pretrain_config = {
            "seed_generation": {
                "exploration_temperatures": self.root_config.get("seed_generation", {}).get("exploration_temperatures"),
                "n_md_steps": self.root_config.get("seed_generation", {}).get("n_md_steps")
            },
            "model": {
                "type": "mace_medium",
                "fmax": 100.0
            }
        }
        self._save_yaml(pretrain_config, self.configs_dir / "02_pretraining.yaml")

    def _generate_sampling_config(self):
        sampling_config = {
            "seed_generation": {
                "n_samples_for_dft": self.root_config.get("seed_generation", {}).get("n_samples_for_dft")
            },
            "strategy": "direct_selection"
        }
        self._save_yaml(sampling_config, self.configs_dir / "03_sampling.yaml")

    def _generate_dft_config(self):
        dft_config = {
            "dft_params": self.root_config.get("dft_params", {}),
            "seed_generation": {
                "types": self.root_config.get("seed_generation", {}).get("types")
            }
        }
        self._save_yaml(dft_config, self.configs_dir / "04_dft.yaml")

    def _generate_training_config(self):
        training_config = {
            "ace_model": self.root_config.get("ace_model", {}),
            "training_params": self.root_config.get("training_params", {})
        }
        self._save_yaml(training_config, self.configs_dir / "05_training.yaml")

    def _generate_prod_config(self):
        prod_config = {
            "md_params": self.root_config.get("md_params", {})
        }
        self._save_yaml(prod_config, self.configs_dir / "06_production.yaml")

    def _generate_al_config(self):
        al_config = {
            "exploration": self.root_config.get("exploration", {}),
            "al_params": self.root_config.get("al_params", {}),
            "kmc_params": self.root_config.get("kmc_params", {}),
            # Required by Config model for full validation even if used in other steps
            "md_params": self.root_config.get("md_params", {}),
            "training_params": self.root_config.get("training_params", {}),
            "ace_model": self.root_config.get("ace_model", {}),
            "dft_params": self.root_config.get("dft_params", {})
        }
        self._save_yaml(al_config, self.configs_dir / "07_active_learning.yaml")

    def generate_pipeline_script(self):
        """Generates the bash script to run the pipeline."""
        if not self.configs_dir:
            raise RuntimeError("Directories not created. Call create_directory_structure first.")

        script_path = self.configs_dir / "run_pipeline.sh"

        script_content = self._get_script_content()

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

        print(f"Generated pipeline script: {script_path}")

    def _get_script_content(self) -> str:
        # Epic 9: Updated pipeline script for Unified Worker
        return f"""#!/bin/bash
# Pipeline Execution Script for Experiment: {self.exp_name}
# This script contains commented-out commands to run each step of the pipeline.
# Uncomment lines to execute steps.

# Define Directories
# Note: $(pwd) refers to the 'configs' directory where this script is located.
WORK_DIR=$(pwd)/../work
CONFIGS_DIR=$(pwd)
DATA_DIR=$(pwd)/../data
ROOT_DIR=$(pwd)/../..

# Ensure directories exist (in case running from elsewhere)
mkdir -p $WORK_DIR $DATA_DIR

echo "Starting Pipeline for {self.exp_name}..."

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
# Using Unified Worker for MD
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data al_md_kmc_worker:latest \\
#     python /app/src/main.py md \\
#     --config $CONFIGS_DIR/02_pretraining.yaml \\
#     --meta-config $CONFIGS_DIR/config_meta.yaml \\
#     --structure $DATA_DIR/01_raw_structures.xyz \\
#     --steps 1000 --gamma 1.0 \\
#     --potential None

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
#     --meta-config $CONFIGS_DIR/config_meta.yaml \\
#     --structure $DATA_DIR/03_selected_structures.xyz \\
#     --output $DATA_DIR/04_labeled_dataset.xyz

# -----------------------------------------------------------------------------
# Step 5: Training
# -----------------------------------------------------------------------------
echo "Step 5: Training"
# docker run --rm --gpus all -v $ROOT_DIR:/app -v $DATA_DIR:/data pace_worker:latest \\
#     python /app/src/main.py train \\
#     --config $CONFIGS_DIR/05_training.yaml \\
#     --meta-config $CONFIGS_DIR/config_meta.yaml \\
#     --dataset $DATA_DIR/04_labeled_dataset.xyz \\
#     --iteration 1

# -----------------------------------------------------------------------------
# Step 6: Production MD
# -----------------------------------------------------------------------------
echo "Step 6: Production MD"
# docker run --rm -v $ROOT_DIR:/app -v $DATA_DIR:/data al_md_kmc_worker:latest \\
#     python /app/src/main.py md \\
#     --config $CONFIGS_DIR/06_production.yaml \\
#     --meta-config $CONFIGS_DIR/config_meta.yaml \\
#     --potential $DATA_DIR/potential.yace \\
#     --structure $DATA_DIR/initial_state.xyz \\
#     --steps 10000 --gamma 0.1

# -----------------------------------------------------------------------------
# Step 7: Active Learning Loop (Unified Worker)
# -----------------------------------------------------------------------------
echo "Step 7: Active Learning"
# Runs the full AL loop.
# IMPORTANT: Mounting docker.sock and setting HOST_WORK_DIR for DinD support.
# HOST_WORK_DIR should point to the 'work' directory on the host machine.
# Here we assume WORK_DIR maps correctly.
# docker run --rm \\
#     -v /var/run/docker.sock:/var/run/docker.sock \\
#     -v $ROOT_DIR:/app \\
#     -v $WORK_DIR:/app/work \\
#     -v $DATA_DIR:/data \\
#     -e HOST_WORK_DIR=$WORK_DIR \\
#     al_md_kmc_worker:latest \\
#     python /app/src/main.py start_loop \\
#     --config $CONFIGS_DIR/07_active_learning.yaml \\
#     --meta-config $CONFIGS_DIR/config_meta.yaml

echo "Pipeline setup complete. Uncomment commands in run_pipeline.sh to run."
"""

    def run(self):
        """Execute the full setup process."""
        self.load_configurations()
        self.create_directory_structure()
        self.generate_step_configs()
        self.generate_pipeline_script()
        print("Setup completed successfully.")
