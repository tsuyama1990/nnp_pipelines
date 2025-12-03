import os
import shutil
import subprocess
import sys
import yaml
import pytest
from pathlib import Path

# Locate repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup_experiment.py"
DEMO_CONFIG = REPO_ROOT / "quickstart/demo_config.yaml"
OUTPUT_DIR = REPO_ROOT / "output"

@pytest.fixture
def test_context():
    """
    Sets up a test context.
    We use the repo root as CWD because setup_experiment.py expects to be run from there
    (and ExperimentSetup creates dirs in CWD by default).
    We ensure we clean up the created experiment directory.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # We will name the experiment 'test_run'
    # So it will create 'experiment_test_run' in CWD (REPO_ROOT)
    exp_dir_name = "experiment_test_run"
    exp_dir = REPO_ROOT / exp_dir_name

    # Pre-cleanup
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    yield exp_dir

    # Post-cleanup
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

def test_real_execution_demo(test_context):
    """
    Runs setup_experiment.py with the demo configuration.
    Verifies that the script executes successfully and creates the expected artifacts.
    """
    assert SETUP_SCRIPT.exists(), "setup_experiment.py not found"
    assert DEMO_CONFIG.exists(), "demo_config.yaml not found"

    # Prepare a modified config file
    with open(DEMO_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    # We set name to 'test_run'
    config['experiment']['name'] = "test_run"

    # Patch missing fields to satisfy Pydantic strict validation
    if 'md_params' in config:
        config['md_params']['pressure'] = 1.0
        config['md_params']['initial_structure'] = "initial.data"
        config['md_params']['masses'] = {"Al": 26.98, "Cu": 63.55}
        config['md_params']['n_steps'] = 10

    if 'al_params' in config:
        config['al_params']['gamma_threshold'] = 0.1
        config['al_params']['potential_yaml_path'] = "potential.yaml"
        config['al_params']['initial_potential'] = "potential.yace"

    # Save the test config INSIDE the repo (in output dir)
    test_config_path = OUTPUT_DIR / "test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)

    # Create dummy initial.data in CWD (REPO_ROOT)
    dummy_structure = REPO_ROOT / "initial.data"
    dummy_structure.touch()

    try:
        # Construct command
        cmd = [
            sys.executable,
            str(SETUP_SCRIPT),
            "--config", str(test_config_path),
            "--name", "test_run",
            "--run"
        ]

        # Run the script
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        )

        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

        assert result.returncode == 0, f"Script failed with exit code {result.returncode}"

        # Verify artifacts
        exp_dir = test_context
        assert exp_dir.exists(), f"Experiment directory not created at {exp_dir}"

        # Check for run script
        assert (exp_dir / "run_pipeline.sh").exists(), "run_pipeline.sh not found"

        # Check for Step Configs
        assert (exp_dir / "configs" / "07_active_learning.yaml").exists()

        # Check for Experiment State or Log
        # We check for training_log.csv which is initialized early
        assert (exp_dir / "training_log.csv").exists(), "training_log.csv not found"

    finally:
        # Cleanup dummy structure
        if dummy_structure.exists():
            dummy_structure.unlink()
