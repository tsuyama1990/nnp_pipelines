import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile
import yaml
import pytest
from ase import Atoms
from ase.io import write

# FIX: Add workers/dft_worker/src to sys.path so that 'configurator' can be imported by main.py
WORKER_SRC = Path(__file__).resolve().parent.parent / "src"
if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from workers.dft_worker.src.main import main

class TestDFTWorkerMain(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "config.yaml"
        self.meta_config_path = Path(self.test_dir) / "meta_config.yaml"
        self.structure_path = Path(self.test_dir) / "input.xyz"
        self.output_path = Path(self.test_dir) / "output.xyz"

        # Create Dummy Structure
        self.atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
        write(self.structure_path, self.atoms)

        # Create Dummy Configs
        self.meta_config = {
            "dft": {
                "command": "pw.x",
                "pseudo_dir": str(self.test_dir),
                "sssp_json_path": str(Path(self.test_dir) / "sssp.json")
            },
            "lammps": {}
        }
        with open(self.meta_config_path, "w") as f:
            yaml.dump(self.meta_config, f)

        self.config = {
            "experiment": {"name": "test", "output_dir": str(self.test_dir)},
            "dft_params": {"kpoint_density": 0.04, "auto_physics": False},
            "ace_model": {"delta_learning_mode": False}, # Pure DFT
            "al_params": {"outlier_energy_max": 10.0, "gamma_threshold": 0.1, "n_clusters": 1, "r_core": 3.0, "initial_potential": "x", "potential_yaml_path": "y"},
            "md_params": {"timestep": 1.0, "temperature": 300, "pressure": 0, "n_steps": 10, "elements": ["H"], "initial_structure": "x", "masses": {"H": 1.0}},
            "lj_params": {"epsilon": 1.0, "sigma": 1.0, "cutoff": 3.0},
            "exploration": {"strategy": "hybrid"},
            "training_params": {"replay_ratio": 1.0, "global_dataset_path": "data/global.pckl"}
        }
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        # Create dummy sssp.json
        sssp_data = {
            "H": {
                "filename": "H.upf",
                "cutoff_wfc": 30,
                "cutoff_rho": 120
            }
        }
        with open(Path(self.test_dir) / "sssp.json", "w") as f:
            import json
            json.dump(sssp_data, f)

        # Create dummy UPF file (AtomicEnergyManager checks for existence if it calculates)
        (Path(self.test_dir) / "H.upf").touch()
        # Also need H.json for E0 if we want to mock E0 loading, but we will mock AtomicEnergyManager

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    @patch("workers.dft_worker.src.main.AtomicEnergyManager")
    @patch("ase.calculators.espresso.EspressoTemplate.read_results")
    @patch("subprocess.Popen")
    @patch("workers.dft_worker.src.main.read")
    @patch("workers.dft_worker.src.main.write")
    def test_main_constructs_pw_command(self, mock_write, mock_read, mock_popen, mock_read_results, mock_ae_manager_cls):
        # Mock Read Results to return valid energy
        import numpy as np
        mock_read_results.return_value = {'energy': -10.0, 'forces': [[0.0, 0.0, 0.0]] * 2, 'stress': np.zeros(6)}

        # Mock CLI args
        test_args = [
            "main.py",
            "--config", str(self.config_path),
            "--meta-config", str(self.meta_config_path),
            "--structure", str(self.structure_path),
            "--output", str(self.output_path)
        ]

        # Mock Read to return our atoms
        mock_read.return_value = [self.atoms]

        # Mock AtomicEnergyManager
        mock_ae_manager = mock_ae_manager_cls.return_value
        # Mock get_e0 to return a dictionary without calling the factory
        mock_ae_manager.get_e0.return_value = {"H": -13.6}

        # Setup Subprocess Mock
        process_mock = MagicMock()
        process_mock.communicate.return_value = (b"output", b"error")
        process_mock.returncode = 0
        process_mock.wait.return_value = 0
        process_mock.__enter__.return_value = process_mock
        mock_popen.return_value = process_mock

        # Run Main
        with patch.object(sys, 'argv', test_args):
            main()

        # Verification
        # 1. Check that subprocess.Popen was called
        self.assertTrue(mock_popen.called, "subprocess.Popen should have been called")

        # 2. Verify command arguments
        call_args_list = mock_popen.call_args_list
        found_pw_command = False
        for call in call_args_list:
            args, kwargs = call
            cmd_list = args[0] # Popen first arg

            # Check if command list contains 'pw.x'
            if isinstance(cmd_list, list) and 'pw.x' in cmd_list:
                found_pw_command = True
                break
            elif isinstance(cmd_list, str) and 'pw.x' in cmd_list:
                found_pw_command = True
                break

        self.assertTrue(found_pw_command, f"Could not find 'pw.x' in subprocess calls: {call_args_list}")

if __name__ == "__main__":
    unittest.main()
