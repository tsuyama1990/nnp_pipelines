"""Tests for the seed generation pipeline."""

import unittest
from unittest.mock import MagicMock, patch, mock_open, call
from pathlib import Path
import sys
import yaml

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from orchestrator.workflows.seed_generation import SeedGenerator
from shared.core.config import Config, SeedGenerationParams, MDParams, ALParams, MetaConfig

class TestSeedGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock(spec=Config)
        self.mock_config.md_params = MagicMock()
        self.mock_config.md_params.elements = ["Al", "Cu"]
        self.mock_config.seed_generation = SeedGenerationParams(
            n_random_structures=5,
            exploration_temperatures=[100.0, 200.0],
            n_md_steps=50,
            n_samples_for_dft=3
        )

        self.config_path = Path("config.yaml")
        self.meta_config_path = Path("meta_config.yaml")

        self.host_data_dir = Path("data").resolve()

    @patch("orchestrator.workflows.seed_generation.LammpsWorker")
    @patch("orchestrator.workflows.seed_generation.PaceWorker")
    @patch("orchestrator.workflows.seed_generation.DftWorker")
    @patch("orchestrator.workflows.seed_generation.GenWorker")
    @patch("orchestrator.workflows.seed_generation.read")
    @patch("orchestrator.workflows.seed_generation.write")
    @patch("orchestrator.workflows.seed_generation.shutil")
    @patch("orchestrator.workflows.seed_generation.pd")
    @patch("builtins.open", new_callable=mock_open)
    @patch("orchestrator.workflows.seed_generation.yaml")
    def test_run_logic(self, mock_yaml, mock_file, mock_pd, mock_shutil, mock_write, mock_read,
                       MockGenWorker, MockDftWorker, MockPaceWorker, MockLammpsWorker):

        # Setup Mocks
        gen_worker = MockGenWorker.return_value
        dft_worker = MockDftWorker.return_value
        pace_worker = MockPaceWorker.return_value
        lammps_worker = MockLammpsWorker.return_value

        # Mock reading random structures
        # 5 structures generated
        mock_atoms = MagicMock()
        mock_read.side_effect = [
            [mock_atoms] * 5, # read(rand_structures_path)
            [mock_atoms] * 10, # read(traj_path) - inside loop (called 5*2 times actually, but logic combines results)
             # Wait, read is called inside the loop for each trajectory.
             # 5 structures * 2 temps = 10 MD runs.
             # read is called 10 times to concatenate.
        ]

        # Fix mock_read side effect to return a list of atoms for each call
        def read_side_effect(*args, **kwargs):
            if "random_structures" in str(args[0]):
                return [mock_atoms] * 5
            elif "traj_" in str(args[0]): # Trajectories
                return [mock_atoms] # 1 frame per traj for simplicity
            elif "labeled_seed" in str(args[0]):
                return [mock_atoms] * 3
            return []
        mock_read.side_effect = read_side_effect

        # Mock shutil move to simulate successful MD output
        mock_shutil.move.return_value = True

        # Mock file existence checks
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = True

            # Initialize Generator
            generator = SeedGenerator(self.mock_config, self.config_path, self.meta_config_path)

            # Override host_data_dir to avoid real IO attempts if any slip through
            generator.host_data_dir = Path("/tmp/mock_data")

            # Execute
            generator.run()

            # Verify GenWorker calls
            gen_worker.generate.assert_called_once()

            # Verify LammpsWorker calls
            # Expected calls: 5 structures * 2 temps = 10 calls
            self.assertEqual(lammps_worker.run_md.call_count, 10)

            # Verify correct arguments for LammpsWorker
            # Specifically check if potential_filename is None (Pure LJ)
            call_args = lammps_worker.run_md.call_args_list[0]
            _, kwargs = call_args
            self.assertIsNone(kwargs['potential_filename'])
            self.assertEqual(kwargs['steps'], 50)

            # Verify PaceWorker sampling
            pace_worker.direct_sample.assert_called_once()
            _, sample_kwargs = pace_worker.direct_sample.call_args
            self.assertEqual(sample_kwargs['n_clusters'], 3) # From config

    @patch("shared.core.config.Config.load_meta")
    def test_config_loading(self, mock_load_meta):
        # Test if config system loads seed_generation section correctly
        # This requires creating a real Config object from dict

        config_dict = {
            "experiment": {"name": "test", "output_dir": "out"},
            "md_params": {
                "timestep": 1.0, "temperature": 300, "pressure": 1.0,
                "n_steps": 1000, "elements": ["Al"], "initial_structure": "s.data",
                "masses": {"Al": 26.98}
            },
            "al_params": {
                "gamma_threshold": 0.1, "n_clusters": 5, "r_core": 4.0, "box_size": 12.0,
                "initial_potential": "p.yace", "potential_yaml_path": "p.yaml"
            },
            "seed_generation": {
                "n_random_structures": 50,
                "exploration_temperatures": [500.0],
                "n_samples_for_dft": 10
            }
        }

        meta_config = MetaConfig(dft={}, lammps={})

        # Config.from_dict requires a complete dictionary or defaults handling logic inside from_dict
        # We need to ensure we provide required fields for ALParams which failed previously

        config_obj = Config.from_dict(config_dict, meta_config)

        self.assertEqual(config_obj.seed_generation.n_random_structures, 50)
        self.assertEqual(config_obj.seed_generation.exploration_temperatures, [500.0])
        self.assertEqual(config_obj.seed_generation.n_samples_for_dft, 10)
        # Default value check
        self.assertEqual(config_obj.seed_generation.n_md_steps, 1000)

if __name__ == '__main__':
    unittest.main()
