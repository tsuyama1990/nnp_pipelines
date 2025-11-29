import unittest
import yaml
import pathlib
import tempfile
import shutil
import sys
import os

# Ensure the orchestrator is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from orchestrator.src.setup.experiment_setup import ExperimentSetup

class TestExperimentSetup(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = pathlib.Path(self.test_dir) / "config.yaml"
        self.constant_path = pathlib.Path(self.test_dir) / "constant.yaml"
        self.meta_path = pathlib.Path(self.test_dir) / "config_meta.yaml"

        # Create dummy config
        self.config_data = {
            "experiment": {"name": "test_exp"},
            "md_params": {"elements": ["Fe"], "masses": {"Fe": 55.8}},
            "generation": {"some": "param"},
            "seed_generation": {
                "n_random_structures": 10,
                "crystal_type": "bcc",
                "exploration_temperatures": [300, 600],
                "n_md_steps": 100,
                "n_samples_for_dft": 5,
                "types": ["Fe"]
            },
            "dft_params": {"cutoff": 500},
            "ace_model": {"r_cut": 5.0},
            "training_params": {"max_iter": 10},
            "exploration": {"strategy": "random"},
            "al_params": {"threshold": 0.1},
            "kmc_params": {"temperature": 300}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

        with open(self.constant_path, 'w') as f:
            yaml.dump({"some_constant": 1}, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_setup_run(self):
        setup = ExperimentSetup(
            config_path=self.config_path,
            meta_config_path=self.meta_path,
            constant_config_path=self.constant_path
        )

        # Override exp_name generation to use temp dir path if needed,
        # but the class creates experiment_{name} relative to CWD.
        # We should probably mock creating directories to avoid polluting CWD or chdir.

        original_cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            setup.run()

            exp_dir = pathlib.Path("experiment_test_exp")
            self.assertTrue(exp_dir.exists())
            self.assertTrue((exp_dir / "configs").exists())
            self.assertTrue((exp_dir / "data").exists())
            self.assertTrue((exp_dir / "work").exists())

            # Check a generated file
            self.assertTrue((exp_dir / "configs" / "01_generation.yaml").exists())
            with open(exp_dir / "configs" / "01_generation.yaml", 'r') as f:
                gen_config = yaml.safe_load(f)
                self.assertEqual(gen_config["seed_generation"]["crystal_type"], "bcc")

            # Check script
            self.assertTrue((exp_dir / "configs" / "run_pipeline.sh").exists())

        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    unittest.main()
