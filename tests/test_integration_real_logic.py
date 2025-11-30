import unittest
import tempfile
import shutil
import json
from pathlib import Path
from ase import Atoms
from ase.io import write
from pydantic import ValidationError

# Ensure shared modules are available
import sys
import os
sys.path.append(os.getcwd())
worker_path = os.path.join(os.getcwd(), 'workers/al_md_kmc_worker')
if worker_path not in sys.path:
    sys.path.append(worker_path)

from shared.core.config import Config, MDParams, ALParams, MetaConfig, ExperimentConfig, ExplorationParams, DFTParams, LJParams, TrainingParams, ACEModelParams
from src.workflows.active_learning_loop import ActiveLearningOrchestrator
from src.state_manager import StateManager

# Add tests dir to path to import fakes
sys.path.append(os.path.join(os.getcwd(), 'tests'))
from fakes import FakeExplorer, FakeALService

class TestIntegrationRealLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.work_dir = Path(self.test_dir) / "work" / "07_active_learning"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Setup config
        meta_config = MetaConfig(
            dft={"command": "pw.x", "pseudo_dir": ".", "sssp_json_path": "sssp.json"},
            lammps={}
        )
        self.config = Config(
            meta=meta_config,
            experiment=ExperimentConfig(name="test", output_dir=Path("output")),
            exploration=ExplorationParams(),
            md_params=MDParams(
                timestep=1.0, temperature=300, pressure=1.0, n_steps=100,
                elements=["Al"], initial_structure="start.xyz", masses={"Al": 26.98}
            ),
            al_params=ALParams(
                gamma_threshold=0.1, n_clusters=2, r_core=3.0, box_size=10.0,
                initial_potential="pot.yace", potential_yaml_path="pot.yaml",
                initial_dataset_path=None, initial_active_set_path=None
            ),
            dft_params=DFTParams(),
            lj_params=LJParams(epsilon=1.0, sigma=1.0, cutoff=2.5),
            training_params=TrainingParams(),
            ace_model=ACEModelParams()
        )

        # Create initial files
        self.original_cwd = Path.cwd()
        # Create start.xyz
        atoms = Atoms('Al', positions=[[0, 0, 0]])
        write(str(Path(self.test_dir) / "start.xyz"), atoms)

        # Create potential
        (Path(self.test_dir) / "pot.yace").touch()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_real_logic_success(self):
        """Test the real orchestrator logic with fakes, no mocking of orchestrator itself."""

        state_manager = StateManager(Path(self.test_dir))
        # Configure fake explorer to fail on 2nd call to break the loop naturally
        explorer = FakeExplorer(max_calls=1)
        al_service = FakeALService()

        # We need to change cwd to test dir so relative paths in config work
        cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            orch = ActiveLearningOrchestrator(
                config=self.config,
                al_service=al_service,
                explorer=explorer,
                state_manager=state_manager
            )
            # Enable AL so loop continues until explorer fails
            orch.al_config_path = Path(self.test_dir) / "al_config.yaml"
            orch.al_config_path.touch()

            orch.run()

            # Verify state file created
            state_file = Path(self.test_dir) / "orchestrator_state.json"
            self.assertTrue(state_file.exists())

            with open(state_file, 'r') as f:
                data = json.load(f)

            # Iteration should be 2 because loop breaks at start of 2nd iteration (or end of 1st)
            # FakeExplorer logic: call_count increases.
            # 1st call: call_count=1 <= max_calls=1. Returns SUCCESS.
            # 2nd call: call_count=2 > max_calls. Returns FAILED.
            # Orchestrator loop:
            # Iter 1: explore -> SUCCESS. updates state.
            # Iter 2: explore -> FAILED. break.
            # State is saved at start of iteration.
            # So state should have iteration 2.

            self.assertEqual(data["iteration"], 2)

            # Check structure is a string path and points to final_structure.xyz
            self.assertTrue(isinstance(data["current_structure"], str))
            self.assertTrue(data["current_structure"].endswith("final_structure.xyz"))

            # Verify the file actually exists on disk
            self.assertTrue(Path(data["current_structure"]).exists())

        finally:
            os.chdir(cwd)

    def test_run_real_logic_fails_gracefully_on_type_error(self):
        """Verify Pydantic validation error if we try to save bad type."""
        state_manager = StateManager(Path(self.test_dir))

        # Try to save an Atoms object (should fail)
        bad_state = {
            "iteration": 1,
            "current_structure": Atoms('H') # Invalid type, expects str
        }

        with self.assertRaises(ValidationError):
            state_manager.save(bad_state)

    def test_atomic_write_integrity(self):
        """Verify atomic write pattern prevents corruption."""
        state_manager = StateManager(Path(self.test_dir))
        state = {"iteration": 1}

        with unittest.mock.patch('os.replace') as mock_replace:
             state_manager.save(state)
             # Check that replace was called
             mock_replace.assert_called_once()
             # Check that tmp file exists (since replace was mocked out and didn't move it)
             tmp_file = state_manager.state_file.with_suffix(".tmp")
             self.assertTrue(tmp_file.exists())
