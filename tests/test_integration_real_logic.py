import unittest
import tempfile
import shutil
import json
from pathlib import Path
from ase import Atoms
from ase.io import write

# Ensure shared modules are available
import sys
import os
sys.path.append(os.getcwd())

from shared.core.config import Config, MDParams, ALParams
from workers.al_md_kmc_worker.src.workflows.active_learning_loop import ActiveLearningOrchestrator
from workers.al_md_kmc_worker.src.state_manager import StateManager
from tests.fakes import FakeExplorer, FakeALService
from pydantic import ValidationError

class TestIntegrationRealLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.work_dir = Path(self.test_dir) / "work" / "07_active_learning"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Setup config
        self.config = Config(
            md_params=MDParams(
                timestep=1.0, temperature=300, pressure=1.0, n_steps=100,
                elements=["Al"], initial_structure="start.xyz", masses={"Al": 26.98}
            ),
            al_params=ALParams(
                gamma_threshold=0.1, n_clusters=2, r_core=3.0, box_size=10.0,
                initial_potential="pot.yace", potential_yaml_path="pot.yaml",
                initial_dataset_path=None, initial_active_set_path=None
            )
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
        explorer = FakeExplorer()
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

            # Hack: break the infinite loop after 1 iteration by side-effecting the explorer
            # But the loop in `run` doesn't check for max iterations.
            # It breaks if exploration failed or AL inactive and exploration finished.
            # FakeExplorer returns COMPLETED.
            # If AL is inactive (al_config_path is None), loop breaks.

            orch.run()

            # Verify state file created
            state_file = Path(self.test_dir) / "orchestrator_state.json"
            self.assertTrue(state_file.exists())

            with open(state_file, 'r') as f:
                data = json.load(f)

            self.assertEqual(data["iteration"], 1)
            # Check structure is a path string
            self.assertTrue(isinstance(data["current_structure"], str))
            self.assertTrue(data["current_structure"].endswith("final_structure.xyz"))

        finally:
            os.chdir(cwd)

    def test_run_real_logic_fails_gracefully_on_type_error(self):
        """Verify Pydantic validation error if we try to save bad type."""
        state_manager = StateManager(Path(self.test_dir))

        # Try to save an Atoms object (should fail)
        bad_state = {
            "iteration": 1,
            "current_structure": Atoms('H') # Invalid type
        }

        with self.assertRaises(ValidationError):
            state_manager.save(bad_state)

    def test_atomic_write_integrity(self):
        """Verify atomic write pattern prevents corruption."""
        state_manager = StateManager(Path(self.test_dir))
        state = {"iteration": 1}

        # Normal save
        state_manager.save(state)
        self.assertTrue(state_manager.state_file.exists())

        # Simulate crash during write (mock open/write to fail?)
        # Hard to simulate crash exactly, but we can check if .tmp file exists during write?
        # Or verify .tmp is used.
        # Use a patch to verify atomic steps

        with unittest.mock.patch('os.replace') as mock_replace:
             state_manager.save(state)
             # Check that replace was called
             mock_replace.assert_called_once()
             # Check that tmp file exists (since replace was mocked out and didn't move it)
             tmp_file = state_manager.state_file.with_suffix(".tmp")
             self.assertTrue(tmp_file.exists())
