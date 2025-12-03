import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from ase import Atoms
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from workers.al_md_kmc_worker.src.services.md_service import MDService
from shared.core.config import Config
from shared.core.enums import SimulationState

class TestMDService(unittest.TestCase):
    def setUp(self):
        self.mock_md_engine = MagicMock()
        self.mock_config = MagicMock(spec=Config)
        self.mock_config.md_params = MagicMock()
        self.mock_config.md_params.temperature = 300
        self.mock_config.md_params.pressure = 1.0
        self.mock_config.md_params.n_md_walkers = 2
        self.mock_config.md_params.n_steps = 100
        self.mock_config.exploration_schedule = []
        self.mock_config.al_params = MagicMock()
        self.mock_config.al_params.gamma_threshold = 0.5

        self.service = MDService(self.mock_md_engine, self.mock_config)

    @patch('workers.al_md_kmc_worker.src.services.md_service.ParallelExecutor')
    def test_run_walkers_parallel_delegation(self, MockExecutor):
        # Mock executor to return a successful result immediately
        mock_executor_instance = MockExecutor.return_value

        # We need to simulate the return value of submit_tasks
        # It returns a list of (state, dump_path)
        mock_executor_instance.submit_tasks.return_value = [
            (SimulationState.COMPLETED, Path("dummy_dump.lammpstrj")),
            (SimulationState.COMPLETED, Path("dummy_dump.lammpstrj"))
        ]

        # Mock reading atoms
        with patch('workers.al_md_kmc_worker.src.services.md_service.read') as mock_read:
            atoms = Atoms('H2')
            atoms.arrays['f_2'] = 0.1 # low gamma
            mock_read.return_value = [atoms]

            # Re-init service to use the mocked executor if it was instantiated in __init__
            # But MDService instantiates ParallelExecutor in __init__, so we need to patch before init
            # Actually we patched the class, so the instance created in setUp is using the real one if we didn't patch class before setup
            # Let's recreate service here
            self.service = MDService(self.mock_md_engine, self.mock_config)

            success, uncertain, failed = self.service.run_walkers(
                iteration=1,
                potential_path=Path("pot.yace"),
                input_structure_path="struct.xyz",
                is_restart=False
            )

            self.assertTrue(success)
            self.assertFalse(failed)
            self.assertEqual(len(uncertain), 0)

            # Verify executor was used
            self.service.executor.submit_tasks.assert_called()

if __name__ == "__main__":
    unittest.main()
