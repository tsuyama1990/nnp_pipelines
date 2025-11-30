import unittest
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from shared.engines.kmc import _run_local_search
from shared.core.config import KMCParams, ALParams, LJParams
from shared.core.enums import KMCStatus
from unittest.mock import MagicMock, patch

class TestKMCPhysics(unittest.TestCase):
    def setUp(self):
        self.kmc_params = KMCParams(
            temperature=300,
            prefactor=1e12,
            search_radius=0.5,
            check_interval=10,
            dimer_fmax=0.05,
            box_size=10.0,
            n_workers=1
        )
        # ALParams requires all fields now
        self.al_params = ALParams(
            gamma_threshold=0.1,
            n_clusters=5,
            r_core=3.0,
            box_size=10.0,
            initial_potential="dummy.yace",
            potential_yaml_path="dummy.yaml"
        )
        self.lj_params = LJParams(epsilon=1.0, sigma=1.0, cutoff=5.0)
        self.e0_dict = {}

        # Create a simple cluster (7 atoms LJ)
        self.cluster = Atoms('Ar7', positions=[
            [0, 0, 0],
            [1.1, 0, 0],
            [0, 1.1, 0],
            [0, 0, 1.1],
            [-1.1, 0, 0],
            [0, -1.1, 0],
            [0, 0, -1.1]
        ])
        self.cluster.calc = LennardJones()
        self.active_indices = list(range(7))

    @patch('shared.engines.kmc._setup_calculator')
    def test_run_local_search_retry_logic(self, mock_setup):
        # We mock _setup_calculator to just attach a simple LJ calculator
        def side_effect(atoms, *args, **kwargs):
            atoms.calc = LennardJones(epsilon=1.0, sigma=2.5) # Strong interaction

        mock_setup.side_effect = side_effect

        # We will mock MinModeAtoms to count initializations (retries)
        with patch('shared.engines.kmc.MinModeAtoms') as mock_dimer_cls, \
             patch('shared.engines.kmc.FIRE') as mock_fire_cls:

            # Setup Dimer behavior
            mock_dimer = MagicMock()
            mock_dimer_cls.return_value = mock_dimer

            # Scenario: First attempt -> Barrier 0 (collapse). Second attempt -> Barrier 0.05 (success)
            mock_dimer.get_potential_energy.side_effect = [0.0, 0.0, 0.5, 0.5] # saddle, initial (diff=0); saddle, initial (diff=0.5 but need to handle sequence correctly)

            # Wait, _run_local_search calls:
            # 1. e_saddle = dimer_atoms.get_potential_energy()
            # 2. e_initial = cluster.get_potential_energy()

            # We can control e_saddle via mock_dimer.
            # e_initial comes from cluster which has LJ attached? No, in _run_local_search 'cluster' is passed.
            # But inside loop, 'cluster.get_potential_energy()' is called.
            # Since cluster has LJ attached in setup (if not mocked out there too? No, mock_setup attaches to 'atoms').
            # 'cluster' passed to _run_local_search needs a calculator.
            # The function starts with: `_setup_calculator(cluster, ...)`

            # So cluster will have LJ.
            # Let's say cluster E = -10.0

            # To simulate barrier=0 (fail): set dimer PE = -10.0
            # To simulate barrier=0.5 (success): set dimer PE = -9.5

            # We need to ensure we return appropriate values for successive calls.
            # 1st attempt: dimer.get_PE returns -10.0
            # 2nd attempt: dimer.get_PE returns -9.5
            mock_dimer.get_potential_energy.side_effect = [-10.0, -9.5]

            # And we need cluster.get_potential_energy() to return -10.0 always.
            # We can mock cluster.get_potential_energy or let real LJ run.
            # Real LJ is fine if we set positions s.t. E is constant?
            # Actually easier to mock return value of `cluster.get_potential_energy`?
            # But `cluster` is passed in.

            # Let's just mock `cluster` object entirely?
            # Or assume real LJ values.
            # Let's mock the `get_potential_energy` on the cluster itself inside the test?
            # But `_run_local_search` calls `_setup_calculator` on `cluster`.
            # Our mock_setup adds LennardJones.

            # Let's rely on the mock_dimer returning a barrier relative to whatever cluster E is.
            # But we don't know cluster E exactly.

            # Better approach: Mock the barrier calculation result or logic?
            # No, logic is `barrier = e_saddle - e_initial`.

            # Let's force e_saddle to be `cluster.get_potential_energy() + delta`.
            # But we need to know cluster.get_potential_energy().

            # Let's pre-calc cluster energy.
            self.cluster.calc = LennardJones(epsilon=1.0, sigma=2.5)
            base_e = self.cluster.get_potential_energy()

            mock_dimer.get_potential_energy.side_effect = [base_e + 0.0001, base_e + 0.5] # 1st: ~0 barrier, 2nd: 0.5 barrier

            # Mock FIRE to converge immediately
            mock_opt = MagicMock()
            mock_fire_cls.return_value = mock_opt
            mock_opt.converged.return_value = True

            # Run
            res = _run_local_search(
                self.cluster,
                "dummy_path",
                self.lj_params,
                self.e0_dict,
                self.kmc_params,
                self.al_params,
                self.active_indices,
                seed=42,
                delta_learning_mode=False
            )

            # We expect success on 2nd attempt
            # So MinModeAtoms should have been instantiated 2 times
            self.assertEqual(mock_dimer_cls.call_count, 2)

            # Result should be valid event (tuple)
            self.assertTrue(isinstance(res, tuple))
            barrier = res[0]
            self.assertAlmostEqual(barrier, 0.5)

    def test_random_initialization(self):
        # This test ensures that we don't crash and return *some* result
        with patch('shared.engines.kmc._setup_calculator') as mock_setup:
             mock_setup.side_effect = lambda atoms, *args, **kwargs: setattr(atoms, 'calc', LennardJones())

             res = _run_local_search(
                self.cluster,
                "dummy_path",
                self.lj_params,
                self.e0_dict,
                self.kmc_params,
                self.al_params,
                self.active_indices,
                seed=42,
                delta_learning_mode=False
            )
             # Should be KMCResult or Tuple
             # With real LJ and random displacement, it might find something or not.
             # Just checking no crash.
             self.assertTrue(res is not None)

if __name__ == '__main__':
    unittest.main()
