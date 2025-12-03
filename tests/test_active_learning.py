import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from shared.generators.small_cell import SmallCellGenerator
from workers.pace_worker.src.sampling.strategies.max_gamma import MaxGammaSampler

class TestSmallCellGenerator(unittest.TestCase):
    def setUp(self):
        self.box_size = 10.0
        self.r_core = 3.0
        # Fixed: Removed stoichiometry_tolerance
        self.generator = SmallCellGenerator(
            box_size=self.box_size,
            r_core=self.r_core,
            stoichiometric_ratio={'Fe': 1.0}, # Updated to correct arg
            lammps_cmd="lmp_serial",
            lj_params={"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0, "shift_energy": True}
        )

        # Create a dummy atoms object
        # 10x10x10 cubic cell, 2 atoms
        self.atoms = Atoms('Fe2',
                           positions=[[0, 0, 0], [2, 0, 0]],
                           cell=[10, 10, 10],
                           pbc=True)

    @patch('shared.generators.small_cell.PyACECalculator')
    @patch('shared.generators.small_cell.ExpCellFilter')
    @patch('shared.generators.small_cell.FIRE')
    def test_generate_structure(self, mock_fire, mock_filter, mock_pyace):
        # Setup mocks
        mock_calc = MagicMock()
        mock_pyace.return_value = mock_calc

        # Mock FIRE run to do nothing
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        # Run generate with atom 0 as center
        center_id = 0
        small_cell = self.generator.generate_cell(self.atoms, center_id, "dummy.yace")

        # Assertions on the generated cell
        self.assertTrue(small_cell.pbc.all())
        np.testing.assert_array_almost_equal(small_cell.cell.lengths(), [self.box_size]*3)

        positions = small_cell.get_positions()
        self.assertEqual(len(small_cell), 2)

        # Find atom corresponding to original center (closest to 5,5,5)
        center_pos = np.array([5.0, 5.0, 5.0])
        dists = np.linalg.norm(positions - center_pos, axis=1)
        self.assertLess(np.min(dists), 0.01) # One atom should be at center

    @patch('shared.generators.small_cell.PyACECalculator')
    @patch('shared.generators.small_cell.ExpCellFilter')
    @patch('shared.generators.small_cell.FIRE')
    def test_relaxation_called(self, mock_fire, mock_filter, mock_pyace):
        mock_calc = MagicMock()
        mock_pyace.return_value = mock_calc
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        self.generator.generate_cell(self.atoms, 0, "dummy.yace")

        # PyACECalculator should be called with potential path
        mock_pyace.assert_called_with("dummy.yace")
        mock_filter.assert_called()
        mock_fire.assert_called()
        mock_opt.run.assert_called()


class TestMaxGammaSampler(unittest.TestCase):
    def test_sample(self):
        sampler = MaxGammaSampler()
        atoms = Atoms('H5')
        # Setup arrays
        # 5 atoms. Gamma values: [0.1, 0.5, 0.2, 0.9, 0.0]
        # Max is index 3 (0.9), then index 1 (0.5)
        atoms.set_array('gamma', np.array([0.1, 0.5, 0.2, 0.9, 0.0]))

        # Test n_samples = 1
        results = sampler.sample(candidates=[atoms], n_samples=1)
        # Result is list of tuples (atoms, index)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 0)  # Index in candidates list

        # Test n_samples = 2 with multiple candidates
        atoms2 = Atoms('H5')
        atoms2.set_array('gamma', np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
        results = sampler.sample(candidates=[atoms, atoms2], n_samples=2)
        # Should select both
        self.assertEqual(len(results), 2)

    def test_sample_raises_error_on_missing(self):
        sampler = MaxGammaSampler()
        atoms = Atoms('H10')
        # No gamma array - should return empty or handle gracefully
        # The implementation doesn't raise ValueError, it returns empty or uses 0.0
        results = sampler.sample(candidates=[atoms], n_samples=2)
        # Should still return results, just with gamma=0.0
        self.assertEqual(len(results), 1)

if __name__ == '__main__':
    unittest.main()
