import unittest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
from workers.pace_worker.src.sampling.strategies.max_vol import MaxVolSampler, LocalStructureDescriptor

class TestMaxVolSampler(unittest.TestCase):

    def setUp(self):
        self.sampler = MaxVolSampler()

    @patch('workers.pace_worker.src.sampling.strategies.max_vol.LocalStructureDescriptor')
    def test_sample_outlier_detection(self, mock_descriptor_cls):
        # Goal: Synthetic data with 10 bulk and 1 outlier. Ensure outlier is selected.

        # 1. Setup mock descriptor
        mock_descriptor_instance = MagicMock()
        mock_descriptor_cls.return_value = mock_descriptor_instance

        # We simulate 10 structures.
        # 9 are "normal" (vectors close to [1, 0, ...])
        # 1 is "outlier" (vector close to [0, 1, ...])

        n_features = 10
        n_structures = 10
        structures = [Atoms('H') for _ in range(n_structures)]

        def compute_side_effect(atoms, indices):
            # If atoms is the last one (index 9), return outlier
            # How to know index? We can rely on order of calls or check atoms identity?
            # Or just check if atoms is structures[-1]
            if atoms == structures[-1]:
                # Outlier: Orthogonal to others
                return np.array([[0.0] * (n_features-1) + [10.0]]) # [0,0,...,10]
            else:
                # Normal: Random small noise around [1,0,...,0]
                base = np.zeros((1, n_features))
                base[0, 0] = 1.0
                noise = np.random.normal(0, 0.01, (1, n_features))
                return base + noise

        mock_descriptor_instance.compute.side_effect = compute_side_effect

        # Mock calculator for Gamma check (step 3)
        mock_calc = MagicMock()
        mock_descriptor_instance.ace_sampler.calculator = mock_calc
        mock_calc.get_property.return_value = np.array([0.5]) # Dummy gamma

        # 2. Run Sample
        # We request n_clusters=2. Expect to pick one normal and the outlier.
        results = self.sampler.sample(
            structures=structures,
            potential_path="dummy",
            n_clusters=2,
            atom_indices=None
        )

        # 3. Verify
        # We expect 2 selected structures
        self.assertEqual(len(results), 2)

        # One of them MUST be the outlier (structures[-1])
        selected_atoms = [res[0] for res in results]
        self.assertIn(structures[-1], selected_atoms)

    @patch('workers.pace_worker.src.sampling.strategies.max_vol.LocalStructureDescriptor')
    def test_max_rows_cap(self, mock_descriptor_cls):
        # Goal: Verify subsampling happens if rows exceed max_rows

        mock_descriptor_instance = MagicMock()
        mock_descriptor_cls.return_value = mock_descriptor_instance

        # Return large descriptor array
        # 2 structures, each generating 30,000 rows. Total 60,000. Cap 50,000.
        def compute_side_effect(atoms, indices):
            return np.zeros((30000, 5))

        mock_descriptor_instance.compute.side_effect = compute_side_effect

        structures = [Atoms('H'), Atoms('H')]

        # Mock calculator
        mock_descriptor_instance.ace_sampler.calculator = MagicMock()
        mock_descriptor_instance.ace_sampler.calculator.get_property.return_value = np.array([0])

        # Spy on np.random.choice to verify it was called
        with patch('numpy.random.choice', wraps=np.random.choice) as mock_choice:
            self.sampler.sample(
                structures=structures,
                potential_path="dummy",
                n_clusters=1,
                max_rows=50000
            )

            # Should have called choice because 60000 > 50000
            self.assertTrue(mock_choice.called)
            # Verify called with size=50000
            args, _ = mock_choice.call_args
            self.assertEqual(args[1], 50000)

if __name__ == '__main__':
    unittest.main()
