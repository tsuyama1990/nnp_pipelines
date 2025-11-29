
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from ase import Atoms

sys.path.append(os.getcwd())

from shared.autostructure.preopt import PreOptimizer

def test_preopt_uses_mace():
    # Mock mace.calculators.mace_mp
    # create=True ensures we can patch it even if it wasn't imported successfully
    with patch("shared.autostructure.preopt.mace_mp", create=True) as mock_mace:
        # Mock the calculator instance returned by mace_mp
        mock_calc = MagicMock()
        mock_mace.return_value = mock_calc

        with patch("shared.autostructure.preopt.HAS_MACE", True):
            # Mock BFGS
            with patch("shared.autostructure.preopt.BFGS") as mock_bfgs:
                mock_bfgs_instance = MagicMock()
                mock_bfgs.return_value = mock_bfgs_instance

                optimizer = PreOptimizer(lj_params={}, fmax=0.05, steps=10)
                atoms = Atoms("Cu", positions=[[0,0,0]])

                optimizer.run_pre_optimization(atoms)

                # Verify mace_mp called
                mock_mace.assert_called()
                # Default is medium
                mock_mace.assert_any_call(model="medium", device="cpu", default_dtype="float64")

                # Verify BFGS called
                mock_bfgs.assert_called()
                mock_bfgs_instance.run.assert_called_with(fmax=0.05, steps=10)

def test_preopt_fallback_to_small():
    with patch("shared.autostructure.preopt.mace_mp", create=True) as mock_mace:
        # First call raises Exception
        # Second call returns calculator
        mock_calc = MagicMock()
        mock_mace.side_effect = [Exception("Medium failed"), mock_calc]

        with patch("shared.autostructure.preopt.HAS_MACE", True):
             with patch("shared.autostructure.preopt.BFGS"):
                optimizer = PreOptimizer(lj_params={})
                atoms = Atoms("Cu", positions=[[0,0,0]])

                optimizer.run_pre_optimization(atoms)

                assert mock_mace.call_count == 2
                mock_mace.assert_any_call(model="medium", device="cpu", default_dtype="float64")
                mock_mace.assert_any_call(model="small", device="cpu", default_dtype="float64")
