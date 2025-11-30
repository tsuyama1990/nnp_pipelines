"""Unit tests for Baseline Optimizer."""

import numpy as np
import pytest
from ase import Atoms
from workers.al_md_kmc_worker.src.services.baseline_optimizer import BaselineOptimizer
from shared.core.config import LJParams
from shared.utils.fast_math import lj_energy_forces_jit

def test_baseline_optimizer_recovery():
    """Test that optimizer can recover known LJ parameters from synthetic data."""

    # 1. Setup Synthetic Ground Truth
    elements = ["Ar"]
    true_eps = 1.0
    true_sig = 3.0
    true_e0 = -5.0
    cutoff = 10.0

    # Generate structures
    structures = []
    # Dimer at various distances
    dists = np.linspace(2.8, 4.0, 5)
    box = np.array([20.0, 20.0, 20.0])

    eps_arr = np.array([true_eps])
    sig_arr = np.array([true_sig])

    for d in dists:
        pos = np.array([
            [0.0, 0.0, 0.0],
            [d, 0.0, 0.0]
        ])
        types = np.array([0, 0])

        # Calculate Truth (using our kernel to ensure consistency)
        e_lj, f_lj = lj_energy_forces_jit(pos, box, types, eps_arr, sig_arr, cutoff)

        # Add E0
        e_total = e_lj + 2 * true_e0

        atoms = Atoms('Ar2', positions=pos, cell=box, pbc=True)
        atoms.info['energy_dft_raw'] = e_total
        atoms.arrays['forces_dft_raw'] = f_lj # Forces unaffected by E0
        structures.append(atoms)

    # 2. Init Optimizer with perturbed guess
    initial_lj = LJParams(epsilon=0.8, sigma=2.8, cutoff=cutoff, shift_energy=True)
    optimizer = BaselineOptimizer(elements, initial_lj)

    # 3. Optimize
    result = optimizer.optimize(structures, force_weight=0.1)

    # 4. Assert Recovery
    assert np.isclose(result['epsilon']['Ar'], true_eps, atol=0.05)
    assert np.isclose(result['sigma']['Ar'], true_sig, atol=0.05)
    assert np.isclose(result['e0']['Ar'], true_e0, atol=0.05)
