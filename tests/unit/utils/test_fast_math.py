"""Unit tests for fast_math utilities."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from shared.utils.fast_math import lj_energy_forces_jit

def test_lj_energy_forces_jit_accuracy():
    """Verify JIT LJ kernel against ASE LennardJones."""

    # Setup System
    # 2 atoms, 1 species for simplicity first, then 2 species
    d = 3.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0]
    ])
    box = np.array([10.0, 10.0, 10.0])

    # Parameters
    epsilon = 1.0
    sigma = 2.5
    cutoff = 6.0

    # JIT Calculation
    atom_types = np.array([0, 0])
    epsilon_arr = np.array([epsilon])
    sigma_arr = np.array([sigma])

    e_jit, f_jit = lj_energy_forces_jit(
        positions, box, atom_types, epsilon_arr, sigma_arr, cutoff
    )

    # ASE Calculation
    atoms = Atoms('Ar2', positions=positions, cell=box, pbc=True)
    calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=cutoff)
    atoms.calc = calc

    e_ase = atoms.get_potential_energy()
    f_ase = atoms.get_forces()

    # Assert
    assert np.isclose(e_jit, e_ase, atol=1e-5)
    assert np.allclose(f_jit, f_ase, atol=1e-5)

def test_lj_energy_forces_jit_mixing():
    """Verify mixing rules."""

    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    box = np.array([10.0, 10.0, 10.0])

    # Species 0 and 1
    atom_types = np.array([0, 1])
    epsilon_arr = np.array([1.0, 2.0]) # sqrt(2) mixing
    sigma_arr = np.array([2.0, 3.0])   # 2.5 mixing
    cutoff = 10.0

    e_jit, f_jit = lj_energy_forces_jit(
        positions, box, atom_types, epsilon_arr, sigma_arr, cutoff
    )

    # Manual Check
    sig_ij = 2.5
    eps_ij = np.sqrt(2.0)
    r = 3.0
    sr = sig_ij / r
    e_raw = 4 * eps_ij * (sr**12 - sr**6)

    # Apply Shift manually
    sr_cut = sig_ij / cutoff
    e_shift = 4 * eps_ij * (sr_cut**12 - sr_cut**6)

    e_expected = e_raw - e_shift

    assert np.isclose(e_jit, e_expected, atol=1e-5)
