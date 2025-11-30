"""Baseline Potential Auto-Optimization Service."""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from scipy.optimize import minimize
from ase import Atoms

from shared.core.config import Config, LJParams
from shared.utils.fast_math import lj_energy_forces_jit

logger = logging.getLogger(__name__)

class BaselineOptimizer:
    """Optimizes LJ parameters (epsilon, sigma, E_offset) against DFT data."""

    def __init__(self, elements: List[str], config_lj: LJParams):
        """
        Args:
            elements: List of element symbols (must be sorted/consistent with index map).
            config_lj: Initial LJ parameters from config.
        """
        self.elements = elements
        self.element_map = {el: i for i, el in enumerate(elements)}
        self.num_species = len(elements)
        self.initial_lj = config_lj

    def _prepare_data(self, labeled_structures: List[Atoms]) -> Tuple[List[Dict[str, Any]], float]:
        """Pre-process structures into numpy arrays for JIT."""
        data = []

        # We need a consistent weighting.
        # Cost = sum((E_dft - E_model)^2) + w_f * sum(|F_dft - F_model|^2)
        # We need to extract E_DFT_RAW (unshifted) because we are fitting the baseline.
        # Target = E_DFT_RAW.
        # Model = E_LJ_shifted + sum(E_offset).

        for atoms in labeled_structures:
            # Check if we have raw dft data
            e_dft = atoms.info.get('energy_dft_raw', atoms.info.get('energy', None))
            f_dft = atoms.arrays.get('forces_dft_raw', atoms.arrays.get('forces', None))

            if e_dft is None or f_dft is None:
                continue

            pos = atoms.get_positions()
            cell = atoms.get_cell()
            if not np.allclose(cell, np.diag(cell.diagonal()), atol=1e-5):
                # Only Orthorhombic supported by JIT currently
                # If non-ortho, skip or warn
                # For now skip
                continue

            box = cell.diagonal().copy()

            symbols = atoms.get_chemical_symbols()
            types = np.array([self.element_map[s] for s in symbols], dtype=np.int32)

            # Count species for E_offset term
            counts = np.zeros(self.num_species, dtype=np.float64)
            for t in types:
                counts[t] += 1.0

            data.append({
                'pos': pos,
                'box': box,
                'types': types,
                'counts': counts,
                'e_target': e_dft,
                'f_target': f_dft,
                'natoms': len(atoms)
            })

        return data

    def optimize(self, labeled_structures: List[Atoms], force_weight: float = 1.0) -> Dict[str, Any]:
        """Run the optimization."""
        data = self._prepare_data(labeled_structures)

        if not data:
            logger.warning("No valid data for Baseline Optimization (Orthorhombic cells required).")
            return {}

        # Optimization Variables:
        # Per species: epsilon, sigma, E_offset
        # Total variables = 3 * num_species

        # Initial Guess
        x0 = []
        bounds = []

        # We use the scalar initial guess from config if species-specific not available
        # But if we want to optimize per-species, we need per-species guess.
        # For now, duplicate the scalar.

        eps_init = self.initial_lj.epsilon
        sig_init = self.initial_lj.sigma

        for i in range(self.num_species):
            # Epsilon
            x0.append(eps_init)
            bounds.append((0.001, 5.0)) # eV

            # Sigma
            x0.append(sig_init)
            bounds.append((0.5, 5.0)) # Angstrom

            # E_offset (Reference Energy per atom)
            # Guess: Average E per atom from data?
            # Or 0.0 if unknown. Usually E_dft is large negative.
            # Let's verify what E_offset represents.
            # E_model = E_LJ + sum(N_i * E_offset_i)
            # So E_offset is basically the isolated atom energy (or chemical potential).
            # We initialize it to the mean energy per atom of the first structure?
            e_per_atom = data[0]['e_target'] / data[0]['natoms']
            x0.append(e_per_atom)
            bounds.append((None, None))

        x0 = np.array(x0)

        # Cost Function
        cutoff = self.initial_lj.cutoff

        def loss_fn(x):
            # Unpack x
            # x structure: [eps0, sig0, off0, eps1, sig1, off1, ...]
            eps_arr = np.zeros(self.num_species)
            sig_arr = np.zeros(self.num_species)
            off_arr = np.zeros(self.num_species)

            for i in range(self.num_species):
                eps_arr[i] = x[3*i]
                sig_arr[i] = x[3*i + 1]
                off_arr[i] = x[3*i + 2]

            total_loss = 0.0

            for d in data:
                # 1. Compute LJ
                e_lj, f_lj = lj_energy_forces_jit(
                    d['pos'], d['box'], d['types'], eps_arr, sig_arr, cutoff
                )

                # 2. Add Offsets
                # E_model = E_LJ + sum(counts * offsets)
                e_offset_sum = np.dot(d['counts'], off_arr)
                e_model = e_lj + e_offset_sum

                # 3. Residuals
                e_res = (d['e_target'] - e_model)
                f_res = (d['f_target'] - f_lj) # (N, 3)

                # 4. Loss
                # Energy term: squared error per atom? Or total?
                # Usually per-atom to normalize across structures.
                e_loss = (e_res / d['natoms'])**2

                # Force term: mean squared error per component?
                f_loss = np.mean(f_res**2)

                total_loss += e_loss + force_weight * f_loss

            return total_loss / len(data)

        logger.info(f"Starting Baseline Optimization on {len(data)} structures...")
        res = minimize(loss_fn, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})

        logger.info(f"Optimization finished. Success: {res.success}, Loss: {res.fun}")

        # Parse Result
        optimized_params = {}
        final_x = res.x

        # Construct dictionaries
        eps_dict = {}
        sig_dict = {}
        e0_dict = {}

        for i, el in enumerate(self.elements):
            eps_dict[el] = final_x[3*i]
            sig_dict[el] = final_x[3*i + 1]
            e0_dict[el] = final_x[3*i + 2]

        return {
            "epsilon": eps_dict,
            "sigma": sig_dict,
            "e0": e0_dict,
            "loss": res.fun
        }
