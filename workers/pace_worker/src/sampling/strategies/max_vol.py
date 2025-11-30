"""MaxVol Sampler Strategy."""

import logging
import numpy as np
from typing import List, Tuple
from ase import Atoms

from shared.core.interfaces import Sampler
from src.sampling.strategies.ace_sampler import ACESampler

try:
    import pyace
except ImportError:
    pyace = None

logger = logging.getLogger(__name__)

class LocalStructureDescriptor:
    """Computes descriptors for specific atomic environments."""

    def __init__(self, potential_path: str):
        self.ace_sampler = ACESampler(potential_path)

    def compute(self, atoms: Atoms, indices: List[int]) -> np.ndarray:
        """Compute descriptors only for specified atoms."""
        full_desc = self.ace_sampler.compute_descriptors(atoms) # (N_atoms, N_feat)

        # Select rows
        if not indices:
            return np.mean(full_desc, axis=0) # Fallback to global mean

        valid_indices = [i for i in indices if i < len(atoms)]
        if not valid_indices:
             return np.mean(full_desc, axis=0)

        local_desc = full_desc[valid_indices]
        # Return mean of local environment descriptors for the "Cluster" representation
        return np.mean(local_desc, axis=0)

class MaxVolSampler(Sampler):
    """Selects high-value structures using MaxVol algorithm via pyace/pacemaker API.

    This sampler computes ACE descriptors for a list of structures and uses the
    MaxVol algorithm to select the most informative ones.
    Supports Epic 6: Local MaxVol Sampling on clusters.
    """

    def __init__(self):
        """Initialize MaxVolSampler."""
        pass

    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        """Select structures using MaxVol on ACE descriptors.

        Args:
            **kwargs: Must contain:
                - structures (List[Atoms]): List of candidate structures.
                - potential_path (str): Path to the potential/basis set (YAML or YACE).
                - n_clusters (int): Number of structures to select.
                - atom_indices (List[int], optional): Indices of atoms defining the local cluster/active region.

        Returns:
            List[Tuple[Atoms, int]]: List of (structure, max_gamma_atom_index).
        """
        structures = kwargs.get('structures')
        potential_path = kwargs.get('potential_path')
        n_selection = kwargs.get('n_clusters')
        atom_indices = kwargs.get('atom_indices')

        if not structures:
            logger.warning("No structures provided to MaxVolSampler.")
            return []

        if not potential_path:
            raise ValueError("potential_path is required for MaxVolSampler.")

        if not n_selection:
             raise ValueError("n_clusters is required for MaxVolSampler.")

        # 1. Compute Descriptors
        logger.info(f"Computing descriptors for {len(structures)} structures...")

        # Use generic sampler for full descriptors, but we can wrap it logic locally
        # ACESampler is in src.sampling.strategies.ace_sampler
        # We need to instantiate it once

        desc_computer = LocalStructureDescriptor(potential_path)

        descriptors_list = []
        valid_indices = []

        for i, atoms in enumerate(structures):
            try:
                # Epic 6: Local MaxVol Sampling
                # If atom_indices provided, compute descriptor for that subset
                desc_vec = desc_computer.compute(atoms, atom_indices)

                descriptors_list.append(desc_vec)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(f"Failed descriptor calc for structure {i}: {e}")

        if not descriptors_list:
            return []

        X = np.array(descriptors_list)

        # 2. MaxVol Selection
        selected_local_indices = []

        if pyace and hasattr(pyace, "SelectMaxVol"):
             selected_local_indices = pyace.SelectMaxVol(X, n_selection)
        else:
             # Fallback: QR pivoting
             from scipy.linalg import qr

             limit = min(n_selection, X.shape[1], X.shape[0])
             if limit < n_selection:
                 logger.warning(f"Requested {n_selection} but capped at {limit}.")

             # X shape: (n_structures, n_features)
             # MaxVol selects rows (structures) that span volume
             _, _, P = qr(X.T, pivoting=True)
             selected_local_indices = P[:limit]

        # 3. Identify Max Gamma Atom in selected structures
        # We need the calculator for this.
        # ACESampler has .calculator
        calc = desc_computer.ace_sampler.calculator

        results = []
        for idx in selected_local_indices:
            global_idx = valid_indices[idx]
            atoms = structures[global_idx]

            try:
                gamma_arr = calc.get_property("gamma", atoms)

                # If local sampling, we should probably pick max gamma within the cluster too?
                # But the return value is center_id for generation.
                # If we have atom_indices, we should pick the one with max gamma among them.

                if atom_indices:
                    valid_subset = [k for k in atom_indices if k < len(gamma_arr)]
                    if valid_subset:
                        sub_gammas = gamma_arr[valid_subset]
                        local_argmax = np.argmax(sub_gammas)
                        max_gamma_idx = valid_subset[local_argmax]
                    else:
                         max_gamma_idx = int(np.argmax(gamma_arr))
                else:
                    max_gamma_idx = int(np.argmax(gamma_arr)) if np.ndim(gamma_arr) > 0 else 0

                results.append((atoms, max_gamma_idx))
            except Exception as e:
                logger.warning(f"Failed to find max gamma for selected structure: {e}")
                results.append((atoms, 0))

        return results
