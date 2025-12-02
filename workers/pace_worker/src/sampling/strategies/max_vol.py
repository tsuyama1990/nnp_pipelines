"""MaxVol Sampler Strategy."""

import logging
import numpy as np
from typing import List, Tuple
from ase import Atoms

from shared.core.interfaces import Sampler
from .ace_sampler import ACESampler

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
        """Compute descriptors only for specified atoms.

        Returns:
             np.ndarray: Shape (N_subset, N_features).
        """
        full_desc = self.ace_sampler.compute_descriptors(atoms) # (N_atoms, N_feat)

        # Select rows
        if not indices:
            # Epic 2: Return all atoms if no indices specified, not mean
            return full_desc

        valid_indices = [i for i in indices if i < len(atoms)]
        if not valid_indices:
             # Fallback to all if indices invalid? Or empty?
             # If filtered result is empty, return empty array?
             # For robustness, let's return all.
             return full_desc

        local_desc = full_desc[valid_indices]
        # Epic 2: Do NOT average. Return raw atomic descriptors.
        return local_desc

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
        max_rows = kwargs.get('max_rows', 50000) # Safety cap (Epic 2)

        if not structures:
            logger.warning("No structures provided to MaxVolSampler.")
            return []

        if not potential_path:
            raise ValueError("potential_path is required for MaxVolSampler.")

        if not n_selection:
             raise ValueError("n_clusters is required for MaxVolSampler.")

        # 1. Compute Descriptors
        logger.info(f"Computing descriptors for {len(structures)} structures...")

        desc_computer = LocalStructureDescriptor(potential_path)

        # We need to stack all descriptors into X_global
        # And keep track of which row belongs to which structure/atom

        X_blocks = []
        row_map = [] # List of (structure_index, atom_index_local_to_subset?)
        # Actually we just need structure_index to select the structure later.

        valid_structure_indices = []

        for i, atoms in enumerate(structures):
            try:
                # Epic 2: Get raw atomic descriptors (N_subset, N_feat)
                desc_matrix = desc_computer.compute(atoms, atom_indices)

                if desc_matrix.shape[0] > 0:
                    X_blocks.append(desc_matrix)
                    # Track which structure these rows belong to
                    # We create an array of structure index 'i' repeated N_subset times
                    row_map.append(np.full(desc_matrix.shape[0], i, dtype=int))
                    valid_structure_indices.append(i)

            except Exception as e:
                logger.warning(f"Failed descriptor calc for structure {i}: {e}")

        if not X_blocks:
            return []

        X_global = np.vstack(X_blocks)
        row_map_global = np.concatenate(row_map) # Map: row_idx -> structure_idx

        # Epic 2: Safety Cap
        total_rows = X_global.shape[0]
        if total_rows > max_rows:
            logger.info(f"X_global size {total_rows} exceeds cap {max_rows}. Subsampling...")
            # Random subsample
            indices_to_keep = np.random.choice(total_rows, max_rows, replace=False)
            X_global = X_global[indices_to_keep]
            row_map_global = row_map_global[indices_to_keep]

        # 2. MaxVol Selection
        selected_row_indices = []

        if pyace and hasattr(pyace, "SelectMaxVol"):
             # pyace expects (N_samples, N_features)
             selected_row_indices = pyace.SelectMaxVol(X_global, n_selection)
        else:
             # Fallback: QR pivoting
             from scipy.linalg import qr

             limit = min(n_selection, X_global.shape[1], X_global.shape[0])
             if limit < n_selection:
                 logger.warning(f"Requested {n_selection} but capped at {limit} (rank/size limit).")

             # QR on transpose
             _, _, P = qr(X_global.T, pivoting=True)
             selected_row_indices = P[:limit]

        # Map selected rows back to structures
        selected_structure_indices = np.unique(row_map_global[selected_row_indices])

        # 3. Identify Max Gamma Atom in selected structures
        calc = desc_computer.ace_sampler.calculator

        results = []
        for i in selected_structure_indices:
            atoms = structures[i]

            try:
                gamma_arr = calc.get_property("gamma", atoms)

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
