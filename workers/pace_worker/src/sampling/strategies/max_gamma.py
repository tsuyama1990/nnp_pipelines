"""Max Gamma Sampler Strategy.

This module implements a sampling strategy that selects structures with the
highest extrapolation grade (gamma), indicating high uncertainty.
"""

import logging
from typing import List, Tuple, Any, Optional
import numpy as np
from ase import Atoms

from shared.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class MaxGammaSampler(Sampler):
    """Selects candidates with the highest uncertainty (Max Gamma)."""

    def sample(self, candidates: List[Atoms], n_samples: int, **kwargs: Any) -> List[Tuple[Atoms, int]]:
        """Select n_samples with highest max gamma.

        Args:
            candidates: List of candidate structures.
            n_samples: Number of structures to select.
            **kwargs:
                - 'gammas' (List[float]): Global max gammas.
                - 'atom_indices' (List[int], optional): If provided, re-evaluates max_gamma
                  only on these atoms (requires per-atom gamma in atoms.arrays['gamma']).

        Returns:
            List[Tuple[Atoms, int]]: List of selected (structure, original_index) tuples.
        """
        n_candidates = len(candidates)
        if n_candidates == 0:
            return []

        if n_samples >= n_candidates:
             return [(c, i) for i, c in enumerate(candidates)]

        # Extract gammas
        gammas = []
        atom_indices = kwargs.get('atom_indices') # Epic 6: Local MaxGamma

        for atoms in candidates:
            if atom_indices is not None and 'gamma' in atoms.arrays:
                # Local evaluation
                # Check if indices are valid
                valid_indices = [i for i in atom_indices if i < len(atoms)]
                if valid_indices:
                    local_gammas = atoms.arrays['gamma'][valid_indices]
                    gammas.append(np.max(local_gammas))
                else:
                    gammas.append(0.0) # Indices out of range or empty
            elif 'max_gamma' in atoms.info:
                gammas.append(atoms.info['max_gamma'])
            elif 'gamma' in atoms.arrays:
                gammas.append(np.max(atoms.arrays['gamma']))
            else:
                gammas.append(0.0)

        if len(gammas) != n_candidates:
            logger.error("Length of gammas does not match candidates.")
            return []

        # Sort by gamma descending
        indices = list(range(n_candidates))
        indices.sort(key=lambda i: gammas[i], reverse=True)

        selected_indices = indices[:n_samples]

        if selected_indices:
            logger.info(f"MaxGammaSampler: Selected {len(selected_indices)} structures. "
                        f"Top Gamma: {gammas[selected_indices[0]]:.4f}, "
                        f"Bottom Gamma: {gammas[selected_indices[-1]]:.4f}")

        return [(candidates[i], i) for i in selected_indices]
