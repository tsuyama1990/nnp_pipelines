"""Shifted Lennard-Jones Calculator."""

import numpy as np
from typing import Optional, Dict, Any, Union
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.neighborlist import NeighborList
from ase.data import covalent_radii, atomic_numbers

class ShiftedLennardJones(Calculator):
    """LennardJones calculator with potential shift to ensure V(rc) = 0.
    Supports species-specific parameters and mixing rules.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self,
                 epsilon: Union[float, Dict[str, float]] = 1.0,
                 sigma: Optional[Union[float, Dict[str, float]]] = None,
                 rcut: float = 2.5,
                 shift_energy: bool = True,
                 **kwargs):
        """Initialize ShiftedLennardJones.

        Args:
            epsilon: Energy well depth. float or dict {element: value}.
            sigma: Distance parameter. float or dict {element: value}.
                   If None, defaults to 2 * covalent_radius * 0.89.
            rcut: Cutoff radius.
            shift_energy: If True, shift the potential energy so that V(rc) = 0.
            **kwargs: Arguments passed to Calculator.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcut = rcut
        self.shift_energy = shift_energy

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=None,
        system_changes=all_changes
    ):
        """Calculate properties."""
        if properties is None:
            properties = ['energy', 'forces', 'stress']

        super().calculate(atoms, properties, system_changes)

        # Reset results
        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(self.atoms), 3)),
            'stress': np.zeros(6)
        }

        # Prepare parameters
        elements = self.atoms.get_chemical_symbols()

        # Helper to get params for index i
        def get_params(idx):
            el = elements[idx]
            if isinstance(self.epsilon, dict):
                eps = self.epsilon.get(el)
                if eps is None:
                    raise ValueError(f"Epsilon not found for element {el}")
            else:
                eps = self.epsilon

            # Dynamic Sigma Logic
            if self.sigma is None:
                z = atomic_numbers[el]
                sig = 2 * covalent_radii[z] * 0.89
            elif isinstance(self.sigma, dict):
                sig = self.sigma.get(el)
                if sig is None:
                    # Fallback to dynamic default if missing in dict
                    z = atomic_numbers[el]
                    sig = 2 * covalent_radii[z] * 0.89
            else:
                sig = self.sigma
            return eps, sig

        # Use NeighborList for efficiency
        # NeighborList uses a list of cutoffs (radii). We use rcut/2 for all.
        cutoffs = [self.rcut / 2.0] * len(self.atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        nl.update(self.atoms)

        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        # Virial stress tensor (Voigt notation or full tensor). We compute full tensor.
        virial = np.zeros((3, 3))

        for i in range(len(self.atoms)):
            indices, offsets = nl.get_neighbors(i)
            cells = np.dot(offsets, self.atoms.get_cell())

            # Get positions
            pos_i = self.atoms.positions[i]
            pos_j = self.atoms.positions[indices] + cells

            # Vectors r_ij = r_j - r_i
            vec_ij = pos_j - pos_i
            dists_sq = np.sum(vec_ij**2, axis=1)

            # Filter by rcut explicitly
            mask = dists_sq < (self.rcut**2)
            if not np.any(mask):
                continue

            dists_sq = dists_sq[mask]
            vec_ij = vec_ij[mask]
            indices = indices[mask]

            # Parameters for atom i
            eps_i, sig_i = get_params(i)

            for k, j_idx in enumerate(indices):
                eps_j, sig_j = get_params(j_idx)

                # Mixing rules
                sig_ij = (sig_i + sig_j) / 2.0
                eps_ij = np.sqrt(eps_i * eps_j)

                r2 = dists_sq[k]
                r6 = (sig_ij**2 / r2)**3
                r12 = r6**2

                # Energy: 4*eps * (r12 - r6)
                e_pair = 4.0 * eps_ij * (r12 - r6)

                # Force magnitude: F = -dV/dr.
                # F_vec_i (force on i) = -dV/dr * (vec_ij / r) = - (24*eps/r^2)*(2*r12 - r6) * vec_ij
                f_scalar = (24.0 * eps_ij / r2) * (2.0 * r12 - r6)
                f_vec_i = - f_scalar * vec_ij[k]

                forces[i] += f_vec_i

                # Virial contribution.
                # W = sum_pairs r_ij \otimes F_ij.
                # F_ij is force on i due to j (f_vec_i).
                # r_ij is vector from j to i = -vec_ij[k].
                # Contribution = (-vec_ij[k]) \otimes f_vec_i
                #              = (-vec_ij[k]) \otimes (-f_scalar * vec_ij[k])
                #              = f_scalar * (vec_ij[k] \otimes vec_ij[k])

                # Since we visit each pair twice (bothways=True), we take half contribution per visit.
                virial += 0.5 * f_scalar * np.outer(vec_ij[k], vec_ij[k])

                # Shift
                if self.shift_energy:
                    sr_cut = sig_ij / self.rcut
                    v_cut = 4.0 * eps_ij * (sr_cut**12 - sr_cut**6)
                    e_pair -= v_cut

                energy += e_pair

        # NeighborList double counts (bothways=True), so we divide energy by 2.
        self.results['energy'] = energy / 2.0
        self.results['forces'] = forces

        # Stress (Voigt) from Virial
        # ASE sign convention: stress = -virial / volume.

        vol = self.atoms.get_volume()
        stress_tensor = -virial / vol

        # Convert to Voigt: xx, yy, zz, yz, xz, xy
        self.results['stress'] = np.array([
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1]
        ])
