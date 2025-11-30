"""Shifted Lennard-Jones Calculator."""

import numpy as np
from typing import Optional, Dict, Any, Union
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList

class ShiftedLennardJones(Calculator):
    """LennardJones calculator with potential shift to ensure V(rc) = 0.
    Supports species-specific parameters and mixing rules.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 epsilon: Union[float, Dict[str, float]] = 1.0,
                 sigma: Union[float, Dict[str, float]] = 1.0,
                 rcut: float = 2.5,
                 shift_energy: bool = True,
                 **kwargs):
        """Initialize ShiftedLennardJones.

        Args:
            epsilon: Energy well depth. float or dict {element: value}.
            sigma: Distance parameter. float or dict {element: value}.
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
            properties = ['energy', 'forces']

        super().calculate(atoms, properties, system_changes)

        # Reset results
        self.results = {'energy': 0.0, 'forces': np.zeros((len(self.atoms), 3))}

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

            if isinstance(self.sigma, dict):
                sig = self.sigma.get(el)
                if sig is None:
                    raise ValueError(f"Sigma not found for element {el}")
            else:
                sig = self.sigma
            return eps, sig

        # Use NeighborList for efficiency
        # cutoffs: we need half of max cutoff? No, NeighborList takes cutoffs list.
        # But we have a global cutoff rcut.
        cutoffs = [self.rcut / 2.0] * len(self.atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        nl.update(self.atoms)

        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))

        for i in range(len(self.atoms)):
            indices, offsets = nl.get_neighbors(i)
            cells = np.dot(offsets, self.atoms.get_cell())

            # Get positions
            pos_i = self.atoms.positions[i]
            pos_j = self.atoms.positions[indices] + cells

            # Vectors r_ij = r_j - r_i
            vec_ij = pos_j - pos_i
            dists_sq = np.sum(vec_ij**2, axis=1)

            # Filter by rcut explicitly (NeighborList uses skin, so might include slightly larger)
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
                # sigma_ij = (sigma_i + sigma_j) / 2
                # epsilon_ij = sqrt(epsilon_i * epsilon_j)
                sig_ij = (sig_i + sig_j) / 2.0
                eps_ij = np.sqrt(eps_i * eps_j)

                r2 = dists_sq[k]
                r6 = (sig_ij**2 / r2)**3
                r12 = r6**2

                # Energy: 4*eps * (r12 - r6)
                e_pair = 4.0 * eps_ij * (r12 - r6)

                # Force magnitude (derivative of potential w.r.t r)
                # F = -dV/dr
                # V = 4*eps*(s^12*r^-12 - s^6*r^-6)
                # dV/dr = 4*eps*(-12*s^12*r^-13 + 6*s^6*r^-7)
                #       = (24*eps/r) * (s^6/r^6 - 2*s^12/r^12)
                #       = (24*eps/r2) * (r6 - 2*r12) * r ? No
                # Let's derive simpler:
                # F_mag = (24 * eps / r2) * (2 * (sig/r)^12 - (sig/r)^6)
                # F_vec = F_mag * (vec_ij / r) ?
                # F_vec = -dV/dr * (vec_ij / r)
                #       = - (24*eps/r) * (s^6/r^6 - 2*s^12/r^12) * (vec_ij/r)
                #       = (24*eps/r^2) * (2*s^12/r^12 - s^6/r^6) * vec_ij
                #       = (24*eps/r^2) * (2*r12 - r6) * vec_ij

                f_factor = (24.0 * eps_ij / r2) * (2.0 * r12 - r6)
                f_vec = f_factor * vec_ij[k] # Force on i due to j?

                # Force on i is F_ij (force exerted by j on i).
                # Potential V depends on |r_i - r_j|.
                # F_i = - gradient_i V
                # gradient_i |r_i - r_j| = (r_i - r_j) / |r_i - r_j| = -vec_ij / r
                # So F_i = - dV/dr * (-vec_ij/r) = (dV/dr) * (vec_ij/r)
                # dV/dr = (24*eps/r) * (r6 - 2*r12)  <-- wait, sign
                # V(x) = x^-12 - x^-6. V'(x) = -12x^-13 + 6x^-7
                # dV/dr = 4*eps * (-12/r * r12 + 6/r * r6) = 24*eps/r * (0.5*r6 - r12)
                # F_i = (24*eps/r2) * (0.5*r6 - r12) * vec_ij
                #     = (24*eps/r2) * (r6 - 2*r12) * (-0.5) * vec_ij ? No.

                # Correct formula: F = 24*eps/r^2 * (2*(sig/r)^12 - (sig/r)^6) * vec_ij
                # This vector points from j to i (repulsive).
                # vec_ij = r_j - r_i. Points from i to j.
                # So if atoms are too close, F should push i away from j (opposite to vec_ij).
                # (2*r12 - r6) is positive for small r.
                # So F matches vec_ij (towards j)? No that would be attractive.
                # So we need negative sign.

                # Let's check ASE implementation or standard.
                # Force on atom i: F_i = sum_j F_ij
                # F_ij = 24 eps (2(sigma/r)^12 - (sigma/r)^6) \frac{r_i - r_j}{r^2}
                # r_i - r_j = -vec_ij.
                # So F_vec = - (24 * eps / r2) * (2 * r12 - r6) * vec_ij[k]

                forces[i] += - (24.0 * eps_ij / r2) * (2.0 * r12 - r6) * vec_ij[k]

                # Shift
                if self.shift_energy:
                    sr_cut = sig_ij / self.rcut
                    v_cut = 4.0 * eps_ij * (sr_cut**12 - sr_cut**6)
                    e_pair -= v_cut

                energy += e_pair

        # NeighborList double counts (bothways=True), so we divide energy by 2.
        # Forces are accumulated correctly because we iterate over all i.
        # But wait, if bothways=True, for pair (i,j) we visit i then j is neighbor, AND visit j then i is neighbor.
        # Forces on i from j are added when we visit i.
        # Forces on j from i are added when we visit j.
        # So forces are fine.
        # Energy is calculated twice per pair.

        self.results['energy'] = energy / 2.0
        self.results['forces'] = forces
