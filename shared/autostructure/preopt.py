import numpy as np
from typing import Dict, Set, Optional
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

try:
    from mace.calculators import mace_mp
    HAS_MACE = True
except ImportError:
    HAS_MACE = False

class PreOptimizer:
    """
    Safety valve to perform geometric sanity checks and basic relaxation
    before expensive DFT calculations.

    NOTE: Default behavior uses ase.optimize.BFGS, which performs Fixed-Cell relaxation
    (only atomic positions are relaxed, lattice vectors remain constant).
    Strategies like MolecularGenerator's 'high_pressure_packing' rely on this behavior
    to maintain compressed cell volumes while relaxing atomic overlaps.
    """

    def __init__(
        self,
        lj_params: Dict[str, float],
        emt_elements: Optional[Set[str]] = None,
        fmax: float = 0.1,
        steps: int = 200,
        mic_distance: float = 0.8
    ):
        """
        Args:
            lj_params (Dict[str, float]): LJ parameters (Ignored in MACE upgrade).
            emt_elements (Set[str], optional): Elements allowed for EMT (Ignored in MACE upgrade).
            fmax (float): Maximum force threshold for relaxation.
            steps (int): Maximum number of relaxation steps.
            mic_distance (float): Minimum interatomic distance threshold (Å) for discarding.
        """
        self.fmax = fmax
        self.steps = steps
        self.mic_distance = mic_distance

    def get_calculator(self, atoms: Atoms) -> Calculator:
        """
        Returns an appropriate lightweight calculator (MACE) for the given structure.
        """
        if not HAS_MACE:
            raise ImportError("mace library is not installed. Cannot perform pre-optimization.")

        try:
            return mace_mp(model="medium", device="cpu", default_dtype="float64")
        except Exception as e:
            print(f"Warning: Failed to load MACE 'medium' model ({e}). Falling back to 'small'.")
            return mace_mp(model="small", device="cpu", default_dtype="float64")

    def run_pre_optimization(self, atoms: Atoms) -> Atoms:
        """
        Run pre-optimization on the given structure.

        Returns:
            Atoms: The relaxed structure.

        Raises:
            ValueError: If the structure is physically unsound (too close atoms) after relaxation.
        """
        # Work on a copy
        atoms = atoms.copy()

        # Attach calculator
        atoms.calc = self.get_calculator(atoms)

        try:
            dyn = BFGS(atoms, logfile=None) # Suppress log output
            dyn.run(fmax=self.fmax, steps=self.steps)
        except Exception:
            # If relaxation fails (e.g., explosion), we catch it
            # But we proceed to check distances. If it exploded, distances might be weird or fine.
            pass

        # Final Sanity Check: Distance Matrix
        # We need to account for PBC.
        if len(atoms) > 1:
            # get_all_distances with mic=True handles PBC
            dists = atoms.get_all_distances(mic=True)
            # Mask diagonal (self-distance is 0)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < self.mic_distance:
                # Discard
                raise ValueError(f"Structure discarded: Atoms too close ({min_dist:.2f} Å < {self.mic_distance} Å)")

        return atoms
