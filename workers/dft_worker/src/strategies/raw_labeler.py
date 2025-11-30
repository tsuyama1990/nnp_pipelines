"""Raw Labeler Strategy."""

import logging
from typing import Optional, Dict
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from shared.core.interfaces import Labeler

logger = logging.getLogger(__name__)

class RawLabeler(Labeler):
    """Calculates the raw DFT energy and forces (no baseline)."""

    def __init__(self, reference_calculator: Calculator,
                 e0_dict: Optional[Dict[str, float]] = None, outlier_energy_max: float = 10.0,
                 magnetism_settings: Optional[Dict[str, float]] = None):
        """Initialize the RawLabeler.

        Args:
            reference_calculator: A configured ASE calculator for the ground truth (DFT).
            e0_dict: Dictionary of isolated atomic energies (subtracted if provided).
            outlier_energy_max: Max energy per atom (eV) allowed before discarding.
            magnetism_settings: Dictionary of initial magnetic moments {Element: moment}.
        """
        self.reference_calculator = reference_calculator
        self.e0_dict = e0_dict or {}
        self.outlier_energy_max = outlier_energy_max
        self.magnetism_settings = magnetism_settings

    def label(self, structure: Atoms) -> Optional[Atoms]:
        """Compute raw DFT energy, forces, and stress for a given structure.

        Args:
            structure: The atomic cluster to label.

        Returns:
            Atoms: The cluster with raw DFT energy, forces, and stress.
                   Returns None if the DFT calculation fails.
        """
        # Work on copy
        cluster_ref = structure.copy()

        # 1. Reference Calculation (DFT)
        try:
            # Apply Initial Magnetic Moments if settings exist
            if self.magnetism_settings:
                initial_mags = [self.magnetism_settings.get(s, 0.0) for s in cluster_ref.get_chemical_symbols()]
                cluster_ref.set_initial_magnetic_moments(initial_mags)
                logger.debug(f"Applied magnetic moments: {initial_mags}")

            cluster_ref.calc = self.reference_calculator
            e_ref = cluster_ref.get_potential_energy()
            f_ref = cluster_ref.get_forces()
            s_ref = cluster_ref.get_stress() # Voigt form (6,) in eV/A^3
        except Exception as e:
            logger.warning(f"Reference (DFT) Calculation failed: {e}")
            return None

        # 2. Subtract E0 if provided (usually Pacemaker wants Binding Energy or similar)
        # But if RawLabeler is for "No Delta", we usually still want E0 subtraction for stability/convention?
        # The prompt says: "Instantiate a new RawLabeler ... that saves raw DFT energy/forces directly as the target."
        # Pacemaker usually fits to Energy - sum(E0).
        # So we should subtract E0 if available.
        e_offset = 0.0
        if self.e0_dict:
            try:
                e_offset = sum(self.e0_dict[s] for s in structure.get_chemical_symbols())
            except KeyError as e:
                logger.error(f"Missing E0 for element: {e}")
                return None

        # Target = DFT - E0
        e_target = e_ref - e_offset

        # [Filter] Outlier Check (using binding energy per atom usually)
        if abs(e_target) / len(structure) > self.outlier_energy_max:
             logger.warning(f"Discarding outlier: Energy {e_target/len(structure):.2f} eV/atom exceeds threshold {self.outlier_energy_max}.")
             return None

        result_cluster = cluster_ref.copy()
        result_cluster.calc = None

        # Convert Stress to Virial (Extensive) [eV]
        volume = structure.get_volume()
        virial = -1.0 * s_ref * volume

        # 3. Store Results
        # Pacemaker reads 'energy' and 'forces' from atoms.info/arrays
        result_cluster.info['energy'] = e_target
        result_cluster.arrays['forces'] = f_ref

        result_cluster.info['virial'] = virial
        if 'stress' in result_cluster.info:
            del result_cluster.info['stress']

        # Store Raw Info
        result_cluster.info['energy_dft_raw'] = e_ref
        result_cluster.arrays['forces_dft_raw'] = f_ref
        result_cluster.info['e0_offset'] = e_offset

        # Weights
        result_cluster.info['energy_weight'] = 1.0
        result_cluster.info['virial_weight'] = 1.0

        return result_cluster
