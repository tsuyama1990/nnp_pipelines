"""Pymatgen-based Heuristics for Physical Parameters.

This module analyzes atomic structures to infer optimal physical parameters for DFT calculations,
such as smearing settings and initial magnetic moments.
It gracefully falls back to internal logic if pymatgen is not installed.
"""

import logging
from typing import Dict, Any, Set, List, Optional
from ase import Atoms
from ase.data import chemical_symbols

logger = logging.getLogger(__name__)

# Fallback Data
# Transition Metals (roughly Sc-Zn, Y-Cd, La-Hg)
FALLBACK_TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
}

# Rare Earths (Lanthanides)
FALLBACK_RARE_EARTHS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}

class PymatgenHeuristics:
    """Infers physical parameters using knowledge of materials physics."""

    # Typical magnetic elements and their initial moments
    MAGNETIC_ELEMENTS = {
        "Fe": 5.0, "Co": 2.0, "Ni": 1.0, "Mn": 5.0, "Cr": 3.0,
        "Eu": 7.0, "Gd": 7.0, "Tb": 6.0, "Dy": 10.0, "Ce": 1.0, "Nd": 3.0, "Sm": 1.0
    }

    # Anions used for oxide/insulator detection
    ANIONS = {"O", "F", "Cl", "S", "N", "P", "Br", "I", "Se", "Te"}

    @staticmethod
    def _is_transition_metal(symbol: str) -> bool:
        """Check if element is a transition metal."""
        try:
            from pymatgen.core import Element
            e = Element(symbol)
            if hasattr(e, "is_transition_metal"):
                return e.is_transition_metal
            return symbol in FALLBACK_TRANSITION_METALS
        except (ImportError, AttributeError):
            return symbol in FALLBACK_TRANSITION_METALS

    @staticmethod
    def _is_rare_earth(symbol: str) -> bool:
        """Check if element is a rare earth metal."""
        try:
            from pymatgen.core import Element
            e = Element(symbol)
            if hasattr(e, "is_rare_earth_metal"):
                return e.is_rare_earth_metal
            if hasattr(e, "is_rare_earth"):
                return e.is_rare_earth
            # Check lanthanoids + Sc, Y
            if symbol in {"Sc", "Y"}:
                return True
            if hasattr(e, "is_lanthanoid"):
                return e.is_lanthanoid
            return symbol in FALLBACK_RARE_EARTHS
        except (ImportError, AttributeError):
            return symbol in FALLBACK_RARE_EARTHS

    @classmethod
    def _estimate_electronic_type(cls, atoms: Atoms) -> str:
        """Estimate if the material is a metal or insulator/semiconductor.

        Strategy (Epic 3):
            - Transition Metal + Anion (e.g. FeO, NiO) -> 'insulator' (Mott insulator logic)
            - Transition Metal ONLY -> 'metal'
            - Main Group + Anion -> 'insulator'
            - Main Group ONLY -> 'metal' (default)
        """
        symbols = set(atoms.get_chemical_symbols())
        has_transition = any(cls._is_transition_metal(s) for s in symbols)
        has_rare_earth = any(cls._is_rare_earth(s) for s in symbols)
        has_anion = any(s in cls.ANIONS for s in symbols)

        if (has_transition or has_rare_earth) and has_anion:
            # Epic 3: FeO, NiO, etc. should be treated as insulators (fixed occupations)
            return "insulator"

        if has_transition or has_rare_earth:
            # Pure metals or alloys
            return "metal"

        if has_anion:
            # Main group + Anion -> Likely Ionic/Covalent Insulator
            return "insulator"

        # Main group, no anions (e.g. Pure Si, Graphite, Al)
        # Defaulting to metal (smearing) is safer for convergence than assuming fixed gap.
        return "metal"

    @classmethod
    def get_recommended_params(cls, atoms: Atoms, override_type: Optional[str] = None) -> Dict[str, Any]:
        """Infer recommended DFT parameters for the given structure.

        Args:
            atoms: The ASE Atoms object to analyze.
            override_type: 'metal' or 'insulator' to force specific logic.

        Returns:
            dict: Dictionary containing recommendations for 'system' and magnetic settings.
        """
        elec_type = override_type if override_type else cls._estimate_electronic_type(atoms)

        recommendations = {
            "system": {},
            "magnetism": {
                "nspin": 1,
                "starting_magnetization": {}
            }
        }

        # 1. Smearing / Occupations
        if elec_type == "insulator":
            recommendations["system"]["occupations"] = "fixed"
        else:
            recommendations["system"]["occupations"] = "smearing"
            recommendations["system"]["smearing"] = "mv" # Methfessel-Paxton
            recommendations["system"]["degauss"] = 0.02

        # 2. Magnetism Logic
        all_symbols = atoms.get_chemical_symbols()
        symbols_set = set(all_symbols)
        has_magnetic = any(s in cls.MAGNETIC_ELEMENTS for s in symbols_set)

        if has_magnetic:
            recommendations["magnetism"]["nspin"] = 2

            # Epic 3: AFM Heuristic (Index-based Alternating)
            # We construct 'starting_magnetization' list or map?
            # QE uses per-species starting magnetization usually, unless we use atomic indices?
            # The interface `starting_magnetization` in QE (via ASE) is typically a dict {Label: Moment}.
            # But to support AFM, we need *per atom* moments, which ASE handles via `atoms.set_initial_magnetic_moments()`.
            # This method returns recommendations for the CALCULATOR parameters or ATOMS preparation?
            # "starting_magnetization" dict usually maps species -> value. This can't do AFM for mono-element systems (like AFM Cr).
            # To support AFM, we must return a per-atom list or rely on caller to set it on atoms.
            # The signature returns a dict. We will add a special key "initial_magnetic_moments" (list)
            # which the caller (configurator) should apply to atoms.

            initial_moments = []
            mag_counter = 0

            # Pymatgen oxidation state guess could be added here, but user said "High Spin" is acceptable default.
            # We stick to cls.MAGNETIC_ELEMENTS for magnitude.

            for s in all_symbols:
                if s in cls.MAGNETIC_ELEMENTS:
                    mag = cls.MAGNETIC_ELEMENTS[s]

                    # Apply Alternating Flip (+ - + -) for magnetic species
                    if mag_counter % 2 == 1:
                        mag = -mag

                    initial_moments.append(mag)
                    mag_counter += 1
                else:
                    initial_moments.append(0.0)

            recommendations["magnetism"]["initial_magnetic_moments"] = initial_moments

            # Also provide the species-wise map for FM fallback or simple calculators
            mag_map = {}
            for s in symbols_set:
                if s in cls.MAGNETIC_ELEMENTS:
                    mag_map[s] = cls.MAGNETIC_ELEMENTS[s]
                else:
                    mag_map[s] = 0.0
            recommendations["magnetism"]["moments"] = mag_map

        return recommendations
