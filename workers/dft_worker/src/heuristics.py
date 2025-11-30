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

        Strategy:
            - Transition Metals / Rare Earths present -> 'metal'
            - Only Main Group elements:
                - If Anions present (O, F, N, etc.) -> Likely 'insulator' (e.g., SiO2, NaCl)
                - No Anions -> Likely 'metal' or 'semimetal' (e.g., pure Al, Si), defaults to 'metal' for safety.
        """
        symbols = set(atoms.get_chemical_symbols())
        has_transition = any(cls._is_transition_metal(s) for s in symbols)
        has_rare_earth = any(cls._is_rare_earth(s) for s in symbols)
        has_anion = any(s in cls.ANIONS for s in symbols)

        if has_transition or has_rare_earth:
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
        symbols = set(atoms.get_chemical_symbols())
        has_magnetic = any(s in cls.MAGNETIC_ELEMENTS for s in symbols)

        if has_magnetic:
            recommendations["magnetism"]["nspin"] = 2
            mag_map = {}
            for s in symbols:
                if s in cls.MAGNETIC_ELEMENTS:
                    mag_map[s] = cls.MAGNETIC_ELEMENTS[s]
                else:
                    mag_map[s] = 0.0
            recommendations["magnetism"]["moments"] = mag_map

        return recommendations
