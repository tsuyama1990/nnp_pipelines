import json
import logging
from pathlib import Path
from typing import Dict, List, Callable
from ase import Atoms
from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

class AtomicEnergyManager:
    """Manages isolated atomic energies (E0) for Delta learning using SSSP JSON as storage."""

    def __init__(self, json_path: Path):
        self.json_path = json_path

    def get_e0(self, elements: List[str], calculator_factory: Callable[[str], Calculator]) -> Dict[str, float]:
        """
        Loads E0 from SSSP JSON or calculates them if missing.
        calculator_factory: Function that accepts an element symbol and returns a new Calculator instance.

        Returns:
            Dict mapping element symbol to atomic energy (E0).
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"SSSP JSON file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            sssp_data = json.load(f)

        e0_dict = {}
        updated = False

        for el in elements:
            if el not in sssp_data:
                 logger.warning(f"Element {el} not found in SSSP JSON. Skipping E0 check for it.")
                 continue

            if "atomic_energy" in sssp_data[el]:
                e0_dict[el] = sssp_data[el]["atomic_energy"]
            else:
                logger.info(f"E0 for {el} not found in SSSP JSON. Calculating...")

                # Calculate
                try:
                    # Create isolated atom in a large box
                    atom = Atoms(el, cell=[15.0, 15.0, 15.0], pbc=True)
                    atom.center()

                    calc = calculator_factory(el)
                    atom.calc = calc

                    e = atom.get_potential_energy()
                    e0_dict[el] = float(e)

                    # Update data structure
                    sssp_data[el]["atomic_energy"] = float(e)
                    updated = True
                    logger.info(f"Calculated E0 for {el}: {e:.4f} eV")

                except Exception as e:
                    logger.error(f"Failed to calculate E0 for {el}: {e}")
                    raise e

        if updated:
            logger.info(f"Updating SSSP JSON with new atomic energies at {self.json_path}")
            # Write back to JSON
            # Preserve existing structure, just add atomic_energy key
            with open(self.json_path, 'w') as f:
                json.dump(sssp_data, f, indent=4)

        return e0_dict
