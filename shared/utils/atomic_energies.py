import json
import logging
from pathlib import Path
from typing import Dict, List, Callable
from ase import Atoms
from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

class AtomicEnergyManager:
    """Manages isolated atomic energies (E0) for Delta learning."""

    def __init__(self, json_path: Path, pseudo_dir: Path):
        self.json_path = json_path
        self.pseudo_dir = pseudo_dir

    def get_e0(self, elements: List[str], calculator_factory: Callable[[str], Calculator]) -> Dict[str, float]:
        """
        Loads E0 from sidecar JSONs in pseudo_dir or calculates them if missing.

        Args:
            elements: List of element symbols.
            calculator_factory: Function that accepts an element symbol and returns a new Calculator instance.

        Returns:
            Dict mapping element symbol to atomic energy (E0).
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"SSSP JSON file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            sssp_data = json.load(f)

        e0_dict = {}

        for el in elements:
            if el not in sssp_data:
                 logger.warning(f"Element {el} not found in SSSP JSON. Skipping E0 check for it.")
                 continue

            # Locate Pseudopotential file
            # Assuming SSSP JSON structure has 'filename'
            filename = sssp_data[el].get("filename")
            if not filename:
                 logger.warning(f"No filename found for {el} in SSSP. Cannot locate PP.")
                 continue

            pp_path = self.pseudo_dir / filename
            cache_path = pp_path.with_suffix(".json")

            # Check cache
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                        e0_dict[el] = data['energy']
                        logger.info(f"Loaded cached E0 for {el} from {cache_path}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_path}: {e}. Recalculating.")

            logger.info(f"E0 for {el} not cached. Calculating...")

            # Calculate
            try:
                # Create isolated atom in a large box
                atom = Atoms(el, cell=[15.0, 15.0, 15.0], pbc=True)
                atom.center()

                calc = calculator_factory(el)
                atom.calc = calc

                e = atom.get_potential_energy()
                e0_dict[el] = float(e)

                logger.info(f"Calculated and cached E0 for {el} to {cache_path}")

                # Save to cache
                try:
                    with open(cache_path, 'w') as f:
                        json.dump({"energy": float(e)}, f)
                except Exception as w_err:
                     logger.warning(f"Could not write cache to {cache_path}: {w_err}")

            except Exception as e:
                logger.error(f"Failed to calculate E0 for {el}: {e}")
                raise e

        return e0_dict
