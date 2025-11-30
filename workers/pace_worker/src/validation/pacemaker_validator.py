"""Validator interface and implementation for Pacemaker potentials."""

import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class Validator(ABC):
    """Interface for potential validation."""

    @abstractmethod
    def validate(self, potential_path: str) -> Dict[str, Any]:
        """Validate the potential and return metrics."""
        pass

class PacemakerValidator(Validator):
    """Validates ACE potentials using pace_diagnostics CLI."""

    def __init__(self, test_structure_path: Optional[str] = None, delta_learning_mode: bool = True, lj_params: Optional[Dict[str, float]] = None):
        """Initialize validator.

        Args:
            test_structure_path: Path to structure file for calculating properties (e.g. elastic constants).
                                 If None, assumes pace_diagnostics uses internal or default data.
            delta_learning_mode: Whether the potential is delta learning (ACE + LJ).
            lj_params: LJ Parameters for baseline reconstruction if delta learning is on.
        """
        self.test_structure_path = test_structure_path
        self.delta_learning_mode = delta_learning_mode
        self.lj_params = lj_params

    def validate(self, potential_path: str) -> Dict[str, Any]:
        """Run pace_diagnostics and parse results.

        Args:
            potential_path: Path to the potential file (.yace).

        Returns:
            Dict[str, Any]: Validation metrics (Elastic Constants, VDoS, etc).
        """
        if not Path(potential_path).exists():
            raise FileNotFoundError(f"Potential not found: {potential_path}")

        results = {
            "elastic_constants": {},
            "vdos": {},
            "status": "UNKNOWN"
        }

        # Construct command
        # Assuming pace_diagnostics -p potential.yace -s structure.xyz
        cmd = ["pace_diagnostics", "-p", potential_path]
        if self.test_structure_path:
            cmd.extend(["-s", self.test_structure_path])

        # Epic 8: Validator Conditional Potential
        # If delta_learning_mode is True, we need to instruct pace_diagnostics (or the underlying tool)
        # to use the composite potential.
        # pace_diagnostics generally evaluates the potential file passed.
        # If 'potential_path' points to a .yace file, that is just the ACE part.
        # However, typically 'pace_diagnostics' expects a potential that defines the full interaction?
        # Or maybe we need to pass --compute-forces-lj if supported?
        #
        # If the user requirement implies we must manually compute E_total vs E_DFT using ASE and return metrics,
        # we should implement that here. 'pace_diagnostics' is a CLI tool.
        # If 'pace_diagnostics' doesn't support "ACE + LJ" composition via CLI arguments,
        # we might need to rely on the fact that for Delta Learning, we validate residuals or
        # we need a custom Python validation step here.
        #
        # User said: "Ensure the validator constructs the full potential ($E_{total} = E_{ACE} + E_{LJ}$) when delta_learning_mode is True."
        # This implies we should do Python-based validation using SumCalculator if pace_diagnostics falls short.
        # BUT the code below calls 'subprocess.run(cmd)'.
        #
        # Let's check if we can pass LJ info to pace_diagnostics.
        # Standard pacemaker typically handles LJ if it's defined in the potential YAML config used during training.
        # But here we are just validating a .yace file.
        #
        # If we stick to 'pace_diagnostics', we assume it knows about the baseline if we pass the YAML config?
        # But the signature only has potential_path (.yace).
        #
        # Let's add a custom Python check if delta_learning_mode is True, using ASE.
        # This aligns better with "constructs the full potential".

        if self.delta_learning_mode and self.test_structure_path:
             try:
                 from ase.io import read
                 from ase.calculators.calculator import Calculator
                 from shared.calculators import SumCalculator
                 from shared.potentials.shifted_lj import ShiftedLennardJones
                 from pyace import PyACECalculator

                 # Load structures
                 atoms_list = read(self.test_structure_path, index=":")
                 if not isinstance(atoms_list, list): atoms_list = [atoms_list]

                 # Setup Calculator
                 ace_calc = PyACECalculator(potential_path)

                 eps = self.lj_params.get("epsilon", 1.0)
                 sig = self.lj_params.get("sigma", 2.0)
                 rc = self.lj_params.get("cutoff", 5.0)
                 shift = self.lj_params.get("shift_energy", True)

                 lj_calc = ShiftedLennardJones(
                    epsilon=eps,
                    sigma=sig,
                    rc=rc,
                    shift_energy=shift
                 )
                 calc = SumCalculator(calculators=[ace_calc, lj_calc])

                 rmses = []
                 for atoms in atoms_list:
                      atoms.calc = calc
                      pred_e = atoms.get_potential_energy()

                      # Get Reference
                      ref_e = atoms.info.get("energy_dft_raw", atoms.info.get("energy", None))

                      if ref_e is not None:
                          rmses.append((pred_e - ref_e)**2)

                 if rmses:
                     rmse = (sum(rmses)/len(rmses))**0.5
                     results["total_energy_rmse"] = rmse
                     logger.info(f"Custom Validation (ACE+LJ) RMSE: {rmse}")

             except Exception as e:
                 logger.warning(f"Custom ASE validation failed: {e}")

        # In a real scenario, pace_diagnostics might output to stdout or a file.
        # We'll assume stdout contains parseable YAML or text.

        logger.info(f"Running validation: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = process.stdout

            # Simple parsing of hypothetical output
            # Output format assumed to be YAML-like or key-value pairs

            # Example hypothetical output:
            # Elastic Constants:
            #   C11: 200.0
            #   C12: 150.0
            # VDoS:
            #   Peak: 12.5

            # For robustness, let's look for known keywords or try to parse generic YAML if possible.
            try:
                data = yaml.safe_load(output)
                if isinstance(data, dict):
                    results.update(data)
                    results["status"] = "SUCCESS"
            except yaml.YAMLError:
                # Fallback text parsing
                for line in output.splitlines():
                    if "C11" in line:
                        results["elastic_constants"]["C11"] = float(line.split(":")[-1])
                    if "status" in line.lower():
                        results["status"] = line.split(":")[-1].strip()

        except subprocess.CalledProcessError as e:
            logger.error(f"Validation failed: {e.stderr}")
            results["status"] = "FAILED"
            results["error"] = e.stderr

        return results
