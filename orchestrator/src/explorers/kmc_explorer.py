import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Optional
from ase.io import read, write

from orchestrator.src.interfaces.explorer import BaseExplorer, ExplorationResult, ExplorationStatus
from orchestrator.src.services.kmc_service import KMCService
from orchestrator.src.services.al_service import ActiveLearningService
from shared.core.config import Config
from shared.core.enums import KMCStatus

logger = logging.getLogger(__name__)

class KMCExplorer(BaseExplorer):
    def __init__(self, kmc_service: KMCService, al_service: ActiveLearningService, config: Config):
        self.kmc_service = kmc_service
        self.al_service = al_service
        self.config = config

    def explore(self,
                current_structure: str,
                potential_path: str,
                iteration: int,
                **kwargs) -> ExplorationResult:

        work_dir = Path.cwd()

        # Check if current_structure is valid
        p = Path(current_structure)
        if not p.exists():
            return ExplorationResult(status=ExplorationStatus.FAILED, metadata={"error": "Structure not found"})

        try:
            kmc_input_atoms = read(current_structure, index=-1, format="lammps-dump-text" if "lammpstrj" in str(current_structure) else None)
        except Exception as e:
            # Fallback for lammps-data
            try:
                kmc_input_atoms = read(current_structure, format="lammps-data")
            except Exception as e2:
                logger.error(f"Failed to read structure {current_structure}: {e2}")
                return ExplorationResult(status=ExplorationStatus.FAILED, metadata={"error": str(e2)})

        self.al_service.ensure_chemical_symbols(kmc_input_atoms)

        kmc_result = self.kmc_service.run_step(kmc_input_atoms, potential_path)

        if kmc_result.status == KMCStatus.SUCCESS:
            next_input_file = work_dir / "kmc_output.data"
            write(next_input_file, kmc_result.structure, format="lammps-data", velocities=True)
            return ExplorationResult(
                status=ExplorationStatus.SUCCESS,
                final_structure=str(next_input_file.resolve()),
                metadata={"is_restart": False, "barrier": kmc_result.metadata.get("barrier")}
            )

        elif kmc_result.status == KMCStatus.UNCERTAIN:
             logger.info("KMC Explorer encountered uncertain Transition State candidate.")
             # Epic 6: TS Structure Rescue
             # The KMC engine returns the saddle point (or uncertain state) in result.structure
             # We pass this up as 'uncertain_structures'.
             # The orchestrator's loop (ActiveLearningOrchestrator) should pick this up
             # and call ALService.trigger_al() on it.
             return ExplorationResult(
                status=ExplorationStatus.UNCERTAIN,
                uncertain_structures=[kmc_result.structure],
                metadata={"reason": kmc_result.metadata.get("reason", "Unknown")}
            )

        elif kmc_result.status == KMCStatus.NO_EVENT:
            return ExplorationResult(
                status=ExplorationStatus.NO_EVENT,
                final_structure=current_structure,
                metadata={"is_restart": True}
            )

        return ExplorationResult(status=ExplorationStatus.FAILED)
