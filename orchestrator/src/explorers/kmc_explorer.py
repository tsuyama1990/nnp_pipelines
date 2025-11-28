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

        # KMC usually starts from a previous simulation state (e.g. restart.chk or dump)
        # The Orchestrator logic for KMC selection was:
        # 1. Select restart.chk.*
        # 2. Select dump.lammpstrj.{seed}
        # 3. Read structure
        # 4. Run KMC

        # If `current_structure` is passed, we should use it.
        # However, the previous MD step might have produced multiple outputs.
        # If this Explorer is run standalone, `current_structure` is the start.

        work_dir = Path.cwd()

        # Check if current_structure is valid
        p = Path(current_structure)
        if not p.exists():
            return ExplorationResult(status=ExplorationStatus.FAILED, metadata={"error": "Structure not found"})

        # If it's a restart.chk file, KMC Service expects Atoms object.
        # We need to read the atoms.
        # But wait, KMCService.run_step takes `initial_atoms: Atoms`.

        # If `current_structure` points to a restart file, we can't easily read it with ASE usually unless it's a supported format.
        # The Orchestrator used `dump.lammpstrj.{seed}` corresponding to the restart file.

        # Let's assume `current_structure` passed here is readable by ASE (e.g. data or dump or traj).
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
                metadata={"is_restart": False}
            )

        elif kmc_result.status == KMCStatus.UNCERTAIN:
             return ExplorationResult(
                status=ExplorationStatus.UNCERTAIN,
                uncertain_structures=[kmc_result.structure]
            )

        elif kmc_result.status == KMCStatus.NO_EVENT:
            return ExplorationResult(
                status=ExplorationStatus.NO_EVENT,
                final_structure=current_structure, # Stay where we are?
                metadata={"is_restart": True}
            )

        return ExplorationResult(status=ExplorationStatus.FAILED)
