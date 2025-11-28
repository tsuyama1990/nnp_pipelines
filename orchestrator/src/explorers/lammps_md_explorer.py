import logging
from pathlib import Path
from typing import Dict, List, Optional
from ase import Atoms

from orchestrator.src.interfaces.explorer import BaseExplorer, ExplorationResult, ExplorationStatus
from orchestrator.src.services.md_service import MDService
from shared.core.config import Config

logger = logging.getLogger(__name__)

class LammpsMDExplorer(BaseExplorer):
    def __init__(self, md_service: MDService, config: Config):
        self.md_service = md_service
        self.config = config

    def explore(self,
                current_structure: str,
                potential_path: str,
                iteration: int,
                is_restart: bool = False,
                **kwargs) -> ExplorationResult:

        md_success, uncertain_structures, md_failed = self.md_service.run_walkers(
            iteration,
            Path(potential_path),
            current_structure,
            is_restart
        )

        if md_failed:
            return ExplorationResult(status=ExplorationStatus.FAILED)

        if uncertain_structures:
             return ExplorationResult(
                status=ExplorationStatus.UNCERTAIN,
                uncertain_structures=uncertain_structures
            )

        # If successful, we need to find the final structure.
        # MDService doesn't return the final structure path directly in run_walkers return value.
        # But it logs it or saves it.
        # In the original code:
        # "If MD was successful and stable"
        # The Orchestrator loop would then proceed to KMC or just restart.

        # We need to identify the output structure.
        # For now, we can assume the final structure is implicitly handled or we need to extract it.
        # The MDService saves "dump.lammpstrj.{seed}".
        # We might need to select one.

        # In the original code, if MD is successful:
        # if self.config.kmc_params.active: ...
        # else: ...

        # The `explore` method should probably return a structure that can be used for the next step.
        # Since run_walkers runs multiple walkers, we might pick one randomly as the "result"
        # for continuity if we are chaining explorers, or let the caller handle it.

        # However, looking at MDService, it produces dump files in the current directory.
        # The orchestrator looks for `restart.chk.*`.

        work_dir = Path.cwd()
        chk_files = list(work_dir.glob("restart.chk.*"))
        final_structure = None

        if chk_files:
            import random
            import shutil

            selected_chk = random.choice(chk_files)
            selected_seed = selected_chk.suffix.split('.')[-1]

            # 1. Update restart state for continuity (e.g. next iteration resume)
            shutil.copy(selected_chk, work_dir / "restart.chk")

            # 2. Identify the readable structure (dump) for downstream tools (like KMC)
            # The original code used dump.lammpstrj.{seed}
            selected_dump = work_dir / f"dump.lammpstrj.{selected_seed}"

            if selected_dump.exists():
                final_structure = str(selected_dump.resolve())
            else:
                # Fallback or error?
                # If dump is missing but restart exists, we might be in trouble for KMC reading
                # unless we have a tool to convert restart -> data.
                # For now, let's warn and return None or the chk path (which will fail later if read is attempted)
                logger.warning(f"Dump file for seed {selected_seed} not found. Downstream explorers might fail.")
                final_structure = str(selected_chk)

        return ExplorationResult(
            status=ExplorationStatus.SUCCESS,
            final_structure=final_structure,
            metadata={"is_restart": True} # Hint for next step (resume from restart.chk)
        )
