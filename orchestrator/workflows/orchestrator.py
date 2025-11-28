"""Orchestrator for the Active Learning Loop.

This module coordinates the interaction between MD, KMC, and Active Learning services.
"""

import os
import logging
import random
import shutil
from pathlib import Path
from typing import Optional
from ase.io import read, write

from shared.core.config import Config
from shared.core.enums import KMCStatus
from shared.utils.logger import CSVLogger

from orchestrator.src.state_manager import StateManager
from orchestrator.src.services.md_service import MDService
from orchestrator.src.services.al_service import ActiveLearningService
from orchestrator.src.services.kmc_service import KMCService

logger = logging.getLogger(__name__)

class ActiveLearningOrchestrator:
    """Manages the active learning loop using specialized services."""

    def __init__(
        self,
        config: Config,
        md_service: MDService,
        al_service: ActiveLearningService,
        kmc_service: KMCService,
        state_manager: StateManager,
        csv_logger: Optional[CSVLogger] = None
    ):
        self.config = config
        self.md_service = md_service
        self.al_service = al_service
        self.kmc_service = kmc_service
        self.state_manager = state_manager
        self.csv_logger = csv_logger or CSVLogger()

    def run(self):
        """Executes the active learning loop (Hybrid MD-kMC)."""

        data_root = Path("data")
        data_root.mkdir(parents=True, exist_ok=True)

        # Load State
        state = self.state_manager.load()
        iteration = state["iteration"]
        current_potential = state.get("current_potential") or self.config.al_params.initial_potential
        current_asi = state.get("current_asi") or self.config.al_params.initial_active_set_path
        current_structure = state.get("current_structure") or self.config.md_params.initial_structure
        is_restart = state.get("is_restart", False)
        al_consecutive_counter = state.get("al_consecutive_counter", 0)

        original_cwd = Path.cwd()

        # Resolve paths
        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        # Initialize Active Set if needed
        if not current_asi and initial_dataset_path and iteration == 0:
             logger.info("Generating initial Active Set...")
             try:
                 init_dir = original_cwd / "data" / "seed"
                 init_dir.mkdir(parents=True, exist_ok=True)
                 os.chdir(init_dir)
                 current_asi = self.al_service.trainer.update_active_set(str(initial_dataset_path), str(potential_yaml_path))
                 state["current_asi"] = current_asi
                 os.chdir(original_cwd)
             except Exception as e:
                 logger.error(f"Failed to generate initial active set: {e}")
                 os.chdir(original_cwd)
                 return

        while True:
            iteration += 1

            # Save State
            state.update({
                "iteration": iteration,
                "current_potential": current_potential,
                "current_structure": current_structure,
                "current_asi": current_asi,
                "is_restart": is_restart,
                "al_consecutive_counter": al_consecutive_counter
            })
            self.state_manager.save(state)

            work_dir = data_root / f"iteration_{iteration}"
            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} (AL Retries: {al_consecutive_counter}) ---")

            try:
                os.chdir(work_dir)

                abs_potential_path = self._resolve_path(current_potential, original_cwd)
                abs_asi_path = self._resolve_path(current_asi, original_cwd) if current_asi else None

                if not abs_potential_path.exists():
                    logger.error(f"Potential file not found: {abs_potential_path}")
                    break

                input_structure_arg = self._prepare_structure_path(
                    is_restart, iteration, current_structure, original_cwd
                )
                if not input_structure_arg:
                    break

                # --- Systematic Deformation Injection ---
                if iteration > 0 and iteration % 5 == 0:
                    self.al_service.inject_deformation_data(input_structure_arg)

                # --- MD Phase ---
                md_success, uncertain_structures, md_failed = self.md_service.run_walkers(
                    iteration, abs_potential_path, input_structure_arg, is_restart
                )

                if md_failed:
                    logger.error("MD Failed in one or more walkers.")
                    break

                if uncertain_structures:
                    if al_consecutive_counter >= 3:
                        raise RuntimeError("Max AL retries reached. Infinite loop detected.")

                    logger.info("MD detected uncertainty. Triggering AL.")

                    new_pot = self.al_service.trigger_al(
                        uncertain_structures, abs_potential_path, potential_yaml_path, abs_asi_path, work_dir, iteration
                    )

                    if new_pot:
                        current_potential = str(Path(new_pot).resolve())
                        is_restart = True
                        new_asi = work_dir / "potential.asi"
                        if new_asi.exists():
                            current_asi = str(new_asi.resolve())
                        al_consecutive_counter += 1
                        continue
                    else:
                        break

                # If MD was successful and stable
                logger.info("MD walkers completed stably.")
                al_consecutive_counter = 0

                # --- KMC Phase ---
                if self.config.kmc_params.active:
                    chk_files = list(work_dir.glob("restart.chk.*"))
                    if not chk_files:
                        logger.error("No restart files found for KMC.")
                        break

                    selected_chk = random.choice(chk_files)
                    selected_seed = selected_chk.suffix.split('.')[-1]
                    logger.info(f"Selected walker seed {selected_seed} for KMC.")

                    shutil.copy(selected_chk, work_dir / "restart.chk") # For next iter restart

                    selected_dump = work_dir / f"dump.lammpstrj.{selected_seed}"
                    if not selected_dump.exists():
                        logger.error("No structure available for KMC.")
                        break

                    kmc_input_atoms = read(selected_dump, index=-1, format="lammps-dump-text")
                    self.al_service.ensure_chemical_symbols(kmc_input_atoms)

                    kmc_result = self.kmc_service.run_step(kmc_input_atoms, str(abs_potential_path))

                    if kmc_result.status == KMCStatus.SUCCESS:
                        next_input_file = work_dir / "kmc_output.data"
                        write(next_input_file, kmc_result.structure, format="lammps-data", velocities=True)
                        current_structure = str(next_input_file.resolve())
                        is_restart = False

                    elif kmc_result.status == KMCStatus.UNCERTAIN:
                        logger.info("KMC detected uncertainty. Triggering AL.")
                        if al_consecutive_counter >= 3:
                            raise RuntimeError("Max AL retries reached in KMC.")

                        new_pot = self.al_service.trigger_al(
                            [kmc_result.structure], abs_potential_path, potential_yaml_path, abs_asi_path, work_dir, iteration
                        )
                        if new_pot:
                            current_potential = str(Path(new_pot).resolve())
                            new_asi = work_dir / "potential.asi"
                            if new_asi.exists():
                                current_asi = str(new_asi.resolve())
                            is_restart = True
                            al_consecutive_counter += 1
                            continue
                        else:
                            break

                    elif kmc_result.status == KMCStatus.NO_EVENT:
                         is_restart = True

                else:
                    # Pure MD
                    chk_files = list(work_dir.glob("restart.chk.*"))
                    if chk_files:
                        selected_chk = random.choice(chk_files)
                        shutil.copy(selected_chk, work_dir / "restart.chk")
                    is_restart = True

            except Exception as e:
                logger.exception(f"An error occurred in iteration {iteration}: {e}")
                break
            finally:
                os.chdir(original_cwd)

    def _resolve_path(self, path_str: str, base_cwd: Path) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (base_cwd / p).resolve()

    def _prepare_structure_path(
        self, is_restart: bool, iteration: int, initial_structure: str, base_cwd: Path
    ) -> Optional[str]:
        if not is_restart and initial_structure and initial_structure.endswith(".data"):
             return str(self._resolve_path(initial_structure, base_cwd))

        if is_restart:
            search_start = iteration - 1
            if search_start < 1:
                pass # Logic from old code

            prev_dir = base_cwd / "data" / f"iteration_{search_start}"
            restart_file = prev_dir / "restart.chk"

            if not restart_file.exists():
                 logger.error(f"Restart file missing for resume: {restart_file}")
                 return None
            return str(restart_file)
        else:
            return str(self._resolve_path(initial_structure, base_cwd))
