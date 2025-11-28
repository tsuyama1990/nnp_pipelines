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
from orchestrator.src.services.al_service import ActiveLearningService
from orchestrator.src.interfaces.explorer import BaseExplorer, ExplorationStatus

logger = logging.getLogger(__name__)

class ActiveLearningOrchestrator:
    """Manages the active learning loop using specialized services."""

    def __init__(
        self,
        config: Config,
        al_service: ActiveLearningService,
        explorer: BaseExplorer,
        state_manager: StateManager,
        csv_logger: Optional[CSVLogger] = None
    ):
        self.config = config
        self.al_service = al_service
        self.explorer = explorer
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

                # --- Exploration Phase ---
                exploration_result = self.explorer.explore(
                    current_structure=input_structure_arg,
                    potential_path=str(abs_potential_path),
                    iteration=iteration,
                    is_restart=is_restart
                )

                if exploration_result.status == ExplorationStatus.FAILED:
                    logger.error("Exploration failed.")
                    break

                if exploration_result.status == ExplorationStatus.UNCERTAIN:
                    if al_consecutive_counter >= 3:
                        raise RuntimeError("Max AL retries reached. Infinite loop detected.")

                    logger.info("Exploration detected uncertainty. Triggering AL.")

                    new_pot = self.al_service.trigger_al(
                        exploration_result.uncertain_structures,
                        abs_potential_path,
                        potential_yaml_path,
                        abs_asi_path,
                        work_dir,
                        iteration
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

                # Success or No Event
                logger.info(f"Exploration phase completed with status: {exploration_result.status}")
                al_consecutive_counter = 0

                if exploration_result.final_structure:
                    current_structure = exploration_result.final_structure

                # Update restart state based on explorer metadata
                is_restart = exploration_result.metadata.get("is_restart", False)

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
