"""Orchestrator for the Active Learning Loop.

This module coordinates the interaction between MD, KMC, and Active Learning services.
"""

import os
import logging
import random
import shutil
import yaml
from pathlib import Path
from typing import Optional, List
from ase.io import read, write

from shared.core.config import Config
from shared.core.enums import KMCStatus
from shared.utils.logger import CSVLogger

from src.state_manager import StateManager, Stage
from src.services.al_service import ActiveLearningService
from src.interfaces.explorer import BaseExplorer, ExplorationStatus

logger = logging.getLogger(__name__)

class HandledOrchestratorError(RuntimeError):
    """Exception raised for handled errors in the orchestrator that should stop execution cleanly."""
    pass

class ActiveLearningOrchestrator:
    """Manages the active learning loop using specialized services."""

    def __init__(
        self,
        config: Config,
        al_service: ActiveLearningService,
        explorer: BaseExplorer,
        state_manager: StateManager,
        csv_logger: Optional[CSVLogger] = None,
        simulation_config_path: Optional[Path] = None,
        al_config_path: Optional[Path] = None
    ):
        self.config = config
        self.al_service = al_service
        self.explorer = explorer
        self.state_manager = state_manager
        self.csv_logger = csv_logger or CSVLogger()
        self.simulation_config_path = simulation_config_path
        self.al_config_path = al_config_path

        # Merge configs if provided
        self._merge_configs()

    def _merge_configs(self):
        """Merges step-specific configs into the main config object."""
        if self.simulation_config_path and self.simulation_config_path.exists():
            logger.info(f"Loading simulation config from {self.simulation_config_path}")
            with open(self.simulation_config_path, 'r') as f:
                sim_data = yaml.safe_load(f) or {}
                # Update md_params
                target_data = sim_data.get('md_params', sim_data)
                self._update_object(self.config.md_params, target_data)

        if self.al_config_path and self.al_config_path.exists():
            logger.info(f"Loading AL config from {self.al_config_path}")
            with open(self.al_config_path, 'r') as f:
                al_data = yaml.safe_load(f) or {}
                # Update al_params
                target_al = al_data.get('al_params', al_data)
                self._update_object(self.config.al_params, target_al)

                # Update training_params if present in AL config
                target_training = al_data.get('training_params', al_data.get('training', {}))
                if target_training:
                     self._update_object(self.config.training_params, target_training)

    def _update_object(self, obj, data_dict):
        """Updates object attributes from a dictionary."""
        for k, v in data_dict.items():
            setattr(obj, k, v)

    def run(self):
        """Executes the active learning loop (Hybrid MD-kMC)."""

        # Define Work Directory for Step 7
        work_root = Path("work/07_active_learning").resolve()
        work_root.mkdir(parents=True, exist_ok=True)

        # Initialize Monitoring (W&B)
        wandb_run = None
        if self.config.monitoring.enabled and self.config.monitoring.use_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=self.config.monitoring.project,
                    entity=self.config.monitoring.entity,
                    config=self.config.model_dump(),
                    name=self.config.experiment.name,
                    resume="allow"
                )
                logger.info("Weights & Biases monitoring initialized.")
            except ImportError:
                logger.warning("wandb not installed. Monitoring disabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        # Load State
        state = self.state_manager.load()
        iteration = state.get("iteration", 0)

        current_potential = state.get("current_potential") or self.config.al_params.initial_potential
        current_asi = state.get("current_asi") or self.config.al_params.initial_active_set_path
        current_structure = state.get("current_structure") or self.config.md_params.initial_structure
        is_restart = state.get("is_restart", False)
        al_consecutive_counter = state.get("al_consecutive_counter", 0)
        current_stage = state.get("current_stage", Stage.EXPLORATION)
        uncertain_structures_path = state.get("uncertain_structures_path", None)

        original_cwd = Path.cwd()

        # Resolve paths
        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        # Check if AL is active
        is_al_active = self.al_config_path is not None and self.al_config_path.exists()

        # Initialize Active Set if needed
        if is_al_active and not current_asi and initial_dataset_path and iteration == 0:
             logger.info("Generating initial Active Set...")
             try:
                 # Generate into seed directory logic could be here, but we just trigger update
                 current_asi = self.al_service.trainer.update_active_set(str(initial_dataset_path), str(potential_yaml_path))
                 state["current_asi"] = current_asi
             except Exception as e:
                 logger.error(f"Failed to generate initial active set: {e}")
                 return

        error_counter = 0
        MAX_CONSECUTIVE_ERRORS = 5

        while True:
            # Check Resume Logic
            if current_stage == Stage.EXPLORATION:
                iteration += 1
            else:
                logger.info(f"Resuming Iteration {iteration} at stage {current_stage}")

            work_dir = work_root / f"iteration_{iteration}"
            work_dir.mkdir(parents=True, exist_ok=True)

            # Save State (at start of iteration or resume)
            state.update({
                "iteration": iteration,
                "current_potential": current_potential,
                "current_structure": current_structure,
                "current_asi": current_asi,
                "is_restart": is_restart,
                "al_consecutive_counter": al_consecutive_counter,
                "current_stage": current_stage,
                "uncertain_structures_path": uncertain_structures_path
            })
            self.state_manager.save(state)

            logger.info(f"--- Starting Iteration {iteration} (Stage: {current_stage}, AL Retries: {al_consecutive_counter}) ---")

            try:
                abs_potential_path = self._resolve_path(current_potential, original_cwd)
                abs_asi_path = self._resolve_path(current_asi, original_cwd) if current_asi else None

                if not abs_potential_path.exists():
                    logger.error(f"Initial potential file not found: {abs_potential_path}")
                    logger.error("The 'start_loop' command requires a pre-existing potential file.")
                    logger.error("To fix this, either run a seed generation workflow to create the initial potential, or provide a valid path in your config's 'al_params.initial_potential' field.")
                    raise HandledOrchestratorError("Initial potential not found, cannot start active learning loop.")

                uncertain_structures = []

                # --- Exploration Phase ---
                if current_stage == Stage.EXPLORATION:
                    input_structure_arg = self._prepare_structure_path(
                        is_restart, iteration, current_structure, original_cwd, work_root
                    )
                    if not input_structure_arg:
                        break

                    # --- Systematic Deformation Injection ---
                    def_freq = getattr(self.config.al_params, 'deformation_frequency', 0)
                    if is_al_active and iteration > 0 and def_freq > 0 and iteration % def_freq == 0:
                        self.al_service.inject_deformation_data(input_structure_arg)

                    exploration_result = self.explorer.explore(
                        current_structure=input_structure_arg,
                        potential_path=str(abs_potential_path),
                        iteration=iteration,
                        is_restart=is_restart,
                        work_dir=work_dir
                    )

                    if exploration_result.status == ExplorationStatus.FAILED:
                        logger.error("Exploration failed.")
                        break

                    if is_al_active and exploration_result.status == ExplorationStatus.UNCERTAIN:
                        # Prepare for AL
                        logger.info("Exploration detected uncertainty. Preparing for AL.")

                        uncertain_structures = exploration_result.uncertain_structures

                        # Save uncertain structures
                        uncertain_path = work_dir / "uncertain_structures.xyz"
                        write(str(uncertain_path), uncertain_structures)
                        uncertain_structures_path = str(uncertain_path)

                        # Update State to LABELING
                        current_stage = Stage.LABELING
                        state.update({
                            "current_stage": current_stage,
                            "uncertain_structures_path": uncertain_structures_path
                        })
                        self.state_manager.save(state)

                    else:
                        # Success or No Event
                        logger.info(f"Exploration phase completed with status: {exploration_result.status}")
                        # Update state for next iteration
                        if exploration_result.final_structure:
                            restart_path = work_dir / "final_structure.xyz"
                            write(str(restart_path), exploration_result.final_structure)
                            current_structure = str(restart_path.resolve())

                        is_restart = exploration_result.metadata.get("is_restart", False)
                        al_consecutive_counter = 0
                        # Stage remains EXPLORATION for next iteration
                        continue

                # --- AL Phase (Labeling & Training) ---
                if current_stage == Stage.LABELING:
                    # Load uncertain structures if empty (resume case)
                    if not uncertain_structures and uncertain_structures_path:
                        try:
                            uncertain_structures = read(uncertain_structures_path, index=":")
                            logger.info(f"Loaded {len(uncertain_structures)} uncertain structures from {uncertain_structures_path}")
                        except Exception as e:
                            logger.error(f"Failed to load uncertain structures from {uncertain_structures_path}: {e}")
                            # Fallback
                            logger.warning("Falling back to Exploration due to missing data.")
                            current_stage = Stage.EXPLORATION
                            uncertain_structures_path = None
                            continue

                    max_retries = getattr(self.config.al_params, 'max_al_retries', 3)
                    if al_consecutive_counter >= max_retries:
                        raise HandledOrchestratorError("Max AL retries reached. Infinite loop detected.")

                    # Use new signature with metrics
                    new_pot, metrics = self.al_service.trigger_al(
                        uncertain_structures,
                        abs_potential_path,
                        potential_yaml_path,
                        abs_asi_path,
                        work_dir,
                        iteration
                    )

                    # Log metrics to W&B
                    if wandb_run and metrics:
                        metrics["iteration"] = iteration
                        wandb_run.log(metrics)

                    # Log to CSV (legacy/local)
                    if self.csv_logger and metrics:
                         self.csv_logger.log_metrics(
                             iteration=iteration,
                             max_gamma=metrics.get("uncertainty_max", 0.0),
                             n_added=metrics.get("num_new_candidates", 0),
                             active_set_size=metrics.get("active_set_size", 0), # Need to check if available
                             rmse_energy=metrics.get("train_rmse_energy"),
                             rmse_forces=metrics.get("train_rmse_force")
                         )

                    if new_pot:
                        current_potential = str(Path(new_pot).resolve())
                        is_restart = True

                        new_asi = work_dir / "potential.asi"
                        if new_asi.exists():
                            current_asi = str(new_asi.resolve())
                        al_consecutive_counter += 1

                        # AL Done -> Back to Exploration
                        current_stage = Stage.EXPLORATION
                        uncertain_structures_path = None

                        # Save state at end of AL
                        state.update({
                            "current_potential": current_potential,
                            "current_asi": current_asi,
                            "is_restart": is_restart,
                            "al_consecutive_counter": al_consecutive_counter,
                            "current_stage": current_stage,
                            "uncertain_structures_path": uncertain_structures_path
                        })
                        self.state_manager.save(state)
                        continue
                    else:
                        logger.warning("AL triggered but no new potential returned.")
                        break

                # Reset error counter on successful iteration
                error_counter = 0

            except Exception as e:
                # If it's a HandledOrchestratorError, re-raise it so the caller can handle or the loop exits
                if isinstance(e, HandledOrchestratorError):
                    raise e

                error_counter += 1
                logger.exception(f"An error occurred in iteration {iteration}: {e}")
                if error_counter < MAX_CONSECUTIVE_ERRORS:
                    logger.info(f"Retrying... (Error count: {error_counter})")
                    continue
                else:
                    logger.error("Max errors reached")
                    raise RuntimeError("Max consecutive errors reached in Orchestrator") from e

        # Finish W&B run if loop exits
        if wandb_run:
            wandb_run.finish()

    def _find_latest_restart(self, iteration: int, work_root: Path, max_search: int = 5) -> Optional[Path]:
        for i in range(1, max_search + 1):
            search_iter = iteration - i
            if search_iter < 1:
                break
            prev_dir = work_root / f"iteration_{search_iter}"
            restart_file = prev_dir / "restart.chk"
            if restart_file.exists():
                return restart_file
        return None

    def _resolve_path(self, path_str: str, base_cwd: Path) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (base_cwd / p).resolve()

    def _prepare_structure_path(
        self, is_restart: bool, iteration: int, initial_structure: str, base_cwd: Path, work_root: Path
    ) -> Optional[str]:
        if not is_restart and initial_structure:
             return str(self._resolve_path(initial_structure, base_cwd))

        if is_restart:
            restart_file = self._find_latest_restart(iteration, work_root)

            if not restart_file:
                 logger.error(f"Restart file missing for resume (searched backwards from {iteration})")
                 return None
            return str(restart_file)
        else:
            return str(self._resolve_path(initial_structure, base_cwd))
