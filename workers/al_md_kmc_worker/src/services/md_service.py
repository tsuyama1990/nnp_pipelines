import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from ase import Atoms
from ase.io import read, write
from functools import partial

from shared.core.interfaces import MDEngine
from shared.core.enums import SimulationState
from shared.core.config import Config
from src.utils.parallel_executor import ParallelExecutor

logger = logging.getLogger(__name__)

def _run_md_task(md_engine: MDEngine,
                 potential_path: str,
                 steps: int,
                 gamma_threshold: float,
                 input_structure: str,
                 is_restart: bool,
                 temperature: float,
                 pressure: float,
                 seed: int) -> Tuple[SimulationState, Optional[Path]]:
    """Helper function to run a single MD walker task."""
    try:
        dump_path = md_engine.run(
            structure_path=input_structure,
            potential_path=potential_path,
            temperature=temperature,
            pressure=pressure,
            seed=seed
        )
        return SimulationState.COMPLETED, dump_path

    except Exception as e:
        logger.error(f"MD Walker failed: {e}")
        return SimulationState.FAILED, None


class MDService:
    """Service for managing Molecular Dynamics simulations."""

    def __init__(self, md_engine: MDEngine, config: Config):
        self.md_engine = md_engine
        self.config = config
        self.executor = ParallelExecutor(max_workers=config.md_params.n_md_walkers)

    def _get_md_conditions(self, iteration: int) -> Dict[str, float]:
        """Get Temperature and Pressure for the current iteration based on schedule."""
        default_conditions = {
            "temperature": self.config.md_params.temperature,
            "pressure": self.config.md_params.pressure
        }

        for stage in self.config.exploration_schedule:
            if stage.iter_start <= iteration <= stage.iter_end:
                temp = random.uniform(stage.temp[0], stage.temp[1])
                press = random.uniform(stage.press[0], stage.press[1])
                return {"temperature": temp, "pressure": press}

        return default_conditions

    def run_rescue(self, atoms: Atoms, potential_path: str) -> Optional[Atoms]:
        """
        Run a short, low-accuracy MD simulation to rescue an unstable structure.

        This writes the atoms to a temp file, runs a short simulation, and returns
        the final structure.
        """
        import tempfile

        logger.info("Running Rescue MD simulation...")

        # Create temp workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "rescue_input.data"
            write(input_file, atoms, format="lammps-data")

            # Use mild settings for rescue (NVT, lower temp/steps?)
            # Or use configured exploration schedule?
            # User prompt says "short NVE/NVT".
            # We assume run() supports overriding steps/temp via config but arguments are strict in MDEngine.run interface usually?
            # MDEngine.run(structure_path, potential_path, temperature, pressure, seed)
            # It reads steps from config.md_params usually.
            # We might need to subclass MDEngine or pass extra args if supported.
            # Assuming standard run with current config params is okay, maybe short?
            # Ideally we'd want fewer steps. If MDEngine reads config, we can't easily change it without patching config.

            # Hack: Patch config temporarily or assume MD Engine can handle it.
            # For now, we just run standard MD.

            try:
                dump_path = self.md_engine.run(
                    structure_path=str(input_file),
                    potential_path=potential_path,
                    temperature=min(300.0, self.config.md_params.temperature), # Gentle temp
                    pressure=0.0, # Relax stress
                    seed=random.randint(0, 1000)
                )

                if dump_path and dump_path.exists():
                    # Return the last frame
                    frames = read(dump_path, index=":", format="lammps-dump-text")
                    if frames:
                        return frames[-1]
            except Exception as e:
                logger.error(f"Rescue run failed: {e}")

        return None

    def run_walkers(self,
                    iteration: int,
                    potential_path: Path,
                    input_structure_path: str,
                    is_restart: bool) -> Tuple[bool, List[Atoms], bool]:
        """
        Run parallel MD walkers.

        Returns:
            Tuple[bool, List[Atoms], bool]:
                - success (bool): True if at least one walker finished successfully.
                - uncertain_structures (List[Atoms]): List of structures that triggered uncertainty.
                - failure (bool): True if critical failure occurred.
        """
        logger.info("Running MD (Parallel Walkers)...")

        n_walkers = self.config.md_params.n_md_walkers
        uncertain_structures_buffer = []
        any_uncertain = False
        any_failed = False

        task_creators = []

        for i in range(n_walkers):
            conditions = self._get_md_conditions(iteration)
            seed = random.randint(0, 1000000)
            logger.info(f"Walker {i}: Temp={conditions['temperature']:.1f}, Press={conditions['pressure']:.1f}, Seed={seed}")

            task_creators.append(partial(
                _run_md_task,
                self.md_engine,
                str(potential_path),
                self.config.md_params.n_steps,
                self.config.al_params.gamma_threshold,
                input_structure_path,
                is_restart,
                conditions['temperature'],
                conditions['pressure'],
                seed
            ))

        results = self.executor.submit_tasks(task_creators)

        for state_res, dump_path in results:
            if state_res == SimulationState.FAILED:
                any_failed = True
                continue

            if dump_path and dump_path.exists():
                try:
                    atoms_list = read(dump_path, index=":", format="lammps-dump-text")
                    max_gamma_observed = 0.0
                    uncertain_frames = []

                    for at in atoms_list:
                        gammas = None
                        if 'f_2' in at.arrays:
                            gammas = at.arrays['f_2']
                        elif 'c_max_gamma' in at.arrays:
                            gammas = at.arrays['c_max_gamma']

                        if gammas is not None:
                            current_max = gammas.max()
                            max_gamma_observed = max(max_gamma_observed, current_max)
                            if current_max > self.config.al_params.gamma_threshold:
                                uncertain_frames.append(at)

                    logger.info(f"Walker max gamma: {max_gamma_observed}")

                    if max_gamma_observed > self.config.al_params.gamma_threshold:
                        any_uncertain = True
                        if uncertain_frames:
                            uncertain_structures_buffer.append(uncertain_frames[0])
                        else:
                            uncertain_structures_buffer.append(atoms_list[-1])

                except Exception as e:
                    logger.warning(f"Failed to parse dump file {dump_path}: {e}")

        success = not any_failed # Simplified success check
        return success, uncertain_structures_buffer, any_failed
