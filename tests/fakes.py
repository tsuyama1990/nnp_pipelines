import sys
import os
worker_path = os.path.join(os.getcwd(), 'workers/al_md_kmc_worker')
if worker_path not in sys.path:
    sys.path.append(worker_path)

from typing import Optional, List
from pathlib import Path
from ase import Atoms
from ase.io import write
from shared.core.interfaces import MDEngine, StructureGenerator, Labeler, Trainer
from src.interfaces.explorer import BaseExplorer, ExplorationResult, ExplorationStatus
from src.state_manager import StateManager

class FakeExplorer(BaseExplorer):
    def __init__(self, should_fail: bool = False, uncertainty: bool = False, max_calls: int = 1):
        self.should_fail = should_fail
        self.uncertainty = uncertainty
        self.call_count = 0
        self.max_calls = max_calls

    def explore(
        self,
        current_structure: str,
        potential_path: str,
        iteration: int,
        is_restart: bool,
        work_dir: Path
    ) -> ExplorationResult:
        # Check inputs
        if not Path(current_structure).exists():
            raise FileNotFoundError(f"Input structure not found: {current_structure}")
        if not Path(potential_path).exists():
            raise FileNotFoundError(f"Potential file not found: {potential_path}")

        self.call_count += 1

        if self.should_fail:
             return ExplorationResult(status=ExplorationStatus.FAILED, metadata={})

        if self.call_count > self.max_calls:
             # Stop the loop by failing or returning failed status
             return ExplorationResult(status=ExplorationStatus.FAILED, metadata={})

        # Create dummy output
        output_file = work_dir / "final_structure.xyz"
        dummy_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
        write(str(output_file), dummy_atoms)

        if self.uncertainty:
             uncertain_struct = work_dir / "uncertain.xyz"
             write(str(uncertain_struct), dummy_atoms)
             return ExplorationResult(
                 status=ExplorationStatus.UNCERTAIN,
                 uncertain_structures=[str(uncertain_struct)],
                 metadata={"is_restart": False}
             )

        return ExplorationResult(
            status=ExplorationStatus.SUCCESS,
            final_structure=dummy_atoms,
            metadata={"is_restart": True}
        )

class FakeTrainer:
    def update_active_set(self, dataset_path, potential_path):
        return "fake.asi"

class FakeALService:
    def __init__(self):
        self.trainer = FakeTrainer()

    def inject_deformation_data(self, structure_path: str):
        pass

    def trigger_al(self, uncertain_structures, potential_path, potential_yaml, asi_path, work_dir, iteration):
        # Simulate new potential generation
        new_pot = work_dir / "new_potential.yace"
        new_pot.touch()
        return str(new_pot)
