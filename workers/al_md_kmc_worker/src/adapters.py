import os
import shutil
import uuid
import pickle
import json
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
from ase import Atoms
from ase.io import read, write

from shared.core.interfaces import MDEngine, Sampler, StructureGenerator, Labeler, Trainer, KMCResult, KMCEngine, Validator
from shared.core.enums import SimulationState
from .wrappers.lammps_wrapper import LammpsWorker
from .wrappers.dft_wrapper import DftWorker
from .wrappers.pace_wrapper import PaceWorker
from .wrappers.gen_wrapper import GenWorker

# Helper to generate unique filenames in data dir
def get_unique_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"

class DockerMDEngine(MDEngine):
    def __init__(self, wrapper: LammpsWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename

    def run(self, potential_path: str, steps: int, gamma_threshold: float,
            input_structure: str, is_restart: bool = False) -> SimulationState:

        pot_name = Path(potential_path).name
        struct_name = Path(input_structure).name

        try:
            self.wrapper.run_md(
                config_filename=self.config_filename,
                meta_config_filename=self.meta_config_filename,
                potential_filename=pot_name,
                structure_filename=struct_name,
                steps=steps,
                gamma=gamma_threshold,
                restart=is_restart
            )
            return SimulationState.COMPLETED
        except Exception as e:
            return SimulationState.FAILED

class DockerLabeler(Labeler):
    def __init__(self, wrapper: DftWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename
        self.host_data_dir = wrapper.host_work_dir # Corrected from host_data_dir

    def label(self, structure: Atoms) -> Optional[Atoms]:
        input_name = get_unique_filename("label_input", ".xyz")
        output_name = get_unique_filename("label_output", ".xyz")

        input_path = self.host_data_dir / input_name
        output_path = self.host_data_dir / output_name

        write(input_path, structure)

        try:
            self.wrapper.label(
                config_filename=self.config_filename,
                meta_config_filename=self.meta_config_filename,
                structure_filename=input_name,
                output_filename=output_name
            )

            if output_path.exists():
                return read(output_path)
            return None
        except Exception:
            return None
        finally:
            if input_path.exists():
                os.remove(input_path)
            if output_path.exists():
                os.remove(output_path)

class DockerTrainer(Trainer):
    def __init__(self, wrapper: PaceWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename

    def prepare_dataset(self, structures: List[Atoms]) -> str:
        import pandas as pd
        df = pd.DataFrame({"ase_atoms": structures})
        fname = get_unique_filename("training_data", ".pckl.gzip")
        path = self.wrapper.host_data_dir / fname
        df.to_pickle(path, compression="gzip")
        return str(path)

    def update_active_set(self, dataset_path: str, potential_yaml_path: str) -> str:
        # Placeholder or assume updated via training side effects
        return "potential.asi"

    def train(self, dataset_path: str, initial_potential: str, **kwargs) -> str:
        ds_name = Path(dataset_path).name
        pot_name = Path(initial_potential).name if initial_potential else None

        pot_yaml = Path(kwargs.get("potential_yaml_path", "")).name if kwargs.get("potential_yaml_path") else None
        asi = Path(kwargs.get("asi_path", "")).name if kwargs.get("asi_path") else None
        iteration = kwargs.get("iteration", 0)

        result_name = self.wrapper.train(
            config_filename=self.config_filename,
            meta_config_filename=self.meta_config_filename,
            dataset_filename=ds_name,
            initial_potential=pot_name,
            potential_yaml=pot_yaml,
            asi=asi,
            iteration=iteration
        )
        return str(self.wrapper.host_data_dir / result_name)

class DockerSampler(Sampler):
    def __init__(self, wrapper: PaceWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename
        self.host_data_dir = wrapper.host_work_dir # Corrected to host_work_dir or handle absence

    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        candidates = kwargs.get("structures", []) # API changed in orchestrator call? Check orchestrator logic.
        # Orchestrator uses: sample(structures=[...], ...)
        if not candidates:
             # Try other key if API varies
             candidates = kwargs.get("candidates", [])

        n_samples = kwargs.get("n_samples", 1)

        if not candidates:
            return []

        cand_name = get_unique_filename("candidates", ".xyz")
        cand_path = self.host_data_dir / cand_name
        write(cand_path, candidates)

        output_name = get_unique_filename("sampled", ".xyz")
        output_path = self.host_data_dir / output_name

        self.wrapper.sample(
            config_filename=self.config_filename,
            meta_config_filename=self.meta_config_filename,
            candidates_filename=cand_name,
            n_samples=n_samples,
            output_filename=output_name
        )

        selected = read(output_path, index=":")
        return [(a, -1) for a in selected]

class DockerStructureGenerator(StructureGenerator):
    def __init__(self, wrapper: GenWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename
        self.host_data_dir = wrapper.host_work_dir # Corrected from host_data_dir

    def generate_cell(self, large_atoms: Atoms, center_id: int, potential_path: str) -> Atoms:
        input_name = get_unique_filename("large_structure", ".xyz")
        output_name = get_unique_filename("small_cell", ".xyz")

        write(self.host_data_dir / input_name, large_atoms)
        pot_name = Path(potential_path).name

        self.wrapper.generate_cell(
            config_filename=self.config_filename,
            meta_config_filename=self.meta_config_filename,
            structure_filename=input_name,
            center=center_id,
            potential_filename=pot_name,
            output_filename=output_name
        )

        return read(self.host_data_dir / output_name)

class DockerKMCEngine(KMCEngine):
    def __init__(self, wrapper: LammpsWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.config_filename = config_filename
        self.meta_config_filename = meta_config_filename
        self.host_data_dir = wrapper.host_work_dir # Corrected from host_data_dir

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        input_name = get_unique_filename("kmc_input", ".xyz")
        output_name = get_unique_filename("kmc_result", ".pckl")

        write(self.host_data_dir / input_name, initial_atoms)
        pot_name = Path(potential_path).name

        self.wrapper.run_kmc(
            config_filename=self.config_filename,
            meta_config_filename=self.meta_config_filename,
            structure_filename=input_name,
            potential_filename=pot_name,
            output_filename=output_name
        )

        with open(self.host_data_dir / output_name, "rb") as f:
            result = pickle.load(f)

        return result

class DockerValidator(Validator):
    def __init__(self, wrapper: PaceWorker, config_filename: str, meta_config_filename: str):
        self.wrapper = wrapper
        self.host_data_dir = wrapper.host_work_dir # Corrected from host_data_dir

    def validate(self, potential_path: str) -> Dict[str, Any]:
        pot_name = Path(potential_path).name
        output_name = get_unique_filename("validation_metrics", ".json")

        self.wrapper.validate(pot_name, output_name)

        with open(self.host_data_dir / output_name, "r") as f:
            metrics = json.load(f)

        return metrics
