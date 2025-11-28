import logging
from pathlib import Path

from shared.core.config import Config
from shared.core.interfaces import MDEngine, Sampler, StructureGenerator, Labeler, Trainer, KMCEngine, Validator
from .adapters import (
    DockerMDEngine, DockerLabeler, DockerTrainer, DockerSampler,
    DockerStructureGenerator, DockerKMCEngine, DockerValidator
)
from .wrappers.lammps_wrapper import LammpsWorker
from .wrappers.dft_wrapper import DftWorker
from .wrappers.pace_wrapper import PaceWorker
from .wrappers.gen_wrapper import GenWorker

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory class for creating Docker-based system components."""

    def __init__(self, config: Config, config_path: Path, meta_config_path: Path):
        self.config = config
        self.config_path = config_path.resolve()
        self.meta_config_path = meta_config_path.resolve()

        self.host_data_dir = Path("data").resolve()
        self._ensure_config_in_data()

    def _ensure_config_in_data(self):
        """Copies config files to data dir if not already there."""
        self.host_data_dir.mkdir(exist_ok=True)

        target_config = self.host_data_dir / self.config_path.name
        if target_config != self.config_path:
            import shutil
            shutil.copy(self.config_path, target_config)

        target_meta = self.host_data_dir / self.meta_config_path.name
        if target_meta != self.meta_config_path:
            import shutil
            shutil.copy(self.meta_config_path, target_meta)

    def create_md_engine(self) -> MDEngine:
        wrapper = LammpsWorker(self.host_data_dir)
        return DockerMDEngine(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_sampler(self) -> Sampler:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerSampler(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_generator(self) -> StructureGenerator:
        wrapper = LammpsWorker(self.host_data_dir)
        return DockerStructureGenerator(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_labeler(self) -> Labeler:
        wrapper = DftWorker(self.host_data_dir)
        return DockerLabeler(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_trainer(self) -> Trainer:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerTrainer(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_kmc_engine(self) -> KMCEngine:
        wrapper = LammpsWorker(self.host_data_dir)
        return DockerKMCEngine(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_validator(self) -> Validator:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerValidator(wrapper, self.config_path.name, self.meta_config_path.name)
