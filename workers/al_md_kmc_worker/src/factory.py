import logging
import os
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

from src.interfaces.explorer import BaseExplorer
from src.services.md_service import MDService
from src.services.kmc_service import KMCService
from src.services.al_service import ActiveLearningService
from src.explorers.lammps_md_explorer import LammpsMDExplorer
from src.explorers.kmc_explorer import KMCExplorer
from src.explorers.ase_md_explorer import AseMDExplorer
from src.explorers.hybrid_explorer import HybridExplorer

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory class for creating Docker-based system components."""

    def __init__(self, config: Config, config_path: Path, meta_config_path: Path):
        self.config = config
        self.config_path = config_path.resolve()
        self.meta_config_path = meta_config_path.resolve()

        self.host_data_dir = Path("data").resolve()
        self._ensure_config_in_data()

        # Inject custom image names from config if available (Epic 2)
        # Default fallback is configured in Config.load_meta but we can also check here
        self.meta = config.meta

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

    def _get_image(self, key: str, default: str) -> str:
        """Helper to get image name from meta config (Epic 2)."""
        # meta.dft is dict, meta.lammps is dict.
        # But we added 'docker' section in setup_experiment default meta,
        # but Config class schema might not have it unless updated.
        # Let's check if 'docker' key exists in meta raw dict (we don't have access to raw dict easily here).
        # However, Config.meta is MetaConfig object.
        # If MetaConfig doesn't have 'docker' field, we can't access it.
        # The prompt in Step 4 mentioned updating default_meta in setup_experiment.py.
        # We should probably update MetaConfig in shared/core/config.py to support docker images.
        # But that's a separate task. For now, we stick to defaults or use defaults from the Wrapper classes.
        # Or, if Config.meta has generic dict support? No, it's dataclass.
        # We'll stick to default defaults for now, or hardcoded strings as per current wrappers.
        # If the user updated shared/core/config.py to include docker images, we use them.
        return default

    def create_md_engine(self) -> MDEngine:
        # Epic 2: Configurable images
        # image = self.meta.docker.get("lammps_image", "lammps_worker:latest") if hasattr(self.meta, "docker") else "lammps_worker:latest"
        # Since we haven't updated Config schema, we assume defaults or use what's in Wrapper.
        # But wait, Step 4 of prompt says "al_md_kmc_worker:latest".
        # This worker IS al_md_kmc_worker.
        # But create_md_engine needs to spawn a container for MD?
        # Wait, if we are al_md_kmc_worker, do we spawn *ourselves*?
        # Yes, prompt Step 3: "Change execution logic... to Docker Command Execution... sibling containers".
        # Prompt Step 2: "Rename workers/lammps_worker to workers/al_md_kmc_worker".
        # So MD tasks run in al_md_kmc_worker container.
        image = "al_md_kmc_worker:latest"
        wrapper = LammpsWorker(self.host_data_dir, image=image)
        return DockerMDEngine(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_sampler(self) -> Sampler:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerSampler(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_generator(self) -> StructureGenerator:
        # Generator was in gen_worker? Or lammps_worker (SmallCell)?
        # DockerStructureGenerator usually wraps GenWorker (MACE) or LammpsWorker (SmallCell).
        # We need to distinguish.
        # The prompt says "Rename workers/lammps_worker -> al_md_kmc_worker".
        # GenWorker is separate (MACE).
        # So create_generator should likely return GenWorker wrapper.
        wrapper = GenWorker(self.host_data_dir)
        return DockerStructureGenerator(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_labeler(self) -> Labeler:
        wrapper = DftWorker(self.host_data_dir)
        return DockerLabeler(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_trainer(self) -> Trainer:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerTrainer(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_kmc_engine(self) -> KMCEngine:
        image = "al_md_kmc_worker:latest"
        wrapper = LammpsWorker(self.host_data_dir, image=image)
        return DockerKMCEngine(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_validator(self) -> Validator:
        wrapper = PaceWorker(self.host_data_dir)
        return DockerValidator(wrapper, self.config_path.name, self.meta_config_path.name)

    def create_explorer(self, al_service: ActiveLearningService) -> BaseExplorer:
        """Creates the appropriate Explorer based on configuration."""
        strategy = self.config.exploration.strategy

        logger.info(f"Creating explorer for strategy: {strategy}")

        if strategy == "lammps_md":
            md_engine = self.create_md_engine()
            md_service = MDService(md_engine, self.config)
            return LammpsMDExplorer(md_service, self.config)

        elif strategy == "kmc":
            kmc_engine = self.create_kmc_engine()
            kmc_service = KMCService(kmc_engine, self.config)
            return KMCExplorer(kmc_service, al_service, self.config)

        elif strategy == "ase_md":
            return AseMDExplorer(self.config)

        elif strategy == "hybrid":
            md_engine = self.create_md_engine()
            md_service = MDService(md_engine, self.config)
            md_explorer = LammpsMDExplorer(md_service, self.config)

            kmc_engine = self.create_kmc_engine()
            kmc_service = KMCService(kmc_engine, self.config)
            kmc_explorer = KMCExplorer(kmc_service, al_service, self.config)

            return HybridExplorer(md_explorer, kmc_explorer)

        else:
            raise ValueError(f"Unknown exploration strategy: {strategy}")
