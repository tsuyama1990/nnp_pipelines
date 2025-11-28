import logging
import yaml
import shutil
import uuid
import pandas as pd
from pathlib import Path
from ase.io import read

from shared.core.config import Config
from orchestrator.src.wrappers.gen_wrapper import GenWorker
from orchestrator.src.wrappers.dft_wrapper import DftWorker
from orchestrator.src.wrappers.pace_wrapper import PaceWorker

logger = logging.getLogger(__name__)

def get_unique_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"

class SeedGenerator:
    """Manages Phase 1: Seed Generation using Workers."""

    def __init__(self, config: Config, config_path: Path, meta_config_path: Path):
        self.config = config
        self.config_path = config_path.resolve()
        self.meta_config_path = meta_config_path.resolve()
        self.host_data_dir = Path("data").resolve()

        # Ensure data dir exists
        self.host_data_dir.mkdir(parents=True, exist_ok=True)

        self.gen_worker = GenWorker(self.host_data_dir)
        self.dft_worker = DftWorker(self.host_data_dir)
        self.pace_worker = PaceWorker(self.host_data_dir)

    def run(self):
        logger.info("Starting Seed Generation Phase...")

        # 1. Random Generation
        # Create temp config for random scenario
        random_conf = {
            "type": "random",
            "elements": self.config.md_params.elements,
            "n_structures": 100,
            "max_atoms": 8
        }

        rand_conf_name = get_unique_filename("random_conf", ".yaml")
        rand_out_name = get_unique_filename("random_structures", ".xyz")

        with open(self.host_data_dir / rand_conf_name, "w") as f:
            yaml.dump(random_conf, f)

        logger.info("Generating random structures...")
        try:
            self.gen_worker.generate(rand_conf_name, rand_out_name)
        except Exception as e:
            logger.error(f"Random generation failed: {e}")
            raise e

        # 2. MACE Filter
        filtered_name = get_unique_filename("filtered", ".xyz")
        logger.info("Filtering with MACE...")
        self.gen_worker.filter(rand_out_name, filtered_name, model="medium", fmax=100.0)

        # 3. Direct Sampling (Diversity)
        sampled_name = get_unique_filename("sampled_seed", ".xyz")
        logger.info("Sampling diverse structures...")
        self.pace_worker.direct_sample(filtered_name, sampled_name, n_clusters=20)

        # 4. DFT Labeling
        labeled_name = get_unique_filename("labeled_seed", ".xyz")
        logger.info("Labeling with DFT...")
        self.dft_worker.label(self.config_path.name, self.meta_config_path.name, sampled_name, labeled_name)

        # 5. Train
        # Prepare dataset locally
        labeled_path = self.host_data_dir / labeled_name
        if not labeled_path.exists():
            raise FileNotFoundError(f"Labeled structures not found at {labeled_path}")

        labeled_atoms = read(labeled_path, index=":")
        if not labeled_atoms:
            raise RuntimeError("No labeled atoms found.")

        df = pd.DataFrame({"ase_atoms": labeled_atoms})
        dataset_name = get_unique_filename("seed_dataset", ".pckl.gzip")
        df.to_pickle(self.host_data_dir / dataset_name, compression="gzip")

        logger.info("Training Seed Potential...")
        pot_name = self.pace_worker.train(
            self.config_path.name,
            self.meta_config_path.name,
            dataset_name,
            iteration=0
        )

        # Copy to final location
        final_pot = Path("data/seed/seed_potential.yace")
        final_pot.parent.mkdir(parents=True, exist_ok=True)
        source_pot = self.host_data_dir / pot_name
        if source_pot.exists():
            shutil.copy(source_pot, final_pot)
            logger.info(f"Seed generation complete. Potential: {final_pot}")
        else:
            logger.error(f"Trained potential {source_pot} not found.")
            raise FileNotFoundError("Training failed to produce potential.")
