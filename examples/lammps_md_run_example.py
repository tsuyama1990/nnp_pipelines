"""
Example script to run Active Learning with LAMMPS MD using the new Explorer architecture.
"""

import sys
import logging
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Ensure the repository root is in PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from shared.core.config import Config
from orchestrator.src.factory import ComponentFactory
from orchestrator.workflows.orchestrator import ActiveLearningOrchestrator
from orchestrator.src.services.al_service import ActiveLearningService
from orchestrator.src.state_manager import StateManager
from shared.utils.logger import CSVLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lammps_md_example")

def main():
    # 1. Define Paths
    # Note: Ensure these config files exist or point to your actual config
    config_path = Path("config.yaml")
    meta_path = Path("meta_config.yaml")

    if not config_path.exists() or not meta_path.exists():
        logger.error("Config files not found. Please run this from the repo root or adjust paths.")
        return

    # 2. Load Configuration
    meta_config = Config.load_meta(meta_path)
    config = Config.load_experiment(config_path, meta_config)

    # 3. Override Exploration Strategy for this example
    # We want to force LAMMPS MD exploration
    config.exploration.strategy = "lammps_md"
    logger.info(f"Exploration strategy set to: {config.exploration.strategy}")

    # 4. Initialize Components
    factory = ComponentFactory(config, config_path, meta_path)

    # Create Services
    # For full functionality, we need all components, but here we focus on AL + Explorer
    try:
        # We need to create dependent components for AL Service
        # Depending on factory implementation, we might need Docker setup or mocks if just testing logic.
        # This example assumes Docker infrastructure is available (like in the real run).

        sampler = factory.create_sampler()
        generator = factory.create_generator()
        labeler = factory.create_labeler()
        trainer = factory.create_trainer()
        validator = factory.create_validator()

        al_service = ActiveLearningService(
            sampler, generator, labeler, trainer, validator, config
        )

        # 5. Create Explorer
        # This will create LammpsMDExplorer because we set strategy="lammps_md"
        explorer = factory.create_explorer(al_service)
        logger.info(f"Created Explorer: {type(explorer).__name__}")

        # 6. Setup Orchestrator
        state_manager = StateManager(Path("data"))
        csv_logger = CSVLogger()

        orchestrator = ActiveLearningOrchestrator(
            config=config,
            al_service=al_service,
            explorer=explorer,
            state_manager=state_manager,
            csv_logger=csv_logger
        )

        # 7. Run (Optional)
        # orchestrator.run()
        logger.info("Orchestrator initialized successfully. Ready to run.")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

if __name__ == "__main__":
    main()
