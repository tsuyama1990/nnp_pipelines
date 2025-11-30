"""Main entry point for the Unified AL-MD-KMC Worker.

This worker handles:
1. Active Learning Loop (Start Loop)
2. MD Simulation (Single Task)
3. KMC Simulation (Single Task)
4. Small Cell Generation (Single Task)
"""

import argparse
import sys
import logging
import pickle
from pathlib import Path
from dataclasses import asdict
from typing import Optional

# Setup path for local execution vs Docker execution
# Docker (standard): /app is root (where shared lives), /app/src (where code lives if copied)
# Local (repo root): ./shared lives here. ./workers/al_md_kmc_worker/src is code.
# We need to ensure 'src' and 'shared' are importable.

# Add repo root (for shared)
sys.path.append("/app")

# Add the directory containing 'src' package (workers/al_md_kmc_worker)
# This allows 'import src.xxx'
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fallback for old Docker structure (if src is at /app/src)
sys.path.append("/app/src")

from ase.io import read, write

from shared.core.config import Config
from shared.io.lammps_input import LAMMPSInputGenerator
from shared.generators.small_cell import SmallCellGenerator
from shared.engines.kmc import OffLatticeKMCEngine
from shared.utils.logger import CSVLogger

# Imports from src (moved from orchestrator)
from src.runner import LAMMPSRunner
from src.factory import ComponentFactory
from src.services.al_service import ActiveLearningService
from src.state_manager import StateManager
from src.workflows.active_learning_loop import ActiveLearningOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def run_active_learning(args):
    """Executes the Active Learning Loop."""
    logger.info("Starting Active Learning Loop...")

    config_path = Path(args.config)
    meta_path = Path(args.meta_config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load Configuration
    try:
        meta_config = Config.load_meta(meta_path)
        config = Config.load_experiment(config_path, meta_config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize Components via Factory
    # Note: ComponentFactory was moved to src/factory.py
    factory = ComponentFactory(config, config_path, meta_path)

    try:
        sampler = factory.create_sampler()
        generator = factory.create_generator()
        labeler = factory.create_labeler()
        trainer = factory.create_trainer()
        validator = factory.create_validator()
    except Exception as e:
        logger.exception(f"Failed to initialize components: {e}")
        sys.exit(1)

    # Instantiate Services
    al_service = ActiveLearningService(sampler, generator, labeler, trainer, validator, config)
    explorer = factory.create_explorer(al_service)

    # State Manager (assumes data/state.json relative to work dir)
    # The container usually mounts work dir to /app/work or /data?
    # Based on setup_experiment.py, -v $(pwd):/app/work.
    # So we are in /app/work usually?
    # If run_pipeline.sh does `docker run ... python src/main.py ...`
    # Check WORKDIR in Dockerfile or assumption.
    # Usually WORKDIR is /app.
    # If we mount to /app/work, we should look there.
    # But StateManager uses `data_dir` arg.
    state_manager = StateManager(Path("data")) # Relative to CWD

    csv_logger = CSVLogger()

    orchestrator = ActiveLearningOrchestrator(
        config=config,
        al_service=al_service,
        explorer=explorer,
        state_manager=state_manager,
        csv_logger=csv_logger
    )

    orchestrator.run()


def run_md(args, config):
    """Executes a single MD simulation task."""
    logger.info("Starting Single MD Task...")
    lj_dict = asdict(config.lj_params)
    md_dict = asdict(config.md_params)
    delta_mode = getattr(config.ace_model, "delta_learning_mode", True)

    generator = LAMMPSInputGenerator(lj_params=lj_dict, md_params=md_dict, delta_learning_mode=delta_mode)
    runner = LAMMPSRunner(cmd=config.meta.lammps_command, input_generator=generator)

    result = runner.run(
        potential_path=args.potential,
        steps=args.steps,
        gamma_threshold=args.gamma,
        input_structure=args.structure,
        is_restart=args.restart
    )
    logger.info(f"Simulation Result: {result.name}")


def run_small_cell(args, config):
    """Generates a small cell for KMC/NEB."""
    logger.info("Starting Small Cell Generation...")
    large_atoms = read(args.structure)
    elements = config.md_params.elements
    ratio = {el: 1.0 for el in elements}

    delta_mode = getattr(config.ace_model, "delta_learning_mode", True)
    lj_dict = asdict(config.lj_params)

    generator = SmallCellGenerator(
        r_core=config.al_params.r_core,
        box_size=config.al_params.box_size,
        stoichiometric_ratio=ratio,
        lammps_cmd=config.meta.lammps_command,
        min_bond_distance=config.al_params.min_bond_distance,
        stoichiometry_tolerance=config.al_params.stoichiometry_tolerance,
        delta_learning_mode=delta_mode,
        lj_params=lj_dict
    )

    small_atoms = generator.generate_cell(
        large_atoms=large_atoms,
        center_id=args.center,
        potential_path=args.potential
    )

    write(args.output, small_atoms)
    logger.info(f"Small cell written to {args.output}")


def run_kmc(args, config):
    """Executes a single KMC step."""
    logger.info("Starting Single KMC Step...")
    e0_dict = {} # TODO: Load from atomic energy manager if needed

    delta_mode = getattr(config.ace_model, "delta_learning_mode", True)

    engine = OffLatticeKMCEngine(
        kmc_params=config.kmc_params,
        al_params=config.al_params,
        lj_params=config.lj_params,
        e0_dict=e0_dict,
        delta_learning_mode=delta_mode
    )

    initial_atoms = read(args.structure)
    result = engine.run_step(initial_atoms, args.potential)

    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"KMC result written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Unified AL-MD-KMC Worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Active Learning Loop ---
    al_parser = subparsers.add_parser("start_loop", help="Start the Active Learning Loop")
    al_parser.add_argument("--config", required=True)
    al_parser.add_argument("--meta-config", required=True)

    # --- Single MD Task ---
    md_parser = subparsers.add_parser("md", help="Run a single MD simulation")
    md_parser.add_argument("--config", required=True)
    md_parser.add_argument("--meta-config", required=True)
    md_parser.add_argument("--potential", required=False, default=None)
    md_parser.add_argument("--structure", required=True)
    md_parser.add_argument("--steps", type=int, required=True)
    md_parser.add_argument("--gamma", type=float, required=True)
    md_parser.add_argument("--restart", action="store_true")

    # --- Small Cell Generation ---
    sc_parser = subparsers.add_parser("small_cell", help="Generate a small cell")
    sc_parser.add_argument("--config", required=True)
    sc_parser.add_argument("--meta-config", required=True)
    sc_parser.add_argument("--potential", required=True)
    sc_parser.add_argument("--structure", required=True)
    sc_parser.add_argument("--center", type=int, required=True)
    sc_parser.add_argument("--output", required=True)

    # --- Single KMC Task ---
    kmc_parser = subparsers.add_parser("kmc", help="Run a single KMC step")
    kmc_parser.add_argument("--config", required=True)
    kmc_parser.add_argument("--meta-config", required=True)
    kmc_parser.add_argument("--potential", required=True)
    kmc_parser.add_argument("--structure", required=True)
    kmc_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    try:
        # Load Config for tasks that need it immediately (MD, KMC, SmallCell)
        # AL Loop loads it internally
        if args.command in ["md", "small_cell", "kmc"]:
            meta = Config.load_meta(Path(args.meta_config))
            config = Config.load_experiment(Path(args.config), meta)

            if args.command == "md":
                run_md(args, config)
            elif args.command == "small_cell":
                run_small_cell(args, config)
            elif args.command == "kmc":
                run_kmc(args, config)

        elif args.command == "start_loop":
            run_active_learning(args)

    except Exception as e:
        logger.error(f"Worker execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
