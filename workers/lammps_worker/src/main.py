import argparse
import sys
from pathlib import Path
import logging
from dataclasses import asdict
from ase.io import read, write

sys.path.append("/app")
from shared.core.config import Config
from runner import LAMMPSRunner
from input_generator import LAMMPSInputGenerator
from strategies.small_cell import SmallCellGenerator
from kmc import OffLatticeKMCEngine
from shared.core.interfaces import KMCEngine
import pickle

logging.basicConfig(level=logging.INFO)

def run_md(args, config):
    lj_dict = asdict(config.lj_params)
    md_dict = asdict(config.md_params)
    # Extract delta_learning_mode, default to True if missing (though config enforces it)
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
    print(f"Simulation Result: {result.name}")

def run_small_cell(args, config):
    large_atoms = read(args.structure)
    elements = config.md_params.elements
    ratio = {el: 1.0 for el in elements}

    generator = SmallCellGenerator(
        r_core=config.al_params.r_core,
        box_size=config.al_params.box_size,
        stoichiometric_ratio=ratio,
        lammps_cmd=config.meta.lammps_command,
        min_bond_distance=config.al_params.min_bond_distance,
        stoichiometry_tolerance=config.al_params.stoichiometry_tolerance
    )

    small_atoms = generator.generate_cell(
        large_atoms=large_atoms,
        center_id=args.center,
        potential_path=args.potential
    )

    write(args.output, small_atoms)
    print(f"Small cell written to {args.output}")

def run_kmc(args, config):
    # e0_dict needs to be loaded?
    # Factory logic used AtomicEnergyManager.
    # Here we might need to pass E0 or calculate it?
    # Config doesn't have e0.
    # For now, pass empty dict or load from default path if available.
    e0_dict = {} # Simplified

    engine = OffLatticeKMCEngine(
        kmc_params=config.kmc_params,
        al_params=config.al_params,
        lj_params=config.lj_params,
        e0_dict=e0_dict
    )

    initial_atoms = read(args.structure)
    result = engine.run_step(initial_atoms, args.potential)

    # KMCResult contains structure and status.
    # We need to serialize the result back to orchestrator.
    # Pickle is best.
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    print(f"KMC result written to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="LAMMPS Worker")
    subparsers = parser.add_subparsers(dest="task", required=True)

    # MD
    md_parser = subparsers.add_parser("md")
    md_parser.add_argument("--config", required=True)
    md_parser.add_argument("--meta-config", required=True)
    md_parser.add_argument("--potential", required=False, default=None, help="Path to ACE potential (optional)")
    md_parser.add_argument("--structure", required=True)
    md_parser.add_argument("--steps", type=int, required=True)
    md_parser.add_argument("--gamma", type=float, required=True)
    md_parser.add_argument("--restart", action="store_true")

    # Small Cell
    sc_parser = subparsers.add_parser("small_cell")
    sc_parser.add_argument("--config", required=True)
    sc_parser.add_argument("--meta-config", required=True)
    sc_parser.add_argument("--potential", required=True)
    sc_parser.add_argument("--structure", required=True)
    sc_parser.add_argument("--center", type=int, required=True)
    sc_parser.add_argument("--output", required=True)

    # KMC
    kmc_parser = subparsers.add_parser("kmc")
    kmc_parser.add_argument("--config", required=True)
    kmc_parser.add_argument("--meta-config", required=True)
    kmc_parser.add_argument("--potential", required=True)
    kmc_parser.add_argument("--structure", required=True)
    kmc_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    try:
        meta = Config.load_meta(Path(args.meta_config))
        config = Config.load_experiment(Path(args.config), meta)

        if args.task == "md":
            run_md(args, config)
        elif args.task == "small_cell":
            run_small_cell(args, config)
        elif args.task == "kmc":
            run_kmc(args, config)

    except Exception as e:
        print(f"LAMMPS Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
