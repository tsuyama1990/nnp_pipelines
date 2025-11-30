import argparse
import sys
from pathlib import Path
import logging
import numpy as np
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.calculators.espresso import Espresso

sys.path.append("/app")

from shared.core.config import Config
from shared.utils.atomic_energies import AtomicEnergyManager
from configurator import DFTConfigurator
from shared.potentials.shifted_lj import ShiftedLennardJones
from strategies.delta_labeler import DeltaLabeler
from strategies.raw_labeler import RawLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DFT Worker")
    parser.add_argument("--config", required=True, help="Path to main config.yaml")
    parser.add_argument("--meta-config", required=True, help="Path to meta_config.yaml")
    parser.add_argument("--structure", required=True, help="Path to input structure (.xyz)")
    parser.add_argument("--output", required=True, help="Path to output labeled structure (.xyz)")

    args = parser.parse_args()

    try:
        # Load Config
        meta = Config.load_meta(Path(args.meta_config))
        config = Config.load_experiment(Path(args.config), meta)

        # Load Structures (all frames)
        all_atoms = read(args.structure, index=":")
        if not isinstance(all_atoms, list):
            all_atoms = [all_atoms]

        labeled_structures = []

        # Setup Labeler Components

        # 1. Setup AtomicEnergyManager and get E0
        # Need all elements present in the structures
        unique_elements = set()
        for atoms in all_atoms:
            unique_elements.update(atoms.get_chemical_symbols())
        sorted_elements = sorted(list(unique_elements))

        # Factory for isolated atom calculation
        def dft_calculator_factory(element: str) -> Calculator:
             # Basic Gamma point calculation for isolated atom
             # Need minimal valid input for Espresso
             # This assumes pseudopotentials are available in pseudo_dir
             # and mapped in SSSP, but typically Espresso requires user to specify pseudopotentials manually
             # or use a profile.
             # However, DFTConfigurator knows how to build calculators.
             # But here we need a simple one.
             # Let's assume we can reuse DFTConfigurator logic or build a simple one.
             # Since we are inside dft_worker, we should use the available tools.
             # But DFTConfigurator builds for a specific structure.

             # Let's create a minimal configurator just for this?
             # Or just manually set it up if we have the pseudopotential info.
             # The user requirement says: "Define a dft_calculator_factory that creates a spin-polarized (nspin=2) Espresso calculator for a single isolated atom (Gamma point only)."

             # We need to look up the pseudopotential filename from SSSP JSON?
             # Yes, we can read it. But AtomicEnergyManager reads it too.
             # Ideally AtomicEnergyManager just handles E0, not the calculator creation details (dependency injection).

             # Let's use a helper that uses DFTConfigurator?
             # Or just direct Espresso.

             # We need to know which pseudopotential file to use.
             # We can load SSSP here or use sssp_loader.
             from shared.utils.sssp_loader import load_sssp_database, get_pseudopotentials_dict

             sssp_db = load_sssp_database(meta.sssp_json_path)
             pseudos = get_pseudopotentials_dict([element], sssp_db)

             input_data = {
                 'control': {
                     'calculation': 'scf',
                     'restart_mode': 'from_scratch',
                     'pseudo_dir': str(meta.pseudo_dir),
                     'tprnfor': True,
                     'tstress': True,
                     'disk_io': 'none', # Optimization
                 },
                 'system': {
                     'ecutwfc': 40, # Default or from SSSP? Ideally from SSSP.
                     'ecutrho': 320,
                     'occupations': 'smearing',
                     'smearing': 'mv',
                     'degauss': 0.01,
                     'nspin': 2, # Spin polarized
                 },
                 'electrons': {
                     'conv_thr': 1e-6,
                     'mixing_beta': 0.7,
                 }
             }

             # Get cutoffs from SSSP
             info = sssp_db[element]
             input_data['system']['ecutwfc'] = info.get('cutoff_wfc', 40)
             input_data['system']['ecutrho'] = info.get('cutoff_rho', input_data['system']['ecutwfc']*8)

             return Espresso(
                 command=meta.dft_command,
                 input_data=input_data,
                 pseudopotentials=pseudos,
                 kpts=(1, 1, 1) # Gamma point
             )

        ae_manager = AtomicEnergyManager(meta.sssp_json_path)
        e0_dict = ae_manager.get_e0(sorted_elements, dft_calculator_factory)

        # Resolve DFT settings for the active crystal type
        dft_overrides = {}
        try:
            c_type = config.seed_generation.crystal_type
            if c_type != "random":
                types_map = getattr(config.seed_generation, "types", {})
                if c_type in types_map:
                    dft_overrides = types_map[c_type].get("dft_settings", {})
        except AttributeError:
             pass

        # 2. Setup Labeler based on Mode
        if config.ace_model.delta_learning_mode:
            logger.info("Delta Learning Mode: ON. Using ShiftedLennardJones baseline.")

            lj = config.lj_params
            # Optimization Consistency: Use config params directly if provided (Optimization updates them)
            # Only fall back to auto-generation if explicitly requested or defaults are generic.
            # But the requirement said "remove manual configuration... automatically generating...".
            # However, Epic 4 says "Update config.lj_params with new optimal values."
            # So, if we have optimized values in config, we should use them.
            # We assume config.lj_params values are the "current best".

            # Since LJParams is scalar, we broadcast it to all species.
            # If we wanted species-specific, we'd need to change how config is passed or stored.
            # Given current constraints, we use the scalar config values.

            lj_epsilon = {el: lj.epsilon for el in sorted_elements}
            lj_sigma = {el: lj.sigma for el in sorted_elements}

            logger.info(f"Using LJ parameters from config (Epic 4 Optimized): Sigma={lj.sigma}, Epsilon={lj.epsilon}")

            base_calc = ShiftedLennardJones(
                epsilon=lj_epsilon,
                sigma=lj_sigma,
                rcut=lj.cutoff,
                shift_energy=lj.shift_energy
            )
        else:
            logger.info("Delta Learning Mode: OFF. Using Raw DFT Labeler.")
            base_calc = None


        for i, atoms in enumerate(all_atoms):
            try:
                logger.info(f"Labeling structure {i+1}/{len(all_atoms)}...")
                elements = sorted(list(set(atoms.get_chemical_symbols())))

                dft_configurator = DFTConfigurator(params=config.dft_params, meta=meta, type_dft_settings=dft_overrides)
                ref_calc, mag_settings = dft_configurator.build(atoms, elements, kpts=None)

                if config.ace_model.delta_learning_mode:
                    labeler = DeltaLabeler(
                        reference_calculator=ref_calc,
                        baseline_calculator=base_calc,
                        e0_dict=e0_dict,
                        outlier_energy_max=config.al_params.outlier_energy_max,
                        magnetism_settings=mag_settings
                    )
                else:
                    labeler = RawLabeler(
                        reference_calculator=ref_calc,
                        e0_dict=e0_dict,
                        outlier_energy_max=config.al_params.outlier_energy_max,
                        magnetism_settings=mag_settings
                    )

                labeled = labeler.label(atoms)
                if labeled:
                    labeled_structures.append(labeled)
                else:
                    logger.warning(f"Skipping structure {i+1} due to labeling failure.")
            except Exception as e:
                logger.error(f"Error labeling structure {i+1}: {e}", exc_info=True)

        if labeled_structures:
            write(args.output, labeled_structures)
            logger.info(f"Labeled {len(labeled_structures)} structures written to {args.output}")
        else:
            logger.warning("No structures labeled.")
            if not labeled_structures and all_atoms:
                sys.exit(1)

    except Exception as e:
        logger.error(f"DFT Worker failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
