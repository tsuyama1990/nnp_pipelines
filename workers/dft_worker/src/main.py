import argparse
import sys
from pathlib import Path
import logging
from ase.io import read, write

sys.path.append("/app")

from shared.core.config import Config
from configurator import DFTConfigurator
from calculators.shifted_lj import ShiftedLennardJones
from strategies.delta_labeler import DeltaLabeler

logging.basicConfig(level=logging.INFO)

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

        # Setup Labeler Components (Re-use for efficiency?)
        # Configurator depends on atoms for heuristics (pymatgen).
        # Heuristics might change per structure.
        # But calculators setup (pseudodir etc) is static.

        # We can reuse LJ.
        lj = config.lj_params
        base_calc = ShiftedLennardJones(
            epsilon=lj.epsilon,
            sigma=lj.sigma,
            rcut=lj.cutoff,
            shift_energy=lj.shift_energy
        )

        # E0 dict
        e0_dict = {} # Placeholder

        # Resolve DFT settings for the active crystal type
        dft_overrides = {}
        try:
            c_type = config.seed_generation.crystal_type
            if c_type != "random":
                # Support both naming conventions for robustness (types vs type_settings)
                types_map = getattr(config.seed_generation, "types", {}) or \
                            getattr(config.seed_generation, "type_settings", {})

                if c_type in types_map:
                    dft_overrides = types_map[c_type].get("dft_settings", {})
        except AttributeError:
             pass

        for i, atoms in enumerate(all_atoms):
            try:
                print(f"Labeling structure {i+1}/{len(all_atoms)}...")
                elements = sorted(list(set(atoms.get_chemical_symbols())))

                dft_configurator = DFTConfigurator(params=config.dft_params, meta=meta, type_dft_settings=dft_overrides)
                ref_calc, mag_settings = dft_configurator.build(atoms, elements, kpts=None)

                labeler = DeltaLabeler(
                    reference_calculator=ref_calc,
                    baseline_calculator=base_calc,
                    e0_dict=e0_dict,
                    outlier_energy_max=config.al_params.outlier_energy_max,
                    magnetism_settings=mag_settings
                )

                labeled = labeler.label(atoms)
                if labeled:
                    labeled_structures.append(labeled)
                else:
                    print(f"Skipping structure {i+1} due to labeling failure.")
            except Exception as e:
                print(f"Error labeling structure {i+1}: {e}")

        if labeled_structures:
            write(args.output, labeled_structures)
            print(f"Labeled {len(labeled_structures)} structures written to {args.output}")
        else:
            print("No structures labeled.")
            # We don't exit 1 here if at least one failed?
            # Or if ALL failed?
            if not labeled_structures and all_atoms:
                sys.exit(1)

    except Exception as e:
        print(f"DFT Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
