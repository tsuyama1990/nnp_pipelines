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

        # 2. Setup Baseline Calculator (Shifted LJ)
        lj = config.lj_params

        # We need to pass dictionaries if they are available, or scalar if not.
        # But config.lj_params is a dataclass with scalar fields typically?
        # Let's check shared/core/config.py
        # LJParams: epsilon: float, sigma: float...
        # Wait, if we want to support species-specific, we need to update Config or LJParams or handle it here.
        # The user said: "Auto-LJ: ... Update both the Python calculator and LAMMPS input generator to support multi-species pair coefficients."
        # And "Implement logic to retrieve vdw_radii from ase.data...".

        # I should generate these parameters here if they are not provided in config?
        # Or if config provides scalars, should I ignore them and use Auto-LJ?
        # "Remove manual configuration dependencies by (1) automatically generating species-specific LJ baseline parameters..."
        # So I should probably compute them here for the elements in the structure.

        # Let's implement Auto-LJ logic here or in a helper.
        from ase.data import vdw_radii, atomic_numbers

        lj_epsilon = {}
        lj_sigma = {}

        # Defaults if vdw not available
        DEFAULT_SIGMA = 2.0
        DEFAULT_EPSILON = 1.0 # No good physical default for Epsilon from radii alone usually?
        # User said: "Calculate per-pair parameters using standard mixing rules...".
        # But for the base parameters (sigma_i, epsilon_i), how do we get epsilon?
        # User only said "retrieve vdw_radii from ase.data. Calculate per-pair parameters...".
        # Usually epsilon is set to a fixed small value or some heuristic.
        # Let's assume we use the scalar epsilon from config as the base for all, or 1.0 if not set.
        # Or maybe the user implies we should guess epsilon too? "based on atomic radii" usually only gives sigma.
        # Let's stick to using vdw_radii for sigma, and config epsilon (scalar) for all, or 1.0.
        # BUT wait, the prompt says "Auto-LJ: ... automatically generating species-specific LJ baseline parameters based on atomic radii".
        # It's plural "parameters". Maybe sigma is from radii, epsilon is...?
        # For now I will use vdw_radii for sigma and the global epsilon for all species.

        base_epsilon = lj.epsilon if lj.epsilon else 0.05 # eV

        for el in sorted_elements:
             z = atomic_numbers.get(el)
             r_vdw = vdw_radii[z] if z < len(vdw_radii) else None

             if np.isnan(r_vdw) or r_vdw is None:
                 logger.warning(f"No vdW radius for {el}, using default.")
                 sigma_val = lj.sigma # Fallback to config scalar
             else:
                 # sigma = r_vdw / 2^(1/6) ?
                 # Usually r_min = 2^(1/6) * sigma.
                 # If we equate r_min to 2 * r_vdw (diameter? or distance of closest approach?)
                 # 2 * r_vdw is roughly the equilibrium distance.
                 # So 2 * r_vdw = 2^(1/6) * sigma
                 # sigma = (2 * r_vdw) / 1.122
                 sigma_val = (2 * r_vdw) / (2**(1/6))

             lj_sigma[el] = sigma_val
             lj_epsilon[el] = base_epsilon

        logger.info(f"Auto-generated LJ parameters: Sigma={lj_sigma}, Epsilon={lj_epsilon}")

        base_calc = ShiftedLennardJones(
            epsilon=lj_epsilon,
            sigma=lj_sigma,
            rcut=lj.cutoff,
            shift_energy=lj.shift_energy
        )

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
                import traceback
                traceback.print_exc()

        if labeled_structures:
            write(args.output, labeled_structures)
            print(f"Labeled {len(labeled_structures)} structures written to {args.output}")
        else:
            print("No structures labeled.")
            if not labeled_structures and all_atoms:
                sys.exit(1)

    except Exception as e:
        print(f"DFT Worker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
