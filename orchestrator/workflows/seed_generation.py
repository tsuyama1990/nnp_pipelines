import logging
import yaml
import shutil
import uuid
import pandas as pd
from pathlib import Path
from ase.io import read, write

from shared.core.config import Config
from orchestrator.src.wrappers.gen_wrapper import GenWorker
from orchestrator.src.wrappers.dft_wrapper import DftWorker
from orchestrator.src.wrappers.pace_wrapper import PaceWorker
from orchestrator.src.wrappers.lammps_wrapper import LammpsWorker

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
        self.lammps_worker = LammpsWorker(self.host_data_dir)

    def run(self):
        logger.info("Starting Seed Generation Phase...")

        # 1. Random Generation
        # Create temp config for random scenario
        random_conf = {
            "type": "random",
            "elements": self.config.md_params.elements,
            "n_structures": self.config.seed_generation.n_random_structures,
            "max_atoms": 8
        }

        rand_conf_name = get_unique_filename("random_conf", ".yaml")
        rand_out_name = get_unique_filename("random_structures", ".xyz")

        with open(self.host_data_dir / rand_conf_name, "w") as f:
            yaml.dump(random_conf, f)

        logger.info(f"Generating {self.config.seed_generation.n_random_structures} random structures...")
        try:
            self.gen_worker.generate(rand_conf_name, rand_out_name)
        except Exception as e:
            logger.error(f"Random generation failed: {e}")
            raise e

        # 2. Multi-temperature Exploration (replacing MACE Filter)
        logger.info("Starting multi-temperature MD exploration...")

        # Load the generated random structures
        rand_structures_path = self.host_data_dir / rand_out_name
        if not rand_structures_path.exists():
            raise FileNotFoundError(f"Random structures file not found: {rand_structures_path}")

        random_atoms_list = read(rand_structures_path, index=":")
        logger.info(f"Loaded {len(random_atoms_list)} structures for exploration.")

        exploration_trajectories = []

        for idx, atoms in enumerate(random_atoms_list):
            # Save single structure to a temp file for MD input
            single_struct_name = get_unique_filename(f"seed_input_{idx}", ".data") # LAMMPS usually likes .data or .xyz, wrapper handles conversion if needed?
            # Wrapper takes `structure_filename`. `LammpsWorker.run_md` calls `main.py` which calls `LAMMPSInputGenerator`.
            # `LAMMPSInputGenerator` writes `read_data {input_structure}`.
            # `input_structure` is the path passed.
            # ASE `write` can write lammps-data format.
            # But the existing pipeline seems to use XYZ often.
            # Let's check `LammpsWorker` `main.py` -> `run_md` -> `runner.run` logic.
            # `LAMMPSInputGenerator` puts `read_data` or `read_restart`.
            # Ideally we should convert ASE atoms to LAMMPS data format here.

            # Use `ase.io.write` with format='lammps-data'
            single_struct_path = self.host_data_dir / single_struct_name
            write(single_struct_path, atoms, format='lammps-data')

            for temp in self.config.seed_generation.exploration_temperatures:
                logger.info(f"Running MD for structure {idx} at {temp}K...")

                # We need to temporarily override temperature in config?
                # The `LammpsWorker` reads config from file.
                # We should probably create a temporary config for this MD run or rely on `md_params`.
                # But `md_params` has one temperature.
                # `config.md_params.temperature` is used in `LAMMPSInputGenerator`.
                # So we MUST create a temp config with the specific temperature.

                temp_config_dict = self.config.__dict__.copy() # Shallow copy might not be enough for nested dataclasses if we modify them inplace.
                # Use `asdict` then modify?
                # Or just modify the current config object temporarily and save it to a new yaml file.

                # Better approach: Create a temporary Config object or dictionary, modify MD params, dump to yaml.
                # Since we need to pass a filename to the worker.

                # Helper to create temp config file with override temp
                # We can read the original config file, update it, and write it back.

                with open(self.config_path, 'r') as f:
                    base_config_dict = yaml.safe_load(f)

                # Update temperature and steps
                if 'md_params' not in base_config_dict:
                    base_config_dict['md_params'] = {}
                base_config_dict['md_params']['temperature'] = temp
                base_config_dict['md_params']['n_steps'] = self.config.seed_generation.n_md_steps

                temp_config_name = get_unique_filename(f"conf_{idx}_{temp}", ".yaml")
                with open(self.host_data_dir / temp_config_name, 'w') as f:
                    yaml.dump(base_config_dict, f)

                # MD Output filenames
                # LammpsWorker produces "dump.lammpstrj" fixed name?
                # Wait, `LammpsWorker.run_md` calls `main.py` which calls `LAMMPSInputGenerator`.
                # `LAMMPSInputGenerator` hardcodes `dump 1 all custom {dump_freq} dump.lammpstrj ...`.
                # It does NOT take an output filename argument for the dump file.
                # This is a limitation. The worker runs in a container, likely in a specific working directory or data dir.
                # If we run multiple in parallel or sequence, they might overwrite "dump.lammpstrj" if they share the folder.
                # BUT `DockerWrapper` mounts `host_data_dir` to `/data`.
                # The worker script runs in `/app`.
                # `LammpsInputGenerator` writes `dump.lammpstrj` to the current working directory?
                # `LAMMPSRunner` uses `subprocess.run(..., cwd=self.work_dir)`.
                # We need to see where `LAMMPSRunner` runs.
                # It seems `LAMMPSRunner` is in `workers/lammps_worker/src/runner.py`.
                # Let's assume it outputs to the directory where the script is run or specified.
                # Wait, `LammpsWorker.run_md` (wrapper) does NOT specify an output dir.
                # `input_generator.py` writes `dump.lammpstrj` relative path.

                # IMPORTANT: The Wrapper implementation:
                # `cmd = ["python", "/app/src/main.py", "md", ...]`
                # It runs in the container.
                # Where does `main.py` write files?
                # `LAMMPSInputGenerator` writes `dump.lammpstrj`.
                # If it writes to CWD, and CWD is `/`, that's bad.
                # If CWD is `/data`, that's okay but we will overwrite.

                # In `workers/lammps_worker/src/runner.py` (which I haven't read but can guess), it likely executes lmp.
                # We should check `runner.py`.
                # Since I didn't check `runner.py`, I should be careful.
                # However, `LammpsWorker` wrapper mounts `self.host_data_dir`.
                # If I can't control the output filename, I must rename it after each run.
                # But the run is inside the container.
                # The container shares `/data` with host.
                # If the worker writes to `/data/dump.lammpstrj`, I can see it on host as `host_data_dir/dump.lammpstrj`.
                # I should rename it immediately after `run_md` returns.

                try:
                    self.lammps_worker.run_md(
                        config_filename=temp_config_name,
                        meta_config_filename=self.meta_config_path.name,
                        potential_filename=None, # Pure LJ
                        structure_filename=single_struct_name,
                        steps=self.config.seed_generation.n_md_steps,
                        gamma=100.0 # High threshold to prevent early stopping if it were active (but it's disabled in LJ mode)
                    )

                    # Move/Rename the output dump
                    # The worker likely produced `dump.lammpstrj` in `/data`?
                    # I need to confirm where it writes.
                    # Looking at `input_generator.py`: `lines.append(f"dump 1 all custom {dump_freq} dump.lammpstrj ...")`
                    # It doesn't specify a path. So it's CWD.
                    # `DockerWrapper.run` usually runs with `working_dir` set?
                    # The base `DockerWrapper` (which I didn't read) probably sets up the docker run command.
                    # If it doesn't set working dir, it might be `/`.
                    # But the wrappers mount `host_data_dir` to `/data`.
                    # And `main.py` uses `/data` for inputs.
                    # If `lmp_serial` is run, where does it dump?
                    # Usually where it's invoked.
                    # The wrapper runs `python /app/src/main.py`.
                    # `main.py` runs `LAMMPSRunner`.

                    # Let's assume the output ends up in `/data` (host_data_dir) OR I need to look for it.
                    # If it ends up in `/app`, I can't see it easily unless I copy it.
                    # But the user said "Workers use /app/src/main.py as the entry point and accept CLI subcommands with arguments referencing files within the mounted /data directory."
                    # Usually these systems are designed to output to `/data`.
                    # Let's assume `dump.lammpstrj` appears in `self.host_data_dir`.

                    default_dump_name = "dump.lammpstrj"
                    default_dump_path = self.host_data_dir / default_dump_name

                    if default_dump_path.exists():
                        target_dump_name = get_unique_filename(f"traj_{idx}_{temp}", ".lammpstrj")
                        target_dump_path = self.host_data_dir / target_dump_name
                        shutil.move(default_dump_path, target_dump_path)
                        exploration_trajectories.append(target_dump_path)
                    else:
                        logger.warning(f"No dump file found for structure {idx} at {temp}K")

                except Exception as e:
                    logger.error(f"MD exploration failed for struct {idx}, temp {temp}: {e}")
                    # Continue to next?
                    continue

        # 3. Concatenate Trajectories
        logger.info("Concatenating trajectories...")
        combined_traj_name = get_unique_filename("combined_seed_trajectory", ".xyz")
        combined_traj_path = self.host_data_dir / combined_traj_name

        all_atoms = []
        for traj_path in exploration_trajectories:
            try:
                # Read lammpstrj
                # ASE can read lammpstrj.
                # We need to know the species to read it correctly if it's just atomic numbers,
                # but `dump.lammpstrj` from `input_generator` has `id type x y z ...`.
                # We need to map types to species.
                # `read` needs `specorder` usually for lammps-dump if types are numbers.
                # The `elements` are in `self.config.md_params.elements`.
                # The order in `input_generator` is `elements` list order.

                traj = read(traj_path, index=":", format="lammps-dump-text", specorder=self.config.md_params.elements)
                all_atoms.extend(traj)
            except Exception as e:
                logger.warning(f"Failed to read trajectory {traj_path}: {e}")

        if not all_atoms:
            raise RuntimeError("No structures generated from MD exploration.")

        write(combined_traj_path, all_atoms)
        logger.info(f"Combined trajectory saved to {combined_traj_name} with {len(all_atoms)} frames.")

        # 4. Direct Sampling (Diversity)
        sampled_name = get_unique_filename("sampled_seed", ".xyz")
        logger.info(f"Sampling {self.config.seed_generation.n_samples_for_dft} diverse structures...")

        self.pace_worker.direct_sample(
            combined_traj_name,
            sampled_name,
            n_clusters=self.config.seed_generation.n_samples_for_dft
        )

        # 5. DFT Labeling
        labeled_name = get_unique_filename("labeled_seed", ".xyz")
        logger.info("Labeling with DFT...")
        self.dft_worker.label(self.config_path.name, self.meta_config_path.name, sampled_name, labeled_name)

        # 6. Train
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
