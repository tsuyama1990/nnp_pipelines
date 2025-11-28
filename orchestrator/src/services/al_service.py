import os
import shutil
import tempfile
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Any
from ase import Atoms
from ase.io import write

from shared.core.interfaces import Sampler, StructureGenerator, Labeler, Trainer, Validator
from shared.core.config import Config
from shared.autostructure.deformation import SystematicDeformationGenerator

logger = logging.getLogger(__name__)

def _run_labeling_task(labeler: Labeler, structure: Atoms) -> Optional[Atoms]:
    """Helper function to run labeling in a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix="label_task_")
    original_cwd = os.getcwd()

    try:
        os.chdir(tmpdir)
        return labeler.label(structure)
    except Exception as e:
        logger.error(f"Labeling task failed in {tmpdir}: {e}")
        return None
    finally:
        os.chdir(original_cwd)
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {tmpdir}: {e}")


class ActiveLearningService:
    """Service for managing the Active Learning Loop."""

    def __init__(
        self,
        sampler: Sampler,
        generator: StructureGenerator,
        labeler: Labeler,
        trainer: Trainer,
        validator: Validator,
        config: Config
    ):
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer
        self.validator = validator
        self.config = config
        self.max_workers = config.al_params.num_parallel_labeling

    def _label_clusters_parallel(self, clusters: List[Atoms]) -> List[Atoms]:
        """Label clusters in parallel."""
        labeled_results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cluster = {
                executor.submit(_run_labeling_task, self.labeler, c): c
                for c in clusters
            }
            for future in as_completed(future_to_cluster):
                try:
                    result = future.result()
                    if result is not None:
                        labeled_results.append(result)
                except Exception as e:
                    logger.error(f"Parallel labeling execution failed: {e}")
        return labeled_results

    def ensure_chemical_symbols(self, atoms: Atoms):
        if 'type' in atoms.arrays:
             types = atoms.get_array('type')
             elements = self.config.md_params.elements
             symbols = []
             for t in types:
                 if 1 <= t <= len(elements):
                     symbols.append(elements[t-1])
                 else:
                     symbols.append("X")
             atoms.set_chemical_symbols(symbols)

    def trigger_al(self,
                   uncertain_structures: List[Atoms],
                   potential_path: Path,
                   potential_yaml_path: Path,
                   asi_path: Optional[Path],
                   work_dir: Path,
                   iteration: int) -> Optional[str]:
        """Trigger Active Learning pipeline."""
        logger.info(f"Triggering AL for {len(uncertain_structures)} uncertain structures.")

        # Ensure symbols
        for u_atoms in uncertain_structures:
             self.ensure_chemical_symbols(u_atoms)

        clusters_to_label = []
        for atoms in uncertain_structures:
            temp_dump = work_dir / "temp_uncertain.lammpstrj"
            write(temp_dump, atoms, format="lammps-dump-text")

            sample_kwargs = {
                "structures": [atoms],
                "potential_path": str(potential_path),
                "n_clusters": self.config.al_params.n_clusters,
                "dump_file": str(temp_dump),
                "potential_yaml_path": str(potential_yaml_path),
                "asi_path": str(asi_path) if asi_path else None,
                "elements": self.config.md_params.elements
            }

            selected_samples = self.sampler.sample(**sample_kwargs)

            for s_atoms, center_id in selected_samples:
                # Check gamma upper bound
                max_gamma = 0.0
                if hasattr(s_atoms, 'info') and 'max_gamma' in s_atoms.info:
                    max_gamma = s_atoms.info['max_gamma']
                elif hasattr(s_atoms, 'arrays') and 'gamma' in s_atoms.arrays:
                     max_gamma = s_atoms.arrays['gamma'].max()

                if max_gamma > self.config.al_params.gamma_upper_bound:
                    logger.warning(f"Gamma {max_gamma} exceeds limit. Attempting rescue.")
                    if hasattr(self.generator, 'pre_optimizer') and self.generator.pre_optimizer:
                         try:
                             s_atoms = self.generator.pre_optimizer.run_pre_optimization(s_atoms)
                             logger.info("Rescue successful.")
                         except Exception as exc:
                             logger.warning(f"Rescue failed: {exc}. Discarding candidate.")
                             continue
                    else:
                        logger.warning("No Pre-Optimizer available. Discarding candidate.")
                        continue

                try:
                    cell = self.generator.generate_cell(s_atoms, center_id, str(potential_path))
                    clusters_to_label.append(cell)
                except Exception as e:
                    logger.warning(f"Generation failed for cluster {center_id}: {e}")

        if not clusters_to_label:
            logger.error("No clusters generated.")
            return None

        logger.info(f"Labeling {len(clusters_to_label)} clusters in parallel...")
        labeled_clusters = self._label_clusters_parallel(clusters_to_label)

        if not labeled_clusters:
            logger.error("No clusters labeled successfully.")
            return None

        logger.info("Training new potential...")
        dataset_path = self.trainer.prepare_dataset(labeled_clusters)

        if asi_path and iteration % 10 == 0:
             if hasattr(self.trainer, "prune_active_set"):
                 self.trainer.prune_active_set(str(asi_path), threshold=0.99)

        new_potential = self.trainer.train(
            dataset_path=dataset_path,
            initial_potential=str(potential_path),
            potential_yaml_path=str(potential_yaml_path),
            asi_path=str(asi_path) if asi_path else None,
            iteration=iteration
        )

        logger.info("Validating new potential...")
        validation_results = self.validator.validate(new_potential)
        logger.info(f"Validation Results: {validation_results}")

        return new_potential

    def inject_deformation_data(self, input_structure_path: str):
        """Inject systematically deformed structures into the dataset."""
        logger.info("Injecting distorted structures for EOS/Elasticity.")
        try:
            struct_to_deform = read(input_structure_path)
            self.ensure_chemical_symbols(struct_to_deform)

            def_gen = SystematicDeformationGenerator(struct_to_deform, self.config.lj_params)
            distorted_structures = def_gen.generate_all()

            logger.info(f"Generated {len(distorted_structures)} distorted structures.")
            labeled_distorted = self._label_clusters_parallel(distorted_structures)

            if labeled_distorted:
                self.trainer.prepare_dataset(labeled_distorted)
                logger.info(f"Added {len(labeled_distorted)} labeled distorted structures to dataset.")
            else:
                logger.warning("All distorted structures failed labeling.")

        except Exception as e:
            logger.error(f"Systematic deformation injection failed: {e}")
