import os
import shutil
import tempfile
import logging
import json
from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict
from ase import Atoms
from ase.io import read, write

from shared.core.interfaces import Sampler, StructureGenerator, Labeler, Trainer, Validator
from shared.core.config import Config
from shared.autostructure.deformation import SystematicDeformationGenerator
from workers.al_md_kmc_worker.src.utils.parallel_executor import ParallelExecutor
# MDService injection for Epic 1
from workers.al_md_kmc_worker.src.services.md_service import MDService
from workers.al_md_kmc_worker.src.services.baseline_optimizer import BaselineOptimizer

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
        config: Config,
        md_service: Optional[MDService] = None
    ):
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer
        self.validator = validator
        self.config = config
        self.md_service = md_service
        self.max_workers = config.al_params.num_parallel_labeling
        self.executor = ParallelExecutor(max_workers=self.max_workers)

        # Setup debug dump directory
        self.debug_dump_dir = Path("debug_dump")
        self.debug_dump_dir.mkdir(exist_ok=True)

    def _label_clusters_parallel(self, clusters: List[Atoms]) -> List[Atoms]:
        """Label clusters in parallel."""
        return self.executor.execute(lambda s: _run_labeling_task(self.labeler, s), clusters)

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

    def _dump_artifact(self, atoms: Atoms, context: str, error: Exception):
        """Dump failed structure and context for debugging."""
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.debug_dump_dir / f"fail_{context}_{ts}.xyz"
            write(filename, atoms)
            meta_file = self.debug_dump_dir / f"fail_{context}_{ts}.json"
            with open(meta_file, 'w') as f:
                json.dump({"error": str(error), "context": context}, f)
            logger.info(f"Dumped failure artifact to {filename}")
        except Exception as e:
            logger.warning(f"Failed to dump artifact: {e}")

    def _rescue_unstable_structure(self, atoms: Atoms, potential_path: Path) -> Optional[Atoms]:
        """Rescue high-gamma structure using short MD via MDService."""
        if not self.md_service:
            logger.warning("MDService not injected. Cannot run rescue MD.")
            return None

        logger.info("Attempting Rescue MD for unstable structure...")
        try:
            rescued_atoms = self.md_service.run_rescue(atoms, str(potential_path))
            return rescued_atoms
        except Exception as e:
            logger.warning(f"Rescue MD failed: {e}")
            self._dump_artifact(atoms, "rescue_md_failed", e)
            return None

    def trigger_al(self,
                   uncertain_structures: List[Atoms],
                   potential_path: Path,
                   potential_yaml_path: Path,
                   asi_path: Optional[Path],
                   work_dir: Path,
                   iteration: int) -> Tuple[Optional[str], Dict[str, Any]]:
        """Trigger Active Learning pipeline."""
        logger.info(f"Triggering AL for {len(uncertain_structures)} uncertain structures.")
        metrics = {}

        # Ensure symbols
        for u_atoms in uncertain_structures:
             self.ensure_chemical_symbols(u_atoms)

        # Calculate uncertainty stats
        gammas = []
        for atoms in uncertain_structures:
            if hasattr(atoms, 'info') and 'max_gamma' in atoms.info:
                gammas.append(atoms.info['max_gamma'])
            elif hasattr(atoms, 'arrays') and 'gamma' in atoms.arrays:
                gammas.append(atoms.arrays['gamma'].max())

        if gammas:
            import numpy as np
            metrics["uncertainty_avg"] = float(np.mean(gammas))
            metrics["uncertainty_max"] = float(np.max(gammas))

        clusters_to_label = []
        for atoms in uncertain_structures:
            temp_dump = work_dir / "temp_uncertain.xyz"
            write(temp_dump, atoms, format="xyz")

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

                    rescued = self._rescue_unstable_structure(s_atoms, potential_path)

                    if rescued:
                         logger.info("Rescue successful. Using rescued structure.")
                         s_atoms = rescued
                    else:
                        # Fallback to PreOptimizer if exists
                        if hasattr(self.generator, 'pre_optimizer') and self.generator.pre_optimizer:
                             try:
                                 s_atoms = self.generator.pre_optimizer.run_pre_optimization(s_atoms)
                                 logger.info("PreOptimizer Rescue successful.")
                             except Exception as exc:
                                 logger.warning(f"PreOptimizer Rescue failed: {exc}. Discarding candidate.")
                                 self._dump_artifact(s_atoms, "preopt_rescue_failed", exc)
                                 continue
                        else:
                            logger.warning("No Rescue mechanism available. Discarding candidate.")
                            self._dump_artifact(s_atoms, "gamma_limit_exceeded_no_rescue", Exception(f"Gamma {max_gamma}"))
                            continue

                try:
                    cell = self.generator.generate_cell(s_atoms, center_id, str(potential_path))
                    clusters_to_label.append(cell)
                except Exception as e:
                    logger.warning(f"Generation failed for cluster {center_id}: {e}")
                    self._dump_artifact(s_atoms, f"generation_failed_cluster_{center_id}", e)

        if not clusters_to_label:
            logger.error("No clusters generated.")
            return None, metrics

        metrics["num_new_candidates"] = len(clusters_to_label)
        logger.info(f"Labeling {len(clusters_to_label)} clusters in parallel...")
        labeled_clusters = self._label_clusters_parallel(clusters_to_label)

        if not labeled_clusters:
            logger.error("No clusters labeled successfully.")
            return None, metrics

        metrics["num_training_structures"] = len(labeled_clusters)

        # Epic 4: Baseline Potential Auto-Optimization
        if self.config.ace_model.delta_learning_mode:
            logger.info("Delta Learning Mode: ON. Attempting to optimize LJ baseline.")
            try:
                # We need elements. Can get from config.md_params.elements
                elements = self.config.md_params.elements

                # Check if we have enough data? BaselineOptimizer handles data filtering.
                optimizer = BaselineOptimizer(elements, self.config.lj_params)

                # Run optimization on newly labeled clusters
                # Note: labeled_clusters contain the new DFT data
                opt_res = optimizer.optimize(labeled_clusters)

                if opt_res:
                    logger.info(f"Baseline Optimization Result: {opt_res}")

                    # Update Config with new params
                    eps_vals = list(opt_res['epsilon'].values())
                    sig_vals = list(opt_res['sigma'].values())

                    avg_eps = sum(eps_vals) / len(eps_vals)
                    avg_sig = sum(sig_vals) / len(sig_vals)

                    self.config.lj_params.epsilon = float(avg_eps)
                    self.config.lj_params.sigma = float(avg_sig)

                    logger.info(f"Updated Global LJ Params: epsilon={avg_eps:.4f}, sigma={avg_sig:.4f}")

            except Exception as e:
                logger.warning(f"Baseline optimization failed: {e}. Continuing with previous baseline.")

        logger.info("Training new potential...")
        dataset_path = self.trainer.prepare_dataset(labeled_clusters)

        pruning_freq = getattr(self.config.training_params, 'pruning_frequency', 0)
        pruning_thresh = getattr(self.config.training_params, 'pruning_threshold', 0.99)

        if asi_path and pruning_freq > 0 and iteration % pruning_freq == 0:
             if hasattr(self.trainer, "prune_active_set"):
                 self.trainer.prune_active_set(str(asi_path), threshold=pruning_thresh)

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

        # Merge validation results into metrics
        if isinstance(validation_results, dict):
            # mapping: train_rmse_energy, train_rmse_force
            # validation_results keys might differ, e.g. "energy_rmse", "force_rmse"
            # We map best effort
            if "energy_rmse" in validation_results:
                metrics["train_rmse_energy"] = validation_results["energy_rmse"]
            if "forces_rmse" in validation_results:
                metrics["train_rmse_force"] = validation_results["forces_rmse"]
            if "force_rmse" in validation_results:
                 metrics["train_rmse_force"] = validation_results["force_rmse"]

            # Also copy others
            metrics.update(validation_results)

        return new_potential, metrics

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
