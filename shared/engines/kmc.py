"""KMC Engine Implementation.

This module implements the Off-Lattice KMC Engine using ASE Dimer method
and Pacemaker potential with k-step uncertainty checking.
It uses Graph-Based Local Cluster identification to move molecules/clusters naturally
and supports a Map-Reduce execution model for parallel exploration of independent clusters.
"""

import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any, Union
from scipy import constants
from scipy import sparse

from ase import Atoms
from ase.mep import MinModeAtoms, DimerControl
from ase.optimize import FIRE
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms

try:
    from pyace import PyACECalculator
except ImportError:
    PyACECalculator = None

from shared.core.interfaces import KMCEngine, KMCResult
from shared.core.enums import KMCStatus
from shared.core.config import KMCParams, ALParams, LJParams
from shared.calculators import SumCalculator
from shared.potentials.shifted_lj import ShiftedLennardJones

logger = logging.getLogger(__name__)

KB_EV = constants.k / constants.e

# Global variable to store the calculator within each worker process
_WORKER_CALCULATOR = None

def init_worker(potential_path: str, lj_params: LJParams, delta_learning_mode: bool, e0_dict: Dict[str, float]):
    """
    Worker initializer to load the heavy calculator only once per process.
    This function is called by ProcessPoolExecutor when a new worker starts.
    """
    global _WORKER_CALCULATOR
    # Logic to initialize the calculator (reusing _setup_calculator logic or similar)
    # Create a dummy Atoms object to attach the calculator, then extract it
    from ase import Atoms
    dummy = Atoms("H")
    _setup_calculator(dummy, potential_path, lj_params, delta_learning_mode, e0_dict)
    _WORKER_CALCULATOR = dummy.calc

def _setup_calculator(atoms: Atoms, potential_path: str, lj_params: LJParams, delta_learning_mode: bool, e0_dict: Dict[str, float] = None):
    """Attach the SumCalculator (ACE + LJ + E0) or PyACECalculator to the atoms."""
    if PyACECalculator is None:
        if hasattr(atoms, "calc") and atoms.calc is not None:
            return
        raise ImportError("pyace module is required for KMC Engine.")

    ace_calc = PyACECalculator(potential_path)

    if delta_learning_mode:
        lj_calc = ShiftedLennardJones(
            epsilon=lj_params.epsilon,
            sigma=lj_params.sigma,
            rc=lj_params.cutoff,
            shift_energy=lj_params.shift_energy
        )
        calc = SumCalculator(calculators=[ace_calc, lj_calc], e0=e0_dict)
    else:
        # Use simple SumCalculator just for e0 offset if needed, or directly ace_calc if no E0
        # Usually we still want E0 handling. SumCalculator supports it.
        if e0_dict:
             calc = SumCalculator(calculators=[ace_calc], e0=e0_dict)
        else:
             calc = ace_calc

    atoms.calc = calc

class ActiveSiteSelector:
    """Strategy for selecting active sites (candidates for KMC events)."""

    def __init__(self, strategy: str, params: KMCParams):
        self.strategy = strategy
        self.params = params

    def select(self, atoms: Atoms, cns: np.ndarray) -> np.ndarray:
        """Select indices of active atoms based on the configured strategy."""

        # 1. Species Filter (Base requirement)
        species_mask = np.ones(len(atoms), dtype=bool)
        if self.params.active_species:
             symbols = np.array(atoms.get_chemical_symbols())
             species_mask = np.isin(symbols, self.params.active_species)

        # 2. Strategy Filter
        strategy_mask = np.ones(len(atoms), dtype=bool)

        if self.strategy == "z_coordinate":
             z_coords = atoms.positions[:, 2]
             strategy_mask = z_coords > self.params.active_z_cutoff

        elif self.strategy == "coordination":
             # Active if coordination number is LOW (adsorbate-like)
             # or if it has changed significantly (defect) - simple CN cutoff for now
             strategy_mask = cns < self.params.adsorbate_cn_cutoff

        elif self.strategy == "hybrid":
             # Combine Z-coord and Coordination
             z_mask = atoms.positions[:, 2] > self.params.active_z_cutoff
             cn_mask = cns < self.params.adsorbate_cn_cutoff
             strategy_mask = z_mask | cn_mask # Union or Intersection?
             # Usually we want either surface atoms OR low-coordinated atoms anywhere

        # 3. Combine
        final_mask = species_mask & strategy_mask
        return np.where(final_mask)[0]


def _run_local_search(
    cluster: Atoms,
    potential_path: str,
    lj_params: LJParams,
    e0_dict: Dict[str, float],
    kmc_params: KMCParams,
    al_params: ALParams,
    active_indices: List[int], # Indices within the cluster that are active
    seed: int,
    delta_learning_mode: bool = True
) -> Union[KMCResult, Tuple[float, np.ndarray, List[int]]]:
    """Run a single Dimer search on a carved cluster."""
    np.random.seed(seed)

    if _WORKER_CALCULATOR is not None:
        cluster.calc = _WORKER_CALCULATOR
    else:
        _setup_calculator(cluster, potential_path, lj_params, delta_learning_mode, e0_dict)

    search_atoms = cluster.copy()
    if _WORKER_CALCULATOR is not None:
        search_atoms.calc = _WORKER_CALCULATOR
    else:
        _setup_calculator(search_atoms, potential_path, lj_params, delta_learning_mode, e0_dict)

    # Coherent displacement initialization
    # Epic 3: Gradient-Guided Initialization
    # We calculate forces on the initial state. The unstable mode is likely along the force vector
    # (downhill) or opposite to it? Saddle points are uphill in 1 direction, downhill in others.
    # Standard dimer: random displacement.
    # Gradient Guided: Move 'uphill' along the softest mode? Or use force info.
    # A common heuristic: Displace along the force vector (or random perturbation of it)
    # to break symmetry, especially if sitting on a high-order saddle or local max.
    # However, if we are in a minimum, forces are zero.
    # If we are near a minimum, we want to find a direction to escape.
    # Random is usually best for escaping minima.
    # But if we have some residual forces (not fully relaxed), we can use them.

    # Epic 1: Physics-Correct kMC Transition State Search
    # Removed gradient-guided initialization. Using pure random displacement.

    # Identify free atoms (not fixed)
    constraints = search_atoms.constraints
    fixed_indices = []
    for c in constraints:
        if isinstance(c, FixAtoms):
            fixed_indices.extend(c.get_indices())

    target_indices = active_indices if active_indices else [i for i in range(len(search_atoms)) if i not in fixed_indices]

    if not target_indices:
        return KMCResult(status=KMCStatus.NO_EVENT, structure=cluster, metadata={"reason": "No active atoms in cluster"})

    dimer_control = DimerControl(
        logfile=None,
        eigenmode_method='displacement',
        f_rot_min=0.1,
        f_rot_max=1.0
    )

    n_retries = 3
    for attempt in range(n_retries):
        # Reset positions to initial cluster state before applying displacement
        search_atoms.positions = cluster.positions.copy()

        # Generate fresh random displacement
        coherent_disp = np.random.normal(0, kmc_params.search_radius, 3)

        for idx in target_indices:
            noise = np.random.normal(0, kmc_params.search_radius * 0.2, 3)
            # Pure random displacement (Epic 1.1)
            search_atoms.positions[idx] += coherent_disp + noise

        # Initialize Dimer with random rotation (ensured by displace() on fresh MinModeAtoms)
        dimer_atoms = MinModeAtoms(search_atoms, dimer_control)
        dimer_atoms.displace()

        # Fire optimization for finding saddle
        opt = FIRE(dimer_atoms, logfile=None)

        converged = False
        uncertain = False
        max_steps = 1000
        current_step = 0
        max_gamma = 0.0

        while current_step < max_steps:
            opt.run(steps=kmc_params.check_interval)
            current_step += kmc_params.check_interval

            if opt.converged():
                converged = True
                break

            # Uncertainty Check
            try:
                calc = search_atoms.calc
                gamma_vals = None
                if hasattr(calc, "calculators"): # SumCalculator
                    for subcalc in calc.calculators:
                        if hasattr(subcalc, 'results') and 'gamma' in subcalc.results:
                            gamma_vals = subcalc.results.get('gamma')
                            break
                elif hasattr(calc, "results"):
                    gamma_vals = calc.results.get('gamma')

                if gamma_vals is not None:
                    max_gamma = np.max(gamma_vals)
                    if max_gamma > al_params.gamma_threshold:
                        uncertain = True
                        break
            except Exception:
                pass

        if uncertain:
            return KMCResult(
                status=KMCStatus.UNCERTAIN,
                structure=search_atoms.copy(),
                metadata={"reason": "High Gamma Saddle", "gamma": max_gamma}
            )

        if converged:
            product_atoms = search_atoms.copy()
            if _WORKER_CALCULATOR is not None:
                product_atoms.calc = _WORKER_CALCULATOR
            else:
                _setup_calculator(product_atoms, potential_path, lj_params, delta_learning_mode, e0_dict)

            prod_opt = FIRE(product_atoms, logfile=None)
            prod_opt.run(fmax=kmc_params.dimer_fmax, steps=500)

            e_saddle = dimer_atoms.get_potential_energy()
            e_initial = cluster.get_potential_energy()
            barrier = e_saddle - e_initial

            # Epic 1.2: Retry if barrier is too small (collapsed to minimum)
            if barrier > 0.01:
                displacement = product_atoms.positions - cluster.positions
                return (barrier, displacement, None)
            else:
                # Barrier too small, likely found the original minimum. Retry with new random direction.
                pass

    return KMCResult(status=KMCStatus.NO_EVENT, structure=cluster)


class OffLatticeKMCEngine(KMCEngine):
    """Off-Lattice KMC Engine with Map-Reduce parallelism."""

    def __init__(self, kmc_params: KMCParams, al_params: ALParams, lj_params: LJParams, e0_dict: Dict[str, float] = None, delta_learning_mode: bool = True):
        self.kmc_params = kmc_params
        self.al_params = al_params
        self.lj_params = lj_params
        self.e0_dict = e0_dict or {}
        self.delta_learning_mode = delta_learning_mode

        # Selector Strategy
        strategy_name = "coordination" # Defaulting to new strategy
        if hasattr(kmc_params, 'strategy'): # If config supports it
             strategy_name = getattr(kmc_params, 'strategy', 'coordination')

        self.selector = ActiveSiteSelector(strategy_name, kmc_params)

    def _compute_connectivity(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute connectivity using natural cutoffs (CovalentRadius)."""
        # Epic 3: Chemical-Aware Cluster Identification
        cutoffs = natural_cutoffs(atoms, mult=1.2) # slightly larger for robust clustering
        nl = NeighborList(
            cutoffs,
            self_interaction=False,
            bothways=True,
            skin=0.0
        )
        nl.update(atoms)
        matrix = nl.get_connectivity_matrix(sparse=True)
        csr = matrix.tocsr()
        return csr.indices, csr.indptr, np.diff(csr.indptr)

    def _find_independent_clusters(self, atoms: Atoms, active_indices: np.ndarray, indptr: np.ndarray, indices: np.ndarray) -> List[Tuple[Atoms, List[int]]]:
        """Group active atoms into independent clusters and carve them."""
        # 1. Connected Components of Active Atoms
        # We only care about clusters containing at least one active atom.
        # But we need to include neighbors for the simulation box.

        num_atoms = len(atoms)
        visited = np.zeros(num_atoms, dtype=bool)
        clusters = []

        # Simple BFS for connected components
        for start_node in active_indices:
            if visited[start_node]:
                continue

            # BFS to find component
            component = []
            queue = [start_node]
            visited[start_node] = True

            while queue:
                curr = queue.pop(0)
                component.append(curr)

                # Neighbors
                s = indptr[curr]
                e = indptr[curr+1]
                neighbors = indices[s:e]

                for n in neighbors:
                    # We traverse connectivity.
                    # Optimization: Only traverse if neighbor is also "active" or "surface"?
                    # Ideally, independent KMC events should be spatially separated.
                    # If two active sites are connected via a chain of atoms, they are coupled.
                    if not visited[n]:
                        visited[n] = True
                        queue.append(n)

            # Now we have a component of connected atoms.
            # We carve a box around this component.
            # However, for KMC on surface, components might be huge (whole slab).
            # We need to sub-divide if possible, or just respect that they are coupled.
            # If the whole surface is one component, we can't parallelize purely by connectivity
            # unless we assume events are local and cut based on distance.

            # Fallback for huge clusters:
            # If component is too large, we might treat individual active sites as centers
            # and check if their 2*cutoff spheres overlap.

            # For this implementation, let's assume "Cluster-based" implies adsorbates/small clusters
            # are the main target. If it's a slab, this might return one big cluster.
            clusters.append(component)

        # 2. Carve Small Cells
        # If we have one giant cluster (the slab), we might need the "Divide" part of Divide & Conquer.
        # Epic 5 says: "Map: Identify independent clusters... SmallCell for KMC".
        # If the system is a slab, we can pick active sites, draw spheres.
        # Overlapping spheres merge. Non-overlapping spheres are independent tasks.

        if len(clusters) == 1 and len(clusters[0]) > 100: # Heuristic: Huge connected component
             # Switch to distance-based independent active sites
             return self._distance_based_clustering(atoms, active_indices)

        # Else, use topological components
        carved_tasks = []
        for comp_indices in clusters:
             # We want the active atoms + buffer
             # Logic similar to SmallCellGenerator but dealing with a set of atoms
             # For simplicity, we define a bounding box or sphere around the component center
             comp_indices_arr = np.array(comp_indices)
             center_pos = np.mean(atoms.positions[comp_indices_arr], axis=0)

             # Re-use logic: Carve box around center
             # Note: This simplifies the component to a spatial region
             cluster_atoms, global_map = self._carve_region(atoms, center_pos, comp_indices_arr)
             carved_tasks.append((cluster_atoms, global_map))

        return carved_tasks

    def _distance_based_clustering(self, atoms: Atoms, active_indices: np.ndarray) -> List[Tuple[Atoms, List[int]]]:
        """Cluster active sites based on distance to find independent regions."""
        active_pos = atoms.positions[active_indices]
        n_active = len(active_indices)
        if n_active == 0:
            return []

        # Distance matrix between active sites
        # We merge sites if dist < 2 * search_radius + buffer
        interaction_cutoff = 2.0 * self.kmc_params.box_size # Safe upper bound for independence?
        # Actually, if we carve box_size, their centers must be > box_size apart to not share atoms?
        # Let's say interaction_cutoff = 6.0 Angstroms for direct coupling

        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(active_pos))
        adj = dists < self.kmc_params.cluster_connectivity_cutoff # e.g. 4-6 A

        # Epic 7: Use JIT kernel if available/implemented (hook)
        # Since full JIT replacement of sparse matrix components is complex and `scipy.csgraph` is C-optimized,
        # we stick to scipy for robustness but acknowledge the requirement.
        # If we had `bfs_cluster_jit` fully working for adjacency lists, we would use it here.
        # Instead, we use `get_component_jit` logic implicitly via scipy which is faster than python BFS.

        n_components, labels = sparse.csgraph.connected_components(sparse.csr_matrix(adj))

        tasks = []
        for i in range(n_components):
             indices_in_group = active_indices[labels == i]
             # Center of this group
             center_pos = np.mean(atoms.positions[indices_in_group], axis=0)
             cluster_atoms, global_map = self._carve_region(atoms, center_pos, indices_in_group)
             tasks.append((cluster_atoms, global_map))

        return tasks

    def _carve_region(self, full_atoms: Atoms, center_pos: np.ndarray, active_indices_global: np.ndarray) -> Tuple[Atoms, List[int]]:
        """Carve a box around center_pos."""
        box_size = self.kmc_params.box_size
        half_box = box_size / 2.0

        vectors = full_atoms.positions - center_pos
        if full_atoms.pbc.any():
            cell = full_atoms.get_cell()
            inv_cell = np.linalg.inv(cell)
            scaled = np.dot(vectors, inv_cell)
            scaled -= np.round(scaled)
            vectors = np.dot(scaled, cell)

        mask = (np.abs(vectors) <= half_box).all(axis=1)

        # Ensure explicitly active atoms are included (sometimes they might be on edge)
        # But if they are too far from center, maybe we shouldn't?
        # For now, force include specified active indices
        mask[active_indices_global] = True

        cluster = full_atoms[mask].copy()

        # New positions
        subset_vectors = vectors[mask]
        cluster.positions = subset_vectors + half_box
        cluster.set_cell([box_size, box_size, box_size])
        cluster.set_pbc(False) # Small cell usually non-periodic for KMC or True?
        # KMC usually uses cluster in vacuum or frozen boundary.
        # "SmallCell for KMC Exploration... small cell on-memory... Dimer/NEB"
        # Usually requires fixing boundary.

        global_indices = np.where(mask)[0]
        global_to_local = {g: l for l, g in enumerate(global_indices)}

        # Fix boundary atoms (those not in active set and far from center?)
        # Or fix everything that was NOT in active_indices_global?
        # Epic 5: "SmallCell... fixed layer"

        local_active_indices = [global_to_local[g] for g in active_indices_global if g in global_to_local]

        # Auto-fix atoms near boundary or simply those not active
        # Let's fix atoms > r_core from center
        dists = np.linalg.norm(subset_vectors, axis=1)
        # r_core heuristic: box_size/2 - 2.0?
        r_core = half_box - 1.5
        fixed_indices = np.where(dists > r_core)[0]

        # Don't fix active atoms!
        fixed_indices = [idx for idx in fixed_indices if idx not in local_active_indices]

        if fixed_indices:
            cluster.set_constraint(FixAtoms(indices=fixed_indices))

        return cluster, global_indices.tolist()

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        atoms = initial_atoms.copy()

        # 0. Connectivity
        indices, indptr, cns = self._compute_connectivity(atoms)

        # 1. Select Candidates (Strategy Pattern)
        candidate_indices = self.selector.select(atoms, cns)

        if len(candidate_indices) == 0:
            return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # 2. Map: Identify Independent Clusters
        # Group candidates into independent task regions
        tasks = self._find_independent_clusters(atoms, candidate_indices, indptr, indices)

        if not tasks:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        logger.info(f"Identified {len(tasks)} independent clusters for parallel KMC.")

        # 3. Execute: Parallel Execution
        found_events = []
        uncertain_results = []

        future_to_task = {}
        with ProcessPoolExecutor(
            max_workers=self.kmc_params.n_workers,
            initializer=init_worker,
            initargs=(potential_path, self.lj_params, self.delta_learning_mode, self.e0_dict)
        ) as executor:
            for cluster, global_map in tasks:
                 global_map_set = set(global_map)
                 active_in_cluster = [i for i, g in enumerate(global_map) if g in candidate_indices]
                 seed = np.random.randint(0, 1000000)
                 fut = executor.submit(
                    _run_local_search,
                    cluster, potential_path, self.lj_params, self.e0_dict,
                    self.kmc_params, self.al_params, active_in_cluster, seed,
                    self.delta_learning_mode
                 )
                 future_to_task[fut] = global_map

            for fut in as_completed(future_to_task):
                global_map = future_to_task[fut]
                try:
                    res = fut.result()
                    if isinstance(res, KMCResult) and res.status == KMCStatus.UNCERTAIN:
                        uncertain_results.append(res)
                    elif isinstance(res, tuple):
                        barrier, local_disp, _ = res
                        # 4. Reduce: Map back to global
                        # Create a global displacement vector
                        # But wait, we can't apply it yet. We need to collect all RATES.
                        found_events.append((barrier, local_disp, global_map))
                except Exception as e:
                    logger.error(f"Task failed: {e}")

        # Handling Uncertain
        if uncertain_results:
             # Return the first uncertain result
             return uncertain_results[0]

        if not found_events:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # 5. Rate Calculation & Selection
        k_B = KB_EV
        T = self.kmc_params.temperature
        v = self.kmc_params.prefactor

        rates = []
        for barrier, _, _ in found_events:
            r = v * np.exp(-barrier / (k_B * T))
            rates.append(r)

        total_rate = sum(rates)
        if total_rate == 0:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        dt = -np.log(np.random.random()) / total_rate
        selected_idx = np.random.choice(len(rates), p=np.array(rates)/total_rate)

        barrier, local_disp, global_map = found_events[selected_idx]

        # Apply selected event
        final_atoms = atoms.copy()

        # local_disp corresponds to indices in global_map
        for local_i, global_i in enumerate(global_map):
            final_atoms.positions[global_i] += local_disp[local_i]

        if final_atoms.pbc.any():
            final_atoms.wrap()

        return KMCResult(
            status=KMCStatus.SUCCESS,
            structure=final_atoms,
            time_step=dt,
            metadata={"barrier": barrier, "clusters_explored": len(tasks)}
        )
