
import pytest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
from workers.lammps_worker.src.kmc import ActiveSiteSelector, OffLatticeKMCEngine, KMCStatus, KMCResult
from shared.core.config import KMCParams, ALParams, LJParams

def get_dummy_al_params():
    return ALParams(
        gamma_threshold=0.5,
        n_clusters=1,
        r_core=3.0,
        box_size=10.0,
        initial_potential="dummy.yace",
        potential_yaml_path="dummy.yaml"
    )

def test_active_site_selector():
    # Use Co which is in default active_species
    atoms = Atoms('Co4', positions=[[0,0,0], [0,0,2.5], [0,0,5], [0,0,10]])
    cns = np.array([12, 8, 4, 1]) # Fake CNs

    # Test Z-Coordinate
    params = KMCParams(active_z_cutoff=4.0)
    selector = ActiveSiteSelector("z_coordinate", params)
    indices = selector.select(atoms, cns)
    assert 2 in indices and 3 in indices
    assert 0 not in indices

    # Test Coordination
    params = KMCParams(adsorbate_cn_cutoff=5)
    selector = ActiveSiteSelector("coordination", params)
    indices = selector.select(atoms, cns)
    assert 3 in indices and 2 in indices
    assert 1 not in indices

def test_cluster_identification_map_reduce():
    # 2 distant dimers
    atoms = Atoms('Co4', positions=[
        [0,0,0], [1.5,0,0], # Cluster 1
        [20,20,20], [21.5,20,20] # Cluster 2
    ])
    atoms.set_cell([30,30,30])
    atoms.set_pbc(True)

    kmc_params = KMCParams(
        box_size=10.0,
        cluster_connectivity_cutoff=3.0,
        n_workers=1,
        active_z_cutoff=-10 # Select all
    )
    al_params = get_dummy_al_params()
    lj_params = LJParams(epsilon=1.0, sigma=1.0, cutoff=3.0)

    engine = OffLatticeKMCEngine(kmc_params, al_params, lj_params)

    # Manually trigger internal methods to test decomposition
    indices, indptr, cns = engine._compute_connectivity(atoms)

    # Assume all active
    active_indices = np.array([0,1,2,3])

    tasks = engine._find_independent_clusters(atoms, active_indices, indptr, indices)

    assert len(tasks) == 2

    # Verify carving
    c1, map1 = tasks[0]
    assert len(c1) == 2
    assert 0 in map1 and 1 in map1

    c2, map2 = tasks[1]
    assert len(c2) == 2
    assert 2 in map2 and 3 in map2

@patch("workers.lammps_worker.src.kmc._run_local_search")
@patch("workers.lammps_worker.src.kmc.ProcessPoolExecutor")
def test_engine_run_step_execution(mock_executor, mock_search):
    # Setup Mock Executor
    mock_future = MagicMock()
    mock_future.result.return_value = (0.5, np.zeros((2,3)), None) # Barrier, Disp, None

    mock_pool = MagicMock()
    mock_pool.submit.return_value = mock_future
    mock_executor.return_value.__enter__.return_value = mock_pool

    # Use Co to satisfy species check
    atoms = Atoms('Co2', positions=[[0,0,0], [2,0,0]])

    kmc_params = KMCParams(n_workers=1, active_z_cutoff=-10)
    al_params = get_dummy_al_params()
    lj_params = LJParams(epsilon=1.0, sigma=1.0, cutoff=3.0)

    engine = OffLatticeKMCEngine(kmc_params, al_params, lj_params)

    # Mock compute connectivity
    engine._compute_connectivity = MagicMock(return_value=(np.array([1,0]), np.array([0,1,2]), np.array([1,1])))

    # Run
    res = engine.run_step(atoms, "dummy.yace")

    # Note: Because _run_local_search is mocked, the result tuple (0.5, zeros, None) is used.
    # The Map-Reduce logic should construct the global event.
    assert res.status == KMCStatus.SUCCESS
    # Check if barrier is propagated
    # Since we mocked the result tuple to have barrier 0.5, we expect that in metadata
    assert res.metadata["barrier"] == 0.5
