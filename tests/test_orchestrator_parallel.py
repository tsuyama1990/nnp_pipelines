
import pytest
from unittest.mock import MagicMock, patch, ANY
# Updated imports based on path patching
from workers.al_md_kmc_worker.src.workflows.active_learning_loop import ActiveLearningOrchestrator
from workers.al_md_kmc_worker.src.services.md_service import MDService, _run_md_task
from shared.core.config import Config, ExplorationStage, MDParams
from shared.core.enums import SimulationState
from pathlib import Path

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.exploration_schedule = [
        ExplorationStage(iter_start=1, iter_end=5, temp=[300.0, 500.0], press=[1.0, 10.0])
    ]
    config.md_params = MagicMock(spec=MDParams)
    config.md_params.temperature = 300.0
    config.md_params.pressure = 1.0
    config.md_params.n_md_walkers = 4
    config.md_params.n_steps = 100

    # Correctly mock al_params which is nested in Config
    mock_al = MagicMock()
    mock_al.gamma_threshold = 2.0
    mock_al.num_parallel_labeling = 2
    mock_al.deformation_frequency = 0
    mock_al.initial_potential = "potential.yace"
    mock_al.initial_active_set_path = "potential.asi"
    mock_al.initial_dataset_path = "dataset.xyz"
    mock_al.max_al_retries = 3

    config.al_params = mock_al

    return config

def test_get_md_conditions(mock_config):
    # This method belongs to MDService, not ActiveLearningOrchestrator
    # We patch ParallelExecutor where it is used in MDService to avoid real processes
    with patch("services.md_service.ParallelExecutor"):
        md_service = MDService(MagicMock(), mock_config)

    # Test inside schedule
    conditions = md_service._get_md_conditions(3)
    assert 300.0 <= conditions["temperature"] <= 500.0
    assert 1.0 <= conditions["pressure"] <= 10.0

    # Test outside schedule
    conditions = md_service._get_md_conditions(10)
    assert conditions["temperature"] == 300.0
    assert conditions["pressure"] == 1.0

def test_parallel_md_execution(mock_config):
    md_engine = MagicMock()

    # Mock Executor
    # We patch ParallelExecutor in the module where it is USED: services.md_service
    with patch("services.md_service.ParallelExecutor") as MockExecutor:
        # Instantiate MDService INSIDE the patch so it uses the mock
        md_service = MDService(md_engine, mock_config)

        mock_executor_instance = MockExecutor.return_value
        mock_executor_instance.__enter__.return_value = mock_executor_instance

        # Mock submit_tasks to return successful results
        mock_executor_instance.submit_tasks.return_value = [(SimulationState.COMPLETED, Path("dump.lammpstrj.1"))] * 4

        # Run walkers
        success, uncertain_structures, failure = md_service.run_walkers(
            iteration=1,
            potential_path=Path("potential.yace"),
            input_structure_path="structure.data",
            is_restart=False
        )

        # Verify success
        assert success is True
        assert failure is False

        # Verify ParallelExecutor was initialized
        assert MockExecutor.called

        # Verify tasks were submitted
        assert mock_executor_instance.submit_tasks.called
        # Check that we submitted 4 tasks (n_md_walkers=4)
        tasks = mock_executor_instance.submit_tasks.call_args[0][0]
        assert len(tasks) == 4
