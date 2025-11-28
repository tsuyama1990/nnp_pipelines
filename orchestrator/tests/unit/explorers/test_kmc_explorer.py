import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from orchestrator.src.explorers.kmc_explorer import KMCExplorer
from orchestrator.src.interfaces.explorer import ExplorationResult, ExplorationStatus
from shared.core.enums import KMCStatus

@pytest.fixture
def mock_kmc_service():
    return MagicMock()

@pytest.fixture
def mock_al_service():
    return MagicMock()

@pytest.fixture
def mock_config():
    return MagicMock()

def test_kmc_explorer_success(mock_kmc_service, mock_al_service, mock_config):
    explorer = KMCExplorer(mock_kmc_service, mock_al_service, mock_config)

    # Mock KMC result
    kmc_result = MagicMock()
    kmc_result.status = KMCStatus.SUCCESS
    kmc_result.structure = MagicMock()
    mock_kmc_service.run_step.return_value = kmc_result

    # Mock file operations
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("orchestrator.src.explorers.kmc_explorer.read") as mock_read, \
         patch("orchestrator.src.explorers.kmc_explorer.write") as mock_write:

        mock_exists.return_value = True
        mock_read.return_value = MagicMock() # Atoms

        result = explorer.explore(
            current_structure="input.data",
            potential_path="pot.yace",
            iteration=1
        )

        assert result.status == ExplorationStatus.SUCCESS
        assert "kmc_output.data" in result.final_structure
        assert result.metadata["is_restart"] is False

def test_kmc_explorer_uncertain(mock_kmc_service, mock_al_service, mock_config):
    explorer = KMCExplorer(mock_kmc_service, mock_al_service, mock_config)

    kmc_result = MagicMock()
    kmc_result.status = KMCStatus.UNCERTAIN
    kmc_result.structure = MagicMock()
    mock_kmc_service.run_step.return_value = kmc_result

    with patch("pathlib.Path.exists") as mock_exists, \
         patch("orchestrator.src.explorers.kmc_explorer.read") as mock_read:

        mock_exists.return_value = True

        result = explorer.explore(
            current_structure="input.data",
            potential_path="pot.yace",
            iteration=1
        )

        assert result.status == ExplorationStatus.UNCERTAIN
        assert result.uncertain_structures == [kmc_result.structure]

def test_kmc_explorer_no_event(mock_kmc_service, mock_al_service, mock_config):
    explorer = KMCExplorer(mock_kmc_service, mock_al_service, mock_config)

    kmc_result = MagicMock()
    kmc_result.status = KMCStatus.NO_EVENT
    mock_kmc_service.run_step.return_value = kmc_result

    with patch("pathlib.Path.exists") as mock_exists, \
         patch("orchestrator.src.explorers.kmc_explorer.read") as mock_read:

        mock_exists.return_value = True

        result = explorer.explore(
            current_structure="input.data",
            potential_path="pot.yace",
            iteration=1
        )

        assert result.status == ExplorationStatus.NO_EVENT
        assert result.final_structure == "input.data"
