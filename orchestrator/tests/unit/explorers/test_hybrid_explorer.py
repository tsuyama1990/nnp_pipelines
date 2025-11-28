import pytest
from unittest.mock import MagicMock
from orchestrator.src.explorers.hybrid_explorer import HybridExplorer
from orchestrator.src.interfaces.explorer import ExplorationResult, ExplorationStatus

@pytest.fixture
def mock_md_explorer():
    return MagicMock()

@pytest.fixture
def mock_kmc_explorer():
    return MagicMock()

def test_hybrid_explorer_success(mock_md_explorer, mock_kmc_explorer):
    explorer = HybridExplorer(mock_md_explorer, mock_kmc_explorer)

    # MD Success
    md_result = ExplorationResult(status=ExplorationStatus.SUCCESS, final_structure="md_out.data")
    mock_md_explorer.explore.return_value = md_result

    # KMC Success
    kmc_result = ExplorationResult(status=ExplorationStatus.SUCCESS, final_structure="kmc_out.data")
    mock_kmc_explorer.explore.return_value = kmc_result

    result = explorer.explore(
        current_structure="input.data",
        potential_path="pot.yace",
        iteration=1
    )

    # Verify sequence
    mock_md_explorer.explore.assert_called_once()
    mock_kmc_explorer.explore.assert_called_once_with(
        "md_out.data", "pot.yace", 1
    )

    assert result == kmc_result

def test_hybrid_explorer_md_uncertain(mock_md_explorer, mock_kmc_explorer):
    explorer = HybridExplorer(mock_md_explorer, mock_kmc_explorer)

    # MD Uncertain
    md_result = ExplorationResult(status=ExplorationStatus.UNCERTAIN)
    mock_md_explorer.explore.return_value = md_result

    result = explorer.explore(
        current_structure="input.data",
        potential_path="pot.yace",
        iteration=1
    )

    # Verify KMC not called
    mock_kmc_explorer.explore.assert_not_called()
    assert result == md_result
