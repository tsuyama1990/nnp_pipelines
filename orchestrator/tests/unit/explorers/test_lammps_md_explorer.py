import pytest
from unittest.mock import MagicMock, ANY, patch
from pathlib import Path

from orchestrator.src.explorers.lammps_md_explorer import LammpsMDExplorer
from orchestrator.src.interfaces.explorer import ExplorationResult, ExplorationStatus

@pytest.fixture
def mock_md_service():
    return MagicMock()

@pytest.fixture
def mock_config():
    return MagicMock()

def test_lammps_md_explorer_success(mock_md_service, mock_config):
    explorer = LammpsMDExplorer(mock_md_service, mock_config)

    # Mock successful run
    mock_md_service.run_walkers.return_value = (True, [], False)

    # We need to simulate the restart file and dump file existence
    with patch("pathlib.Path.glob") as mock_glob, \
         patch("shutil.copy") as mock_copy, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.resolve") as mock_resolve:

        restart_path = MagicMock(spec=Path)
        restart_path.suffix = ".123"
        restart_path.name = "restart.chk.123"

        mock_glob.return_value = [restart_path]
        mock_exists.return_value = True # For dump file check
        mock_resolve.return_value = "/abs/path/dump.lammpstrj.123"

        result = explorer.explore(
            current_structure="input.data",
            potential_path="pot.yace",
            iteration=1
        )

        assert result.status == ExplorationStatus.SUCCESS
        # Should point to the dump file now
        assert "dump.lammpstrj.123" in str(result.final_structure)
        assert result.metadata["is_restart"] is True

        # Verify restart file was copied
        mock_copy.assert_called_once()

def test_lammps_md_explorer_uncertain(mock_md_service, mock_config):
    explorer = LammpsMDExplorer(mock_md_service, mock_config)

    # Mock uncertain run
    uncertain_struct = MagicMock()
    mock_md_service.run_walkers.return_value = (False, [uncertain_struct], False)

    result = explorer.explore(
        current_structure="input.data",
        potential_path="pot.yace",
        iteration=1
    )

    assert result.status == ExplorationStatus.UNCERTAIN
    assert result.uncertain_structures == [uncertain_struct]

def test_lammps_md_explorer_failed(mock_md_service, mock_config):
    explorer = LammpsMDExplorer(mock_md_service, mock_config)

    # Mock failed run
    mock_md_service.run_walkers.return_value = (False, [], True)

    result = explorer.explore(
        current_structure="input.data",
        potential_path="pot.yace",
        iteration=1
    )

    assert result.status == ExplorationStatus.FAILED
