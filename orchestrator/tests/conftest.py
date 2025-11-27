import pytest
from unittest.mock import MagicMock
import shutil
from pathlib import Path
import sys

# Add repo root to path (so 'shared' can be imported)
sys.path.append(str(Path(__file__).parent.parent.parent))
# Add orchestrator to path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def mock_subprocess(mocker):
    """
    Mocks subprocess.run and returns the mock object.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0
    return mock_run

@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Creates a temporary data directory.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
