
import pytest
from unittest.mock import MagicMock
from ase import Atoms
from orchestrator.src.services.al_service import ActiveLearningService
from shared.core.config import Config, ALParams

def test_trigger_al_rescue_injection():
    # Setup mocks
    mock_sampler = MagicMock()
    mock_generator = MagicMock()
    mock_labeler = MagicMock()
    mock_trainer = MagicMock()
    mock_validator = MagicMock()
    mock_config = MagicMock(spec=Config)
    mock_config.al_params = MagicMock(spec=ALParams)
    mock_config.al_params.num_parallel_labeling = 1
    mock_config.al_params.n_clusters = 1
    mock_config.al_params.gamma_upper_bound = 1.0 # Limit
    mock_config.md_params = MagicMock()
    mock_config.md_params.elements = ["Cu"]
    mock_config.training_params = MagicMock()
    mock_config.training_params.pruning_frequency = 0

    # Mock MDService
    mock_md_service = MagicMock()

    # Init Service
    service = ActiveLearningService(
        mock_sampler, mock_generator, mock_labeler, mock_trainer, mock_validator, mock_config,
        md_service=mock_md_service
    )

    # Input with High Gamma
    high_gamma_atoms = Atoms('Cu1', positions=[[0,0,0]])
    high_gamma_atoms.info['max_gamma'] = 5.0 # Exceeds 1.0

    # Sampler returns this structure
    mock_sampler.sample.return_value = [(high_gamma_atoms, 0)]

    # Mock Rescue
    rescued_atoms = Atoms('Cu1', positions=[[0.1,0.1,0.1]])
    mock_md_service.run_rescue.return_value = rescued_atoms

    # Mock Generator (should be called with RESCUED atoms)
    mock_generator.generate_cell.return_value = rescued_atoms

    # Run
    service.trigger_al(
        [high_gamma_atoms], "pot.yace", "pot.yaml", None, Path("."), 1
    )

    # Verify Rescue Called
    mock_md_service.run_rescue.assert_called_once()

    # Verify Generator called with RESCUED atoms
    args, _ = mock_generator.generate_cell.call_args
    # args[0] is the atoms object
    assert np.allclose(args[0].positions, rescued_atoms.positions)

import numpy as np
from pathlib import Path
