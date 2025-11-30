import sys
import os
sys.path.append(os.getcwd())

import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from ase import Atoms
import numpy as np
from typing import List, Tuple

# Append worker path to sys.path to resolve 'src' imports correctly
import sys
import os
worker_path = os.path.join(os.getcwd(), 'workers/al_md_kmc_worker')
if worker_path not in sys.path:
    sys.path.append(worker_path)

from src.workflows.active_learning_loop import ActiveLearningOrchestrator
from src.interfaces.explorer import ExplorationResult, ExplorationStatus
from shared.core.config import Config, MDParams, ALParams, DFTParams, LJParams, TrainingParams, MetaConfig, ExperimentConfig, ACEModelParams, ExplorationParams
from shared.core.enums import SimulationState
from shared.core.interfaces import MDEngine, StructureGenerator, Labeler, Trainer

# MaxVolSampler tests removed due to namespace conflicts with 'src'.
# They should be in a separate test file running in pace_worker context.

@patch('src.workflows.active_learning_loop.os.chdir')
@patch('src.workflows.active_learning_loop.Path.mkdir')
@patch('src.workflows.active_learning_loop.Path.exists')
@patch('src.workflows.active_learning_loop.read')
def test_orchestrator_initial_asi_generation(mock_read, mock_exists, mock_mkdir, mock_chdir):
    # Setup Config
    meta_config = MetaConfig(
        dft={"command": "pw.x", "pseudo_dir": ".", "sssp_json_path": "sssp.json"},
        lammps={}
    )
    config = Config(
        meta=meta_config,
        experiment=ExperimentConfig(name="test", output_dir=Path("output")),
        exploration=ExplorationParams(),
        md_params=MDParams(
            timestep=1.0, temperature=300, pressure=1.0, n_steps=100,
            elements=["Al"], initial_structure="start.xyz", masses={"Al": 26.98}
        ),
        al_params=ALParams(
            gamma_threshold=0.1, n_clusters=2, r_core=3.0, box_size=10.0,
            initial_potential="pot.yace", potential_yaml_path="pot.yaml",
            initial_dataset_path="data.pckl", initial_active_set_path=None
        ),
        dft_params=DFTParams(),
        lj_params=LJParams(epsilon=1.0, sigma=1.0, cutoff=2.5),
        training_params=TrainingParams(),
        ace_model=ACEModelParams()
    )

    # Mocks
    al_service = MagicMock()
    explorer = MagicMock()
    state_manager = MagicMock()

    # Need to setup state_manager mock to return dict
    state_manager.load.return_value = {"iteration": 0, "current_potential": None, "current_asi": None}

    # Mock Path.exists logic
    mock_exists.return_value = True

    orch = ActiveLearningOrchestrator(
        config, al_service, explorer, state_manager
    )
    orch.al_config_path = MagicMock()
    orch.al_config_path.exists.return_value = True

    # Mocking resolve_path to return dummy paths
    orch._resolve_path = MagicMock(side_effect=lambda x, y: Path(x))

    # Mock trainer.update_active_set (part of al_service)
    al_service.trainer.update_active_set.return_value = "generated.asi"

    # We must NOT suppress exceptions here. The loop runs indefinitely if everything works.
    # To test 'run' without infinite loop, we must break it.
    # We can make explorer.explore return FAILED or AL logic break.

    # Let's verify `run` by breaking loop via exception, BUT we want to verify initial asi generation.
    # The generation happens BEFORE the loop.

    # Mock explorer to return FAILED status to break loop
    explorer.explore.return_value = ExplorationResult(status=ExplorationStatus.FAILED)

    orch.run()

    # Verify update_active_set called
    # With iteration=0 and current_asi=None and initial_dataset_path set, it SHOULD be called.
    al_service.trainer.update_active_set.assert_called_once_with("data.pckl", "pot.yaml")
