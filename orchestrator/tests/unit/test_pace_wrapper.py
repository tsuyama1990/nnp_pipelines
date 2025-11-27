import pytest
from pathlib import Path
from orchestrator.src.wrappers.pace_wrapper import PaceWorker

class TestPaceWorkerUnit:
    def test_train_command_generation(self, mock_subprocess, temp_data_dir):
        """
        Verify PaceWorker.train() generates correct Docker command.
        """
        # Arrange
        worker = PaceWorker(host_data_dir=temp_data_dir)
        dataset = "labeled_data.pckl"
        initial_pot = "old.yace"
        config = "config.yaml"
        meta = "meta.yaml"

        # Act
        worker.train(
            config_filename=config,
            meta_config_filename=meta,
            dataset_filename=dataset,
            initial_potential=initial_pot
        )

        # Assert
        # 1. subprocess.run called once
        assert mock_subprocess.call_count == 1

        # 2. Check arguments
        args, _ = mock_subprocess.call_args
        cmd_list = args[0]

        assert "docker" in cmd_list
        assert "run" in cmd_list
        # Volume mount check
        assert "-v" in cmd_list
        # Volume path string construction might vary slightly, checking presence
        assert f"{temp_data_dir}:/data" in cmd_list

        # Image check
        assert "pace_worker:latest" in cmd_list

        # Script and args check
        assert "/app/src/main.py" in cmd_list
        assert "train" in cmd_list
        assert f"/data/{dataset}" in cmd_list
        assert f"/data/{initial_pot}" in cmd_list
        assert f"/data/{config}" in cmd_list
