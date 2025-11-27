import pytest
from pathlib import Path
from unittest.mock import MagicMock
from orchestrator.src.wrappers.gen_wrapper import GenWorker

# Integration test marker (requires setup in pytest.ini or conftest or just -m)
@pytest.mark.integration
class TestGenWorkerIntegration:
    def test_generate_structure_flow(self, mock_subprocess, temp_data_dir):
        """
        Verify GenWorker flow with mocked Docker execution simulating file creation.
        """
        # Arrange
        worker = GenWorker(host_data_dir=temp_data_dir)
        config_name = "gen_config.yaml"
        output_name = "initial_structures.xyz"
        expected_output = temp_data_dir / output_name

        # Define side effect to simulate Docker creating the file
        def docker_side_effect(*args, **kwargs):
            # args[0] is the command list
            cmd_list = args[0]
            # verify command looks right inside side_effect if needed
            if "generate" in cmd_list and f"/data/{output_name}" in cmd_list:
                expected_output.touch()
            return MagicMock(returncode=0)

        mock_subprocess.side_effect = docker_side_effect

        # Act
        worker.generate(config_filename=config_name, output_filename=output_name)

        # Assert
        # 1. subprocess called
        assert mock_subprocess.called

        # 2. File exists (simulated)
        assert expected_output.exists()
