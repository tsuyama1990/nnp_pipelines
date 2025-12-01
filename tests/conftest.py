import sys
import os
import pytest
import subprocess
from unittest.mock import MagicMock

# --- Path Patching ---
# We need to add worker source directories to sys.path so tests can import from 'src'
# This is a temporary fix for legacy tests.
WORKER_PATHS = [
    "workers/al_md_kmc_worker/src",
    "workers/gen_worker/src",
    "workers/pace_worker/src",
    "workers/dft_worker/src",
    # Add root for shared
    ".",
]

@pytest.fixture(scope="session", autouse=True)
def patch_sys_path():
    """Automatically adds worker source directories to sys.path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for rel_path in WORKER_PATHS:
        full_path = os.path.join(project_root, rel_path)
        if full_path not in sys.path:
            sys.path.insert(0, full_path)


@pytest.fixture(autouse=True)
def mock_subprocess_run(monkeypatch):
    """
    Mocks subprocess.run to intercept Docker calls and return success.
    This prevents actual Docker commands (which fail in CI/sandbox) from executing.
    """
    original_run = subprocess.run

    def side_effect(command, *args, **kwargs):
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Intercept Docker run commands
        if "docker run" in cmd_str:
            print(f"[MOCK] Intercepted Docker command: {cmd_str}")
            # Return a mock CompletedProcess with returncode 0 (success)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"Mock Success", stderr=b"")

        # Allow other commands to pass through or mock them as needed
        # For safety in this environment, we might want to mock everything,
        # but let's stick to the prompt's request for Docker.
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", side_effect)
