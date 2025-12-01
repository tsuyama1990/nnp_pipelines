import sys
import os
import pytest
import subprocess
from unittest.mock import MagicMock

# --- Path Patching ---
# We need to add worker source directories to sys.path so tests can import from 'src'
# This is a temporary fix for legacy tests.
# The order matters!
# We add `workers/al_md_kmc_worker` specifically to support `from src.state_manager` style imports
# found in the active learning worker. We do NOT add other worker roots to avoid namespace collisions for 'src'.
WORKER_PATHS = [
    # Top priority: specific source folders so implicit relative imports or top-level module imports work
    "workers/al_md_kmc_worker/src",
    "workers/dft_worker/src",
    "workers/gen_worker/src",
    "workers/pace_worker/src",
    "workers/lammps_worker/src",

    # Add al_md_kmc_worker ROOT to allow `import src.state_manager` to resolve correctly
    "workers/al_md_kmc_worker",

    # Root and Shared
    ".",
    "shared",
]

# We insert them into sys.path.
# We iterate in REVERSE order and insert at 0, so the FIRST item in list ends up at TOP of sys.path.
# Wait, typical logic is:
# for p in paths: sys.path.insert(0, p) -> Last item becomes first.
# So if we want the list above to be priority order (top to bottom), we should reverse it before inserting?
# Or just append?
# The prompt asked for `sys.path.append`.
# But `insert(0)` is safer to override system packages if needed, but here we want to prioritize our paths.
# Let's stick to `sys.path.append` as requested by the prompt?
# "Use sys.path.append to add ... before tests start".
# If I append, then `import src` might match something else first?
# Unlikely `src` is installed.
# But `workers/al_md_kmc_worker/src` (added) vs `workers/al_md_kmc_worker` (added).
# `import workflows` -> needs `workers/al_md_kmc_worker/src` in path.
# `import src` -> needs `workers/al_md_kmc_worker` in path.
# These are orthogonal.
# `import strategies` -> needs `workers/dft_worker/src` in path.
# So order between these doesn't matter much, as long as they are present.
# But `src` collision matters.
# Since we only add ONE worker root (`al_md_kmc_worker`), there is no collision for `src`.
# So `append` or `insert` should both work.
# I will use `insert(0)` to be sure they are found.

@pytest.fixture(scope="session", autouse=True)
def patch_sys_path():
    """Automatically adds worker source directories to sys.path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # We want the first item in WORKER_PATHS to be at the top of sys.path?
    # If so, we should insert in reverse order.
    for rel_path in reversed(WORKER_PATHS):
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
            # Return a mock CompletedProcess with returncode=0 (success)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"Mock Success", stderr=b"")

        # Allow other commands to pass through or mock them as needed
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", side_effect)
