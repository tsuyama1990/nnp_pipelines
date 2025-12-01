
import sys
import os
import pytest

# Ensure root is in path
sys.path.append(os.getcwd())

from shared.core.config import ACEModelParams

def test_ace_model_params_default():
    """Verify delta_learning_mode is True by default."""
    params = ACEModelParams()
    assert params.delta_learning_mode is True
