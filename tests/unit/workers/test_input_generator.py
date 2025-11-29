
import sys
import os
import pytest
from unittest.mock import MagicMock

# Adjust path to find the worker code
sys.path.append(os.path.join(os.getcwd(), "workers/lammps_worker/src"))

from input_generator import LAMMPSInputGenerator

@pytest.fixture
def mock_params():
    lj_params = {"epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5, "shift_energy": True}
    md_params = {
        "elements": ["Cu"],
        "timestep": 0.001,
        "temperature": 300,
        "pressure": 1.0,
        "masses": {"Cu": 63.55}
    }
    return lj_params, md_params

def test_generate_delta_learning_on(mock_params, tmp_path):
    lj, md = mock_params
    generator = LAMMPSInputGenerator(lj, md, delta_learning_mode=True)

    filepath = tmp_path / "in.lammps"
    generator.generate(
        filepath=str(filepath),
        potential_path="pot.ace",
        steps=100,
        gamma_threshold=0.1,
        input_structure="struct.data",
        is_restart=False
    )

    with open(filepath, "r") as f:
        content = f.read()

    assert "pair_style hybrid/overlay pace/extrapolation lj/cut" in content
    # Check for pair_coeff for both
    # pace
    assert "pair_coeff * * pace/extrapolation pot.ace Cu" in content
    # lj - Since we have Cu (type 1), look for pair_coeff 1 1 lj/cut ... or wildcards
    # The generator writes explicit pair coefficients for elements if epsilon/sigma are dicts.
    # In my mock they are scalars.
    # _write_lj_coeffs falls back to wildcard if scalar.
    assert "pair_coeff * * lj/cut 1.0 1.0" in content

def test_generate_delta_learning_off(mock_params, tmp_path):
    lj, md = mock_params
    generator = LAMMPSInputGenerator(lj, md, delta_learning_mode=False)

    filepath = tmp_path / "in.lammps"
    generator.generate(
        filepath=str(filepath),
        potential_path="pot.ace",
        steps=100,
        gamma_threshold=0.1,
        input_structure="struct.data",
        is_restart=False
    )

    with open(filepath, "r") as f:
        content = f.read()

    assert "pair_style pace/extrapolation" in content
    assert "pair_style hybrid/overlay" not in content
    # lj/cut might appear in comments if any, but shouldn't be in pair_style
    # strict check on pair_style line
    assert "pair_style lj/cut" not in content
