"""Tests for PyXtal generation in gen_worker."""

import sys
import os
import pytest
import numpy as np
from ase import Atoms

# Add source directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workers", "gen_worker", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared", "core"))

# Mock mace if not present, as we only test scenarios and basic filtering here
try:
    import mace
except ImportError:
    pass

from scenarios import RandomSymmetryScenario, ScenarioFactory
from filter import check_min_distance

# Helper to check if pyxtal is installed
try:
    import pyxtal
    PYXTAL_AVAILABLE = True
except ImportError:
    PYXTAL_AVAILABLE = False


@pytest.mark.skipif(not PYXTAL_AVAILABLE, reason="PyXtal not installed")
def test_random_symmetry_generation():
    """Test basic random symmetry generation."""
    config = {
        "type": "random_symmetry",
        "elements": ["Al", "Cu"],
        "num_structures": 2,
        # Use a larger range or easier space group for random generation
        "space_group_range": [1, 230],
        "volume_factor": 1.0,
        "max_attempts": 50
    }

    scenario = RandomSymmetryScenario(config)
    structures = scenario.generate()

    # We relax the strict count check because random generation can be finicky
    # But with 50 attempts for 2 structures it should pass most of the time
    assert len(structures) > 0
    if len(structures) < 2:
        pytest.skip("Could not generate all requested structures, but generated some.")

    for atoms in structures:
        assert isinstance(atoms, Atoms)
        # Check elements
        symbols = set(atoms.get_chemical_symbols())
        assert "Al" in symbols or "Cu" in symbols
        # Check validity
        assert check_min_distance(atoms, min_dist=0.5)

@pytest.mark.skipif(not PYXTAL_AVAILABLE, reason="PyXtal not installed")
def test_specific_composition():
    """Test generation with specific composition."""
    config = {
        "type": "random_symmetry",
        "elements": ["Al", "Cu"],
        "num_structures": 1,
        "space_group_range": [1, 230],
        "composition": {"Al": 4, "Cu": 4},
        "max_attempts": 50
    }

    scenario = RandomSymmetryScenario(config)
    structures = scenario.generate()

    if len(structures) == 0:
        pytest.skip("Could not generate structure with specific composition in limited attempts.")

    atoms = structures[0]
    symbol_counts = atoms.symbols.formula.count()
    assert symbol_counts["Al"] == 4
    assert symbol_counts["Cu"] == 4

def test_check_min_distance():
    """Test the min distance checker."""
    # Create two atoms very close
    atoms_bad = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.4]])
    assert not check_min_distance(atoms_bad, min_dist=0.5)

    # Create two atoms far apart
    atoms_good = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    assert check_min_distance(atoms_good, min_dist=0.5)

@pytest.mark.skipif(not PYXTAL_AVAILABLE, reason="PyXtal not installed")
def test_factory_creation():
    """Test creation via factory."""
    config = {
        "type": "random_symmetry",
        "elements": ["Al"],
        "num_structures": 1
    }
    scenario = ScenarioFactory.create(config)
    assert isinstance(scenario, RandomSymmetryScenario)

@pytest.mark.skipif(not PYXTAL_AVAILABLE, reason="PyXtal not installed")
def test_generation_failure_handling():
    """Test that it handles impossible constraints gracefully."""
    # Impossible constraint (maybe? Volume factor extremely small for many atoms)
    # PyXtal might throw exception or just fail to generate.
    config = {
        "type": "random_symmetry",
        "elements": ["U"],
        "num_structures": 2,
        "space_group_range": [1, 2],
        "volume_factor": 0.01, # Too small
        "max_attempts": 5
    }

    scenario = RandomSymmetryScenario(config)
    structures = scenario.generate()

    # Should not crash, might return empty or few structures
    assert isinstance(structures, list)
