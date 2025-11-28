import pytest
from ase import Atoms
from shared.autostructure.alloy import AlloyGenerator

def test_generate_stacking_faults():
    """Test that stacking faults are generated without error using ASE API."""
    # Create a simple FCC structure (e.g., Al)
    from ase.build import bulk
    atoms = bulk("Al", "fcc", a=4.05)

    # Initialize Generator
    # Note: AlloyGenerator constructor handles Structure or Atoms
    # Provide valid dummy LJ params to satisfy PreOptimizer
    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    gen = AlloyGenerator(atoms, lj_params=lj_params)

    # Configure features
    gen.config = {"features": ["stacking_faults"]}

    # Call generate_all
    gen.generate_all()

    # Check if a structure was added (stacking fault + possibly others if logic defaults to them, but here strict?)
    # With the logic: `if features and "stacking_faults" in features: ...` it should run.
    # But wait, `if not features or ...` logic in my code means if features IS present, others won't run unless in list.
    # So `random_substitution` won't run if not in list.

    # Check if stacking fault type exists in generated structures
    types = [s.info.get("type") for s in gen.generated_structures]
    assert "stacking_fault" in types
    assert len(gen.generated_structures) > 0

    # Check metadata
    sf = gen.generated_structures[0]
    assert sf.info.get("type") == "stacking_fault"

    # Check atoms count (should be 6x supercell)
    # Original has 1 atom. Supercell (1,1,6) -> 6 atoms.
    assert len(sf) == 6
