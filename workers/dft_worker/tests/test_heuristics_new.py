
import pytest
import numpy as np
from ase import Atoms
from workers.dft_worker.src.heuristics import PymatgenHeuristics

def test_heuristics_electronic_type():
    # Test Metal (Transition Metal)
    fe_atoms = Atoms('Fe2', positions=[[0,0,0], [2,0,0]])
    assert PymatgenHeuristics._estimate_electronic_type(fe_atoms) == "metal"
    params = PymatgenHeuristics.get_recommended_params(fe_atoms)
    assert params["system"]["occupations"] == "smearing"

    # Test Metal (Main Group only)
    al_atoms = Atoms('Al4', positions=np.random.rand(4,3))
    assert PymatgenHeuristics._estimate_electronic_type(al_atoms) == "metal"

    # Test Insulator (Main Group + Anion)
    sio2_atoms = Atoms('Si1O2', positions=[[0,0,0], [1.5,0,0], [0,1.5,0]])
    assert PymatgenHeuristics._estimate_electronic_type(sio2_atoms) == "insulator"
    params = PymatgenHeuristics.get_recommended_params(sio2_atoms)
    assert params["system"]["occupations"] == "fixed"

    # Test Magnetism
    assert params["magnetism"]["nspin"] == 1 # SiO2 non-magnetic

    fe_o_atoms = Atoms('Fe1O1', positions=[[0,0,0], [2,0,0]])
    # Should be metal because of Fe?
    # Actually Transition Metal check takes precedence in current logic -> "metal"
    # This implies TMOs are treated as metallic by default which is safer for smearing convergence
    # unless strictly controlled.
    assert PymatgenHeuristics._estimate_electronic_type(fe_o_atoms) == "metal"
    params = PymatgenHeuristics.get_recommended_params(fe_o_atoms)
    assert params["magnetism"]["nspin"] == 2
    assert "Fe" in params["magnetism"]["moments"]
