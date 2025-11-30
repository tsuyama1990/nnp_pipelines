
import pytest
import numpy as np
from ase import Atoms
from workers.pace_worker.src.sampling.strategies.max_gamma import MaxGammaSampler

def test_max_gamma_sampler_local():
    sampler = MaxGammaSampler()

    # Structure 1: High gamma on atom 0, low on atom 1
    a1 = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    a1.new_array('gamma', np.array([5.0, 0.1]))

    # Structure 2: Low gamma on atom 0, High on atom 1
    a2 = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    a2.new_array('gamma', np.array([0.1, 5.0]))

    candidates = [a1, a2]

    # Case 1: Mask only Atom 0 (Index 0)
    # a1 should have max_gamma=5.0, a2 max_gamma=0.1. a1 wins.
    selected = sampler.sample(candidates, 1, atom_indices=[0])
    assert len(selected) == 1
    assert selected[0][1] == 0 # Index of a1

    # Case 2: Mask only Atom 1 (Index 1)
    # a1 should have max_gamma=0.1, a2 max_gamma=5.0. a2 wins.
    selected = sampler.sample(candidates, 1, atom_indices=[1])
    assert len(selected) == 1
    assert selected[0][1] == 1 # Index of a2
