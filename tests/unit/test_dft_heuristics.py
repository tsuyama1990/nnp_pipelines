import unittest
from ase import Atoms
from workers.dft_worker.src.heuristics import PymatgenHeuristics

class TestDFTHeuristics(unittest.TestCase):

    def test_estimate_electronic_type(self):
        # 1. Pure Metal (Fe) -> metal
        fe_bulk = Atoms("Fe2", positions=[[0,0,0], [1,1,1]])
        self.assertEqual(PymatgenHeuristics._estimate_electronic_type(fe_bulk), "metal")

        # 2. Main Group Metal (Al) -> metal
        al_bulk = Atoms("Al2", positions=[[0,0,0], [1,1,1]])
        self.assertEqual(PymatgenHeuristics._estimate_electronic_type(al_bulk), "metal")

        # 3. Main Group Insulator (SiO2) -> insulator
        sio2 = Atoms("Si1O2", positions=[[0,0,0], [1,0,0], [0,1,0]])
        self.assertEqual(PymatgenHeuristics._estimate_electronic_type(sio2), "insulator")

        # 4. Transition Metal Oxide (FeO) -> insulator (Epic 3 Change)
        feo = Atoms("Fe1O1", positions=[[0,0,0], [1,1,1]])
        self.assertEqual(PymatgenHeuristics._estimate_electronic_type(feo), "insulator")

    def test_get_recommended_params_afm(self):
        # Epic 3: Verify alternating spins
        # Fe chain: Fe Fe Fe Fe
        atoms = Atoms("Fe4", positions=[[0,0,0], [1,0,0], [2,0,0], [3,0,0]])

        rec = PymatgenHeuristics.get_recommended_params(atoms)

        # Check type inference
        self.assertEqual(rec["system"]["occupations"], "smearing") # Fe pure is metal

        # Check magnetism
        self.assertEqual(rec["magnetism"]["nspin"], 2)

        moments = rec["magnetism"]["initial_magnetic_moments"]
        self.assertEqual(len(moments), 4)

        # Fe magnitude is 5.0 in FALLBACK
        # Should be +5, -5, +5, -5
        self.assertEqual(moments[0], 5.0)
        self.assertEqual(moments[1], -5.0)
        self.assertEqual(moments[2], 5.0)
        self.assertEqual(moments[3], -5.0)

    def test_get_recommended_params_tmo(self):
        # TMO (FeO) -> Insulator + AFM
        atoms = Atoms("Fe2O2", positions=[[0,0,0], [2,0,0], [1,0,0], [3,0,0]])
        # Symbols: Fe, Fe, O, O (ASE usually groups by species or keeps order?)
        # Atoms("Fe2O2") usually creates Fe Fe O O if using formula
        # Let's verify order
        syms = atoms.get_chemical_symbols()
        # Expecting Fe, Fe, O, O

        rec = PymatgenHeuristics.get_recommended_params(atoms)

        # 1. Insulator check
        self.assertEqual(rec["system"]["occupations"], "fixed")

        # 2. AFM check
        moments = rec["magnetism"]["initial_magnetic_moments"]
        # Fe indices should alternate
        # O indices should be 0

        # Indices of Fe in "Fe2O2" are 0 and 1
        self.assertEqual(moments[0], 5.0)
        self.assertEqual(moments[1], -5.0)
        self.assertEqual(moments[2], 0.0)
        self.assertEqual(moments[3], 0.0)

if __name__ == '__main__':
    unittest.main()
