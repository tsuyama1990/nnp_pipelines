import unittest
from pydantic import ValidationError
from hypothesis import given, strategies as st
from shared.core.config import MDParams, DFTParams, Config, MetaConfig, ExperimentConfig, ALParams, LJParams, ACEModelParams, TrainingParams

class TestConfigRobustness(unittest.TestCase):
    def test_md_params_validation(self):
        """Test that invalid MD parameters raise ValidationError."""

        # Test negative timestep
        with self.assertRaises(ValidationError):
            MDParams(
                timestep=-1.0, # Invalid
                temperature=300,
                pressure=0,
                n_steps=100,
                elements=["Al"],
                initial_structure="structure.xyz",
                masses={"Al": 26.98}
            )

        # Test zero steps
        with self.assertRaises(ValidationError):
            MDParams(
                timestep=1.0,
                temperature=300,
                pressure=0,
                n_steps=0, # Invalid (gt=0)
                elements=["Al"],
                initial_structure="structure.xyz",
                masses={"Al": 26.98}
            )

        # Test empty elements
        with self.assertRaises(ValidationError):
            MDParams(
                timestep=1.0,
                temperature=300,
                pressure=0,
                n_steps=100,
                elements=[], # Invalid (min_length=1)
                initial_structure="structure.xyz",
                masses={"Al": 26.98}
            )

    @given(
        timestep=st.floats(min_value=0.1, max_value=5.0),
        temperature=st.floats(min_value=0, max_value=5000),
        pressure=st.floats(min_value=0, max_value=1000),
        n_steps=st.integers(min_value=1, max_value=1000000)
    )
    def test_md_params_hypothesis(self, timestep, temperature, pressure, n_steps):
        """Property-based testing for MDParams."""
        params = MDParams(
            timestep=timestep,
            temperature=temperature,
            pressure=pressure,
            n_steps=n_steps,
            elements=["Al"],
            initial_structure="structure.xyz",
            masses={"Al": 26.98}
        )
        self.assertEqual(params.timestep, timestep)
        self.assertEqual(params.temperature, temperature)

    def test_dft_params_validation(self):
        """Test DFT Params validation."""
        # Test negative kpoint density
        with self.assertRaises(ValidationError):
            DFTParams(kpoint_density=-0.04, auto_physics=True)

    @given(kpoint_density=st.floats(min_value=0.01, max_value=1.0))
    def test_dft_params_hypothesis(self, kpoint_density):
        params = DFTParams(kpoint_density=kpoint_density, auto_physics=True)
        self.assertEqual(params.kpoint_density, kpoint_density)

if __name__ == "__main__":
    unittest.main()
