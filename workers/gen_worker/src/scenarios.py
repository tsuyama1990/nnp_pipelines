"""Module for scenario-driven structure generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from ase import Atoms
from ase.build import bulk, surface, stack
import numpy as np

from candidate import RandomStructureGenerator
from shared.autostructure.alloy import AlloyGenerator
from shared.autostructure.ionic import IonicGenerator
from shared.autostructure.covalent import CovalentGenerator

# Try to import pyxtal
try:
    from pyxtal import pyxtal
    PYXTAL_AVAILABLE = True
except ImportError:
    PYXTAL_AVAILABLE = False


class BaseScenario(ABC):
    """Abstract base class for scenario generators."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the scenario generator.

        Args:
            config: Configuration dictionary for the scenario.
        """
        self.config = config

    @abstractmethod
    def generate(self) -> List[Atoms]:
        """Generate structures based on the scenario.

        Returns:
            List[Atoms]: List of generated structures.
        """
        pass


class InterfaceGenerator(BaseScenario):
    """Generates interface structures by stacking two materials."""

    def generate(self) -> List[Atoms]:
        substrate_conf = self.config.get("substrate", {})
        layer_conf = self.config.get("layer", {})
        vacuum = self.config.get("vacuum", 10.0)

        # Create substrate
        sub = bulk(
            substrate_conf.get("symbol", "MgO"),
            crystalstructure=substrate_conf.get("structure", "rocksalt"),
            a=substrate_conf.get("lattice", 4.21),
            orthorhombic=True,
        )
        # Create layer
        lattice_layer = layer_conf.get("lattice", 3.8)
        if isinstance(lattice_layer, list):
            lattice_layer_val = lattice_layer[0]
        else:
            lattice_layer_val = lattice_layer

        lay = bulk(
            layer_conf.get("symbol", "Fe"),
            crystalstructure=layer_conf.get("structure", "bcc"),
            a=lattice_layer_val,
            orthorhombic=True,
        )

        try:
            interface = stack(sub, lay, maxstrain=0.15)
            if vacuum > 0:
                interface.center(vacuum=vacuum, axis=2)
            return [interface]
        except Exception as e:
            return []


class SurfaceGenerator(BaseScenario):
    """Generates surface structures."""

    def generate(self) -> List[Atoms]:
        element = self.config.get("element", "Pt")
        indices_list = self.config.get("indices", [[1, 1, 1]])
        layers = self.config.get("layers", 4)
        vacuum = self.config.get("vacuum", 10.0)

        structures = []
        for hkl in indices_list:
            try:
                b = bulk(element)
                surf = surface(b, tuple(hkl), layers)
                surf.center(vacuum=vacuum, axis=2)
                structures.append(surf)
            except Exception as e:
                logging.error(f"Surface generation failed for {element} {hkl}: {e}", exc_info=True)
                continue
        return structures


class DefectGenerator(BaseScenario):
    """Generates structures with defects (vacancies or impurities)."""

    def generate(self) -> List[Atoms]:
        base_conf = self.config.get("base", {})
        defect_type = self.config.get("defect_type", "vacancy")
        species = self.config.get("species", "O")
        concentration = self.config.get("concentration", 0.02)

        atoms = bulk(
            base_conf.get("symbol", "MgO"),
            crystalstructure=base_conf.get("structure", "rocksalt"),
            a=base_conf.get("lattice", 4.21),
        )

        size = base_conf.get("size", [3, 3, 3])
        atoms = atoms.repeat(size)

        num_sites = len(atoms)
        num_defects = max(1, int(num_sites * concentration))

        generated = []
        indices = list(range(num_sites))
        if species:
            indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]

        if len(indices) < num_defects:
            return []

        np.random.shuffle(indices)
        target_indices = indices[:num_defects]

        new_atoms = atoms.copy()

        if defect_type == "vacancy":
            del new_atoms[target_indices]
            generated.append(new_atoms)

        return generated


class GrainBoundaryGenerator(BaseScenario):
    """Generates grain boundary structures."""

    def generate(self) -> List[Atoms]:
        return []

class RandomScenario(BaseScenario):
    """Generates random structures using PyXtal."""

    def generate(self) -> List[Atoms]:
        elements = self.config.get("elements", [])
        n_structures = self.config.get("n_structures", 10)
        max_atoms = self.config.get("max_atoms", 8)

        # Deprecated: usage of candidate.RandomStructureGenerator
        # Keeping it for backward compatibility if configured explicitly as 'random'
        # without using the new RandomSymmetryScenario logic
        try:
            gen = RandomStructureGenerator(elements=elements, max_atoms=max_atoms)
            return gen.generate(n_structures)
        except ImportError:
            logging.error("PyXtal not available.", exc_info=True)
            return []

class RandomSymmetryScenario(BaseScenario):
    """Generates random structures using PyXtal with explicit symmetry control."""

    def generate(self) -> List[Atoms]:
        if not PYXTAL_AVAILABLE:
            logging.warning("PyXtal not available. Skipping RandomSymmetryScenario.")
            return []

        elements = self.config.get("elements", [])
        num_structures = self.config.get("num_structures", 5)
        space_group_range = self.config.get("space_group_range", [1, 230])
        volume_factor = self.config.get("volume_factor", 1.0)
        max_attempts = self.config.get("max_attempts", 50)

        # Determine number of atoms per species
        composition = self.config.get("composition")

        # Parse composition once if possible
        species_fixed = []
        num_ions_fixed = []
        if composition:
            species_fixed = list(composition.keys())
            num_ions_fixed = list(composition.values())

        if not composition and not elements:
             logging.error("No elements provided for RandomSymmetryScenario.")
             return []

        # Parse space group range once
        if len(space_group_range) != 2:
             sg_start, sg_end = 1, 230
        else:
             sg_start, sg_end = space_group_range[0], space_group_range[1]

        generated_structures: List[Atoms] = []
        attempts = 0

        while len(generated_structures) < num_structures and attempts < max_attempts * num_structures:
            attempts += 1

            # Select random space group
            sg = np.random.randint(sg_start, sg_end + 1)

            # Select composition if not fixed
            if composition:
                species = species_fixed
                num_ions = num_ions_fixed
            else:
                species = elements
                # Random number of ions between 1 and 4 per species
                num_ions = [np.random.randint(1, 5) for _ in elements]

            try:
                struc = pyxtal()
                # from_random arguments: dim, group, species, numIons, factor
                struc.from_random(3, sg, species, num_ions, factor=volume_factor)

                if struc.valid:
                    atoms = struc.to_ase(resort=False)
                    generated_structures.append(atoms)

            except Exception as e:
                # PyXtal generation failed for this attempt
                continue

        if len(generated_structures) < num_structures:
            logging.warning(f"Only generated {len(generated_structures)}/{num_structures} structures after {attempts} attempts.")

        return generated_structures


class CrystalAwareScenario(BaseScenario):
    """Adapter for Crystal-Aware Generators from shared.autostructure."""

    def __init__(self, config: Dict[str, Any], generator_cls):
        super().__init__(config)
        self.generator_cls = generator_cls

    def generate(self) -> List[Atoms]:
        # Need a base structure to start with.
        # This wrapper assumes the Scenario is called with a 'base_structure' in config
        # or it creates a simple one if possible, or generates from scratch if the generator supports it.
        # Most shared.autostructure generators require a base_structure in __init__.
        # We need to bridge this.

        # If 'structure' is passed in config (as Atoms object or similar), use it.
        # But config usually comes from YAML.
        # Maybe we need to generate a seed structure first (e.g. from elements/lattice)?

        # For now, let's assume the config provides parameters to build a simple bulk,
        # or we reuse RandomSymmetry to get a seed, then apply the generator.

        # BUT, the prompt says: "If "metallic" -> Instantiate AlloyGenerator... Pass the generation params".
        # AlloyGenerator needs `base_structure`.

        # Let's create a simple bulk based on elements if not provided.
        elements = self.config.get("elements", [])
        if not elements:
            # Fallback
            return []

        # Create a simple random/bulk structure to seed the generator
        # Or check if 'structure' is in config.

        import ase.build

        # Try to build a simple structure
        # If multiple elements, maybe a random alloy or just the first element?
        # Let's use `bulk` for the first element as a placeholder seed if nothing else exists.
        try:
             # Use RandomSymmetry to get a valid starting point?
             # Or just a simple bulk.
             seed = ase.build.bulk(elements[0], cubic=True)
        except Exception as e:
             logging.error(f"Failed to create seed structure for CrystalAwareScenario: {e}")
             return []

        # Instantiate the generator
        gen = self.generator_cls(seed, lj_params=None)

        # Configure the generator?
        # The generator classes (AlloyGenerator etc) use `generate_all()` which calls specific methods.
        # We might want to selectively call methods based on config, but `generate_all` does everything.
        # The prompt says: "Pass the generation params from config to these generators."
        # The generator base class has `self.config` but it's not populated in __init__ in the current code I read.
        # Wait, BaseGenerator has `self.config = {}`.
        # So we can set it.

        gen.config = self.config

        return gen.generate_all()


class ScenarioFactory:
    """Factory to create scenario generators."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseScenario:
        t = config.get("type")
        crystal_type = config.get("crystal_type")

        # Dispatch based on crystal_type if present
        if crystal_type == "metallic":
             return CrystalAwareScenario(config, AlloyGenerator)
        elif crystal_type == "ionic":
             return CrystalAwareScenario(config, IonicGenerator)
        elif crystal_type == "covalent":
             return CrystalAwareScenario(config, CovalentGenerator)

        # Legacy dispatch
        if t == "interface":
            return InterfaceGenerator(config)
        elif t == "surface":
            return SurfaceGenerator(config)
        elif t == "defect":
            return DefectGenerator(config)
        elif t == "grain_boundary":
            return GrainBoundaryGenerator(config)
        elif t == "random":
            return RandomScenario(config)
        elif t == "random_symmetry":
            return RandomSymmetryScenario(config)
        else:
            # If no type matches, default to Random or raise
            # If crystal_type was "random", we might fall through here if t is missing.
            if crystal_type == "random":
                 return RandomScenario(config)
            raise ValueError(f"Unknown scenario type: {t}")
