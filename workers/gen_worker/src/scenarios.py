"""Module for scenario-driven structure generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ase import Atoms
from ase.build import bulk, surface, stack
import numpy as np

from candidate import RandomStructureGenerator

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
                print(f"Surface generation failed for {element} {hkl}: {e}")
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

        try:
            gen = RandomStructureGenerator(elements=elements, max_atoms=max_atoms)
            return gen.generate(n_structures)
        except ImportError:
            print("PyXtal not available.")
            return []

class ScenarioFactory:
    """Factory to create scenario generators."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseScenario:
        t = config.get("type")
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
        else:
            raise ValueError(f"Unknown scenario type: {t}")
