import pytest
from unittest.mock import MagicMock
from ase import Atoms
from ase.calculators.espresso import Espresso
from workers.dft_worker.src.configurator import DFTConfigurator
from shared.core.config import DFTParams, MetaConfig

from unittest.mock import patch

@patch("workers.dft_worker.src.configurator.load_sssp_database")
@patch("workers.dft_worker.src.configurator.validate_pseudopotentials")
@patch("workers.dft_worker.src.configurator.get_pseudopotentials_dict")
@patch("workers.dft_worker.src.configurator.calculate_cutoffs")
def test_dft_configurator_metallic(mock_cutoffs, mock_get_pseudo, mock_validate, mock_load_sssp):
    """Test that metallic settings are applied correctly."""
    mock_cutoffs.return_value = (30, 200) # wfc, rho
    mock_get_pseudo.return_value = {"Al": "Al.upf"}
    mock_load_sssp.return_value = {}

    params = DFTParams()
    meta = MetaConfig(dft={"sssp_json_path": "sssp.json"}, lammps={})

    metallic_settings = {
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.02,
        "mixing_beta": 0.4
    }

    configurator = DFTConfigurator(params, meta, type_dft_settings=metallic_settings)

    atoms = Atoms("Al", positions=[[0,0,0]])
    elements = ["Al"]

    # Mock pymatgen heuristics or let it run (it uses basic rules)
    # We rely on the fact that build returns an Espresso object with input_data
    calc, _ = configurator.build(atoms, elements)

    assert isinstance(calc, Espresso)

    # ASE Espresso calculator stores parameters in 'parameters' attribute, not 'input_data' directly exposed?
    # Or in 'input_data'. Let's check how ASE stores it.
    # Actually, recent ASE versions put it in calc.parameters or calc.input_data depending on initialization.
    # But checking source code of DFTConfigurator, it passes input_data to constructor.
    # In ASE, calc.input_data IS usually available for Espresso.
    # Maybe we need to check calc.parameters["input_data"]?

    # Debugging: check dir(calc) if possible, but simpler to check calc.parameters
    input_data = calc.parameters.get("input_data", {})
    system = input_data.get("system", {})
    electrons = input_data.get("electrons", {})

    assert system["occupations"] == "smearing"
    assert system["smearing"] == "mv"
    assert system["degauss"] == 0.02
    assert electrons["mixing_beta"] == 0.4

@patch("workers.dft_worker.src.configurator.load_sssp_database")
@patch("workers.dft_worker.src.configurator.validate_pseudopotentials")
@patch("workers.dft_worker.src.configurator.get_pseudopotentials_dict")
@patch("workers.dft_worker.src.configurator.calculate_cutoffs")
def test_dft_configurator_ionic(mock_cutoffs, mock_get_pseudo, mock_validate, mock_load_sssp):
    """Test that ionic settings are applied correctly."""
    mock_cutoffs.return_value = (30, 200)
    mock_get_pseudo.return_value = {"Na": "Na.upf", "Cl": "Cl.upf"}
    mock_load_sssp.return_value = {}

    params = DFTParams()
    meta = MetaConfig(dft={"sssp_json_path": "sssp.json"}, lammps={})

    ionic_settings = {
        "occupations": "fixed"
    }

    configurator = DFTConfigurator(params, meta, type_dft_settings=ionic_settings)

    atoms = Atoms("NaCl", positions=[[0,0,0], [2.4,0,0]])
    elements = ["Na", "Cl"]

    calc, _ = configurator.build(atoms, elements)

    input_data = calc.parameters.get("input_data", {})
    system = input_data.get("system", {})
    assert system["occupations"] == "fixed"
