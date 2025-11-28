import pytest
from unittest.mock import MagicMock
from workers.gen_worker.src.scenarios import ScenarioFactory, CrystalAwareScenario, RandomScenario
from shared.autostructure.alloy import AlloyGenerator

def test_scenario_factory_metallic():
    """Verify ScenarioFactory returns AlloyGenerator for metallic type."""
    config = {
        "crystal_type": "metallic",
        "elements": ["Al", "Cu"]
    }

    scenario = ScenarioFactory.create(config)

    assert isinstance(scenario, CrystalAwareScenario)
    assert scenario.generator_cls == AlloyGenerator

def test_scenario_factory_random():
    """Verify ScenarioFactory returns RandomScenario for random type."""
    config = {
        "crystal_type": "random",
        "elements": ["Al", "Cu"]
    }

    scenario = ScenarioFactory.create(config)

    assert isinstance(scenario, RandomScenario)
