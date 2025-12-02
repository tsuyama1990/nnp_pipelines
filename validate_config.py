#!/usr/bin/env python3
"""
Config Validation Script

Checks if the provided configuration file contains all critical keys and valid types
required for the pipeline to run. This script validates the structure against the
schema expected by the orchestrator and workers.
"""

import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def validate_nested_key(config: Dict[str, Any], keys: List[str], expected_type: Optional[Union[type, tuple]] = None) -> bool:
    """
    Traverses the config dictionary using a list of keys.
    Returns True if the key exists and matches expected_type (if provided).
    """
    current = config
    path_str = ""

    for i, key in enumerate(keys):
        path_str = f"{path_str}.{key}" if path_str else key

        if not isinstance(current, dict) or key not in current:
            logger.error(f"Missing key: '{path_str}'")
            return False
        current = current[key]

    if expected_type and not isinstance(current, expected_type):
        logger.error(f"Key '{path_str}' must be of type {expected_type}, got {type(current).__name__}")
        return False

    return True

def validate_config(config_path: str) -> bool:
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        return False

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse YAML: {e}")
        return False

    if not config:
        logger.error("Config file is empty.")
        return False

    valid = True

    # 1. Experiment Section
    if not validate_nested_key(config, ["experiment", "name"], str): valid = False
    if not validate_nested_key(config, ["experiment", "output_dir"], str): valid = False

    # 2. Generation Section
    # Validate that 'generation.scenarios' exists and is a list
    if validate_nested_key(config, ["generation", "scenarios"], list):
        scenarios = config["generation"]["scenarios"]
        if not scenarios:
             logger.warning("Generation scenarios list is empty.")
        for i, scenario in enumerate(scenarios):
             if not isinstance(scenario, dict):
                  logger.error(f"Scenario #{i} in generation.scenarios is not a dictionary.")
                  valid = False
             elif "type" not in scenario:
                  logger.error(f"Scenario #{i} missing 'type' key.")
                  valid = False
    else:
        valid = False

    # 3. Seed Generation
    if not validate_nested_key(config, ["seed_generation", "crystal_type"], str): valid = False
    if not validate_nested_key(config, ["seed_generation", "n_random_structures"], int): valid = False

    # 4. DFT Params
    # Validate that dft_params exists top-level
    if "dft_params" not in config:
        logger.error("Missing top-level key: 'dft_params'")
        valid = False
    else:
        if not validate_nested_key(config, ["dft_params", "kpoint_density"], (float, int)): valid = False

    # 5. ACE Model (Deeply Nested)
    # ace_model -> pacemaker_config -> potential -> elements
    # This is a specific requirement to ensure correct nesting
    if not validate_nested_key(config, ["ace_model", "pacemaker_config", "potential", "elements"], list):
        valid = False

    # Check pacemaker cutoff as well
    if not validate_nested_key(config, ["ace_model", "pacemaker_config", "cutoff"], (float, int)): valid = False

    # 6. MD Params
    if not validate_nested_key(config, ["md_params", "temperature"], (float, int)): valid = False
    if not validate_nested_key(config, ["md_params", "n_steps"], int): valid = False
    if not validate_nested_key(config, ["md_params", "elements"], list): valid = False

    if validate_nested_key(config, ["md_params", "timestep"], (float, int)):
         ts = config["md_params"]["timestep"]
         if not (0.1 <= ts <= 5.0):
              logger.error(f"md_params.timestep ({ts}) must be between 0.1 and 5.0 fs.")
              valid = False
    else:
        valid = False

    # 7. AL Params
    if not validate_nested_key(config, ["al_params", "n_clusters"], int): valid = False
    if not validate_nested_key(config, ["al_params", "r_core"], (float, int)): valid = False

    # box_size is optional (smart default), but if present, must be valid.
    # We also enforce the safety check here: box_size >= 2 * cutoff + 2.0
    box_size = config.get("al_params", {}).get("box_size")
    if box_size is not None:
        if not isinstance(box_size, (float, int)):
             logger.error("al_params.box_size must be a number.")
             valid = False
        else:
             cutoff = config.get("ace_model", {}).get("pacemaker_config", {}).get("cutoff", 5.0)
             min_box = 2 * cutoff + 2.0
             if box_size < min_box:
                 logger.error(f"al_params.box_size ({box_size}) is too small. Must be >= 2*cutoff + 2.0 ({min_box}).")
                 valid = False

    # initial_potential can be null (None) or string. If key exists, check type.
    if "al_params" in config:
        init_pot = config["al_params"].get("initial_potential")
        if init_pot is not None and not isinstance(init_pot, str):
             logger.error(f"Key 'al_params.initial_potential' must be string or null, got {type(init_pot).__name__}")
             valid = False

    if valid:
        logger.info(f"✅ Configuration '{config_path}' passed validation.")
    else:
        logger.error("❌ Configuration validation failed.")

    return valid

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_config.py <config.yaml>")
        sys.exit(1)

    if not validate_config(sys.argv[1]):
        sys.exit(1)
