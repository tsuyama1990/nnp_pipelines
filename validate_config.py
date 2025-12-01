#!/usr/bin/env python3
"""
Config Validation Script

Checks if the provided configuration file contains all critical keys and valid types
required for the pipeline to run.
"""

import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def validate_nested_key(config: Dict[str, Any], keys: List[str], expected_type: Optional[type] = None) -> bool:
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
        logger.error(f"Key '{path_str}' must be of type {expected_type.__name__}, got {type(current).__name__}")
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
    valid &= validate_nested_key(config, ["experiment", "name"], str)
    valid &= validate_nested_key(config, ["experiment", "output_dir"], str)

    # 2. Generation Section
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

    # 3. Seed Generation
    valid &= validate_nested_key(config, ["seed_generation", "crystal_type"], str)
    valid &= validate_nested_key(config, ["seed_generation", "n_random_structures"], int)

    # 4. DFT Params
    valid &= validate_nested_key(config, ["dft_params", "kpoint_density"], (float, int))

    # 5. ACE Model (Deeply Nested)
    # ace_model -> pacemaker_config -> potential -> elements
    valid &= validate_nested_key(config, ["ace_model", "pacemaker_config", "potential", "elements"], list)
    valid &= validate_nested_key(config, ["ace_model", "pacemaker_config", "cutoff"], (float, int))

    # 6. MD Params
    valid &= validate_nested_key(config, ["md_params", "temperature"], (float, int))
    valid &= validate_nested_key(config, ["md_params", "n_steps"], int)
    valid &= validate_nested_key(config, ["md_params", "elements"], list)

    # 7. AL Params
    valid &= validate_nested_key(config, ["al_params", "n_clusters"], int)
    valid &= validate_nested_key(config, ["al_params", "r_core"], (float, int))
    valid &= validate_nested_key(config, ["al_params", "box_size"], (float, int))
    # initial_potential can be null or string, checking existence
    if "al_params" in config and "initial_potential" not in config["al_params"]:
         logger.error("Missing key: 'al_params.initial_potential'")
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
