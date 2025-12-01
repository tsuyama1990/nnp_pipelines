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
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "experiment",
    "md_params",
    "dft_params",
    "ace_model",
    "al_params",
    "training_params",
    "exploration"
]

REQUIRED_KEYS = {
    "experiment": ["name"],
    "md_params": ["elements", "temperature", "n_steps"],
    "dft_params": ["kpoint_density"],
    "ace_model": ["elements", "cutoff"],
    "al_params": ["n_clusters", "r_core", "box_size", "initial_potential"],
    "training_params": ["loss_energy", "loss_force"],
}

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

    # Check Sections
    for section in REQUIRED_SECTIONS:
        if section not in config:
            logger.error(f"Missing required section: '{section}'")
            valid = False
        elif not isinstance(config[section], dict):
             logger.error(f"Section '{section}' must be a dictionary.")
             valid = False
        else:
            # Check Keys within Section
            if section in REQUIRED_KEYS:
                for key in REQUIRED_KEYS[section]:
                    if key not in config[section]:
                        logger.error(f"Missing key '{key}' in section '{section}'")
                        valid = False

    if valid:
        logger.info(f"✅ Configuration '{config_path}' is valid.")
    else:
        logger.error("❌ Configuration validation failed.")

    return valid

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_config.py <config.yaml>")
        sys.exit(1)

    if not validate_config(sys.argv[1]):
        sys.exit(1)
