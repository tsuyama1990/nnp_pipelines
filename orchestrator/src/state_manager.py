import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StateManager:
    """Manages the persistence of the orchestrator state."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_file = data_dir / "orchestrator_state.json"

    def save(self, state: Dict[str, Any]) -> None:
        """Save the current state to a JSON file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load(self) -> Dict[str, Any]:
        """Load the state from a JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {
            "iteration": 0,
            "current_potential": None,
            "current_asi": None,
            "current_structure": None,
            "is_restart": False,
            "al_consecutive_counter": 0
        }
