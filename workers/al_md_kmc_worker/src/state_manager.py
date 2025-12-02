import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class Stage(str, Enum):
    GENERATION = "GENERATION"
    EXPLORATION = "EXPLORATION"
    LABELING = "LABELING"
    TRAINING = "TRAINING"

class OrchestratorState(BaseModel):
    iteration: int = 0
    current_potential: Optional[str] = None
    current_asi: Optional[str] = None
    current_structure: Optional[str] = None
    is_restart: bool = False
    al_consecutive_counter: int = 0
    current_stage: Stage = Stage.EXPLORATION

    class Config:
        extra = "allow"
        use_enum_values = True

class StateManager:
    """Manages the persistence of the orchestrator state."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_file = data_dir / "orchestrator_state.json"

    def save(self, state: Dict[str, Any]) -> None:
        """Save the current state to a JSON file using atomic write."""
        # Validate with Pydantic (will raise ValidationError if invalid)
        model = OrchestratorState(**state)

        # Atomic write pattern
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, 'w') as f:
                f.write(model.model_dump_json(indent=4))
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise

    def load(self) -> Dict[str, Any]:
        """Load the state from a JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                # Validate
                model = OrchestratorState(**data)
                return model.model_dump()
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

        # Return default state if load fails or file doesn't exist
        return OrchestratorState().model_dump()
