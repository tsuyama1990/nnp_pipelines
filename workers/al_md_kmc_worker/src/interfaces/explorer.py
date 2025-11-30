from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List
from pathlib import Path
from ase import Atoms

class ExplorationStatus(Enum):
    SUCCESS = "success"
    UNCERTAIN = "uncertain"
    FAILED = "failed"
    NO_EVENT = "no_event"

@dataclass
class ExplorationResult:
    status: ExplorationStatus
    final_structure: Optional[str] = None
    trajectory_path: Optional[str] = None
    uncertain_structures: List[Atoms] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class BaseExplorer(ABC):
    @abstractmethod
    def explore(self,
                current_structure: str,
                potential_path: str,
                iteration: int,
                **kwargs) -> ExplorationResult:
        """
        Explores the configuration space starting from current_structure.
        """
        pass
