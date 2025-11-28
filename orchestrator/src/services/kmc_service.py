import logging
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from shared.core.interfaces import KMCEngine, KMCResult
from shared.core.enums import KMCStatus
from shared.core.config import Config

logger = logging.getLogger(__name__)

class KMCService:
    """Service for managing Kinetic Monte Carlo steps."""

    def __init__(self, kmc_engine: KMCEngine, config: Config):
        self.kmc_engine = kmc_engine
        self.config = config

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        """Run a single KMC step."""
        logger.info("Starting KMC Step...")

        result = self.kmc_engine.run_step(initial_atoms, potential_path)

        if result.status == KMCStatus.SUCCESS:
            logger.info(f"KMC Event Successful. Time step: {result.time_step:.3e} s")
            # Apply thermalization to the new structure
            MaxwellBoltzmannDistribution(
                result.structure,
                temperature_K=self.config.md_params.temperature
            )
        elif result.status == KMCStatus.UNCERTAIN:
            logger.info("KMC detected uncertainty.")
        elif result.status == KMCStatus.NO_EVENT:
            logger.info("KMC found no event.")

        return result
