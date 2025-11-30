import logging
from src.interfaces.explorer import BaseExplorer, ExplorationResult, ExplorationStatus

logger = logging.getLogger(__name__)

class HybridExplorer(BaseExplorer):
    def __init__(self, md_explorer: BaseExplorer, kmc_explorer: BaseExplorer):
        self.md_explorer = md_explorer
        self.kmc_explorer = kmc_explorer

    def explore(self,
                current_structure: str,
                potential_path: str,
                iteration: int,
                is_restart: bool = False,
                **kwargs) -> ExplorationResult:

        # Step 1: Run MD
        logger.info("HybridExplorer: Starting MD phase...")
        md_result = self.md_explorer.explore(current_structure, potential_path, iteration, is_restart, **kwargs)

        if md_result.status != ExplorationStatus.SUCCESS:
            return md_result

        # If MD was successful, we proceed to KMC.
        # However, MD might have finished just fine but we need to select a structure for KMC.
        # LammpsMDExplorer returns `final_structure` which is likely a restart file or similar.

        # Check if we should run KMC (e.g. config check is likely done inside KMC explorer or factory,
        # but here we assume if Hybrid is used, we want both).

        logger.info("HybridExplorer: MD phase complete. Starting KMC phase...")

        # We use the result of MD as input for KMC
        kmc_input = md_result.final_structure
        if not kmc_input:
            logger.error("HybridExplorer: MD succeeded but returned no structure for KMC.")
            return md_result # Or failed?

        kmc_result = self.kmc_explorer.explore(kmc_input, potential_path, iteration, **kwargs)

        # If KMC returns NO_EVENT, we might just want to return the MD result (or the KMC result which points to MD end)
        if kmc_result.status == ExplorationStatus.NO_EVENT:
             # If no event, we basically just did MD.
             # KMCExplorer returns final_structure = input (which is MD output)
             pass

        return kmc_result
