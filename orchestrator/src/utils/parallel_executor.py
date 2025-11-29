import logging
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import List, Callable, TypeVar, Any, Dict, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ParallelExecutor:
    """
    A utility class to execute tasks in parallel using ProcessPoolExecutor.
    """

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers

    def execute(self, task: Callable[..., T], items: List[Any], *args, **kwargs) -> List[T]:
        """
        Executes a task in parallel for a list of items.

        The task function should accept an item as its first argument.
        Additional args and kwargs are passed to the task.
        """
        results: List[T] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item: Dict[Future, Any] = {
                executor.submit(task, item, *args, **kwargs): item
                for item in items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Task failed for item {item}: {e}")

        return results

    def submit_tasks(self, task_creators: List[Callable[[], Any]]) -> List[Any]:
        """
        Executes a list of task creating functions (lambdas or partials) in parallel.
        Useful when tasks have different arguments.
        """
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(tc) for tc in task_creators]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                     logger.error(f"Task execution failed: {e}")
        return results
