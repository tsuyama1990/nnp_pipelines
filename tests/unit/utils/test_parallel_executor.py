import unittest
from concurrent.futures import Future
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from orchestrator.src.utils.parallel_executor import ParallelExecutor

def square(x):
    return x * x

def slow_square(x):
    time.sleep(0.1)
    return x * x

def add(x, y):
    return x + y

class TestParallelExecutor(unittest.TestCase):
    def test_execute_simple(self):
        executor = ParallelExecutor(max_workers=2)
        items = [1, 2, 3, 4, 5]
        results = executor.execute(square, items)
        self.assertEqual(sorted(results), [1, 4, 9, 16, 25])

    def test_execute_with_args(self):
        executor = ParallelExecutor(max_workers=2)
        items = [1, 2, 3]
        # Adds 5 to each item
        results = executor.execute(add, items, 5)
        self.assertEqual(sorted(results), [6, 7, 8])

if __name__ == "__main__":
    unittest.main()
