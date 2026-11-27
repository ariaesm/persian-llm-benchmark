"""
Persian LLM Benchmark — A Rigorous Academic Evaluation Suite for Foundation Models in Persian.
"""

from persian_bench.dataset import BenchmarkDataset, BenchmarkSample, TaskType
from persian_bench.metrics import evaluate_predictions, normalize_persian
from persian_bench.evaluator import BenchmarkEvaluator
from persian_bench.models import BaseLLMClient, MockLLMClient

__version__ = "1.0.0"
__all__ = [
    "BenchmarkDataset",
    "BenchmarkSample",
    "TaskType",
    "evaluate_predictions",
    "normalize_persian",
    "BenchmarkEvaluator",
    "BaseLLMClient",
    "MockLLMClient",
]
