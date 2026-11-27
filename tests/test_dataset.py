from pathlib import Path
import pytest
from persian_bench.dataset import BenchmarkDataset, BenchmarkSample, TaskType


def test_benchmark_sample_creation():
    sample = BenchmarkSample(
        id="test_01",
        task=TaskType.REASONING,
        prompt="۲ + ۲ برابر با چند است؟",
        reference_answer="۴",
        rationale="جمع ساده دو عدد",
    )
    assert sample.id == "test_01"
    assert sample.task == TaskType.REASONING
    assert sample.reference_answer == "۴"
    assert sample.options is None


def test_dataset_loading_and_filtering(tmp_path: Path):
    jsonl_content = (
        '{"id": "r1", "task": "reasoning", "prompt": "Q1", "reference_answer": "A1"}\n'
        '{"id": "k1", "task": "knowledge", "prompt": "Q2", "reference_answer": "A2"}\n'
        '{"id": "f1", "task": "factuality", "prompt": "Q3", "reference_answer": "A3"}\n'
    )
    test_file = tmp_path / "test_data.jsonl"
    test_file.write_text(jsonl_content, encoding="utf-8")

    ds = BenchmarkDataset.load_from_jsonl(test_file)
    assert len(ds) == 3

    reasoning_ds = ds.filter_by_task(TaskType.REASONING)
    assert len(reasoning_ds) == 1
    assert reasoning_ds.samples[0].id == "r1"

    knowledge_ds = ds.filter_by_task(TaskType.KNOWLEDGE)
    assert len(knowledge_ds) == 1
    assert knowledge_ds.samples[0].id == "k1"


def test_real_dataset_files_validity():
    data_dir = Path("data")
    assert data_dir.exists(), "Data directory must exist"

    for task in [TaskType.REASONING, TaskType.KNOWLEDGE, TaskType.FACTUALITY]:
        file_path = data_dir / f"{task.value}.jsonl"
        assert file_path.exists(), f"Benchmark dataset missing: {file_path}"
        ds = BenchmarkDataset.load_from_jsonl(file_path)
        assert len(ds) > 0, f"Benchmark dataset {task.value} has 0 samples"
        for sample in ds:
            assert sample.id
            assert sample.prompt
            assert sample.reference_answer
