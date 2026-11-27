import pytest
from persian_bench.dataset import BenchmarkDataset, BenchmarkSample, TaskType
from persian_bench.models import MockLLMClient
from persian_bench.evaluator import BenchmarkEvaluator


def test_evaluator_prompt_formatting():
    client = MockLLMClient()
    evaluator = BenchmarkEvaluator(client=client, chain_of_thought=False)

    sample_mc = BenchmarkSample(
        id="k1",
        task=TaskType.KNOWLEDGE,
        prompt="پایتخت ایران کدام است؟",
        options={"الف": "شیراز", "ب": "تهران"},
        reference_answer="ب",
    )

    prompt = evaluator.format_prompt(sample_mc)
    assert "پایتخت ایران کدام است؟" in prompt
    assert "(الف) شیراز" in prompt
    assert "(ب) تهران" in prompt
    assert "فقط گزینه یا پاسخ کوتاه نهایی" in prompt


def test_evaluator_cot_prompt_formatting():
    client = MockLLMClient()
    evaluator = BenchmarkEvaluator(client=client, chain_of_thought=True)

    sample = BenchmarkSample(
        id="r1",
        task=TaskType.REASONING,
        prompt="۲ + ۲ چند می‌شود؟",
        reference_answer="۴",
    )

    prompt = evaluator.format_prompt(sample)
    assert "۲ + ۲ چند می‌شود؟" in prompt
    assert "استدلال گام‌به‌گام" in prompt


def test_evaluator_end_to_end_mock():
    client = MockLLMClient()
    evaluator = BenchmarkEvaluator(client=client)

    samples = [
        BenchmarkSample(
            id="s1",
            task=TaskType.KNOWLEDGE,
            prompt="پایتخت ایران چیست؟",
            reference_answer="تهران",
        ),
        BenchmarkSample(
            id="s2",
            task=TaskType.REASONING,
            prompt="حاصل ۲ + ۲ چیست؟",
            reference_answer="4",
        ),
    ]
    ds = BenchmarkDataset(samples)

    result = evaluator.evaluate_dataset(ds)
    assert "overall_metrics" in result
    assert "per_task_metrics" in result
    assert result["overall_metrics"]["samples_evaluated"] == 2
    # Both mock answers ("تهران" and "۴" vs "4") should match exactly
    assert result["overall_metrics"]["exact_match"] == 1.0
