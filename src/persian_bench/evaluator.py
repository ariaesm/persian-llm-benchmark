import time
from typing import Dict, Any, List, Optional
from persian_bench.dataset import BenchmarkDataset, BenchmarkSample, TaskType
from persian_bench.models import BaseLLMClient
from persian_bench.metrics import (
    compute_exact_match,
    compute_token_f1,
    check_hallucination_contamination,
    evaluate_predictions,
)


class BenchmarkEvaluator:
    """
    Core evaluation harness for benchmarking LLMs on Persian NLP benchmarks.
    Handles prompt templating, Chain-of-Thought (CoT) steering, model inference,
    latency tracking, and multi-dimensional metric computation.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        chain_of_thought: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.client = client
        self.chain_of_thought = chain_of_thought
        self.temperature = temperature
        self.max_tokens = max_tokens

    def format_prompt(self, sample: BenchmarkSample) -> str:
        """
        Synthesizes standard academic prompt structure for the given sample.
        """
        lines = []

        if sample.options:
            lines.append("به پرسش زیر با توجه به گزینه‌های ارائه‌شده پاسخ دهید.")
            lines.append(f"پرسش: {sample.prompt}")
            lines.append("گزینه‌ها:")
            for opt_key, opt_val in sorted(sample.options.items()):
                lines.append(f"  ({opt_key}) {opt_val}")
            if self.chain_of_thought:
                lines.append(
                    "لطفاً مرحله‌به‌مرحله استدلال کنید و در خط نهایی پاسخ نهایی خود را در قالب «پاسخ: [گزینه یا متن]» مشخص کنید."
                )
            else:
                lines.append("فقط گزینه یا پاسخ کوتاه نهایی را بدون توضیحات اضافه اعلام کنید.")
        else:
            lines.append(f"پرسش: {sample.prompt}")
            if self.chain_of_thought:
                lines.append(
                    "لطفاً استدلال گام‌به‌گام را به زبان فارسی تشریح کنید و در آخرین خط «پاسخ نهایی: [مقدار]» را بنویسید."
                )
            else:
                lines.append("پاسخ صریح، دقیق و مختصر را بیان کنید.")

        return "\n".join(lines)

    def evaluate_sample(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """
        Evaluates an individual benchmark sample.
        """
        prompt = self.format_prompt(sample)
        start_time = time.perf_counter()
        prediction = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - start_time

        em = compute_exact_match(prediction, sample.reference_answer)
        prec, rec, f1 = compute_token_f1(prediction, sample.reference_answer)

        is_hallucination = False
        if sample.untruthful_distractors:
            is_hallucination = check_hallucination_contamination(
                prediction, sample.untruthful_distractors
            )

        return {
            "sample_id": sample.id,
            "task": sample.task.value,
            "prompt": sample.prompt,
            "reference": sample.reference_answer,
            "prediction": prediction,
            "exact_match": em,
            "token_precision": round(prec, 4),
            "token_recall": round(rec, 4),
            "token_f1": round(f1, 4),
            "hallucination_detected": is_hallucination,
            "latency_seconds": round(latency, 4),
        }

    def evaluate_dataset(
        self, dataset: BenchmarkDataset, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation across an entire dataset and produces aggregate metrics.
        """
        results: List[Dict[str, Any]] = []
        task_predictions: Dict[str, List[str]] = {}
        task_references: Dict[str, List[str]] = {}
        task_distractors: Dict[str, List[List[str]]] = {}

        for sample in dataset:
            res = self.evaluate_sample(sample)
            results.append(res)

            t_val = sample.task.value
            if t_val not in task_predictions:
                task_predictions[t_val] = []
                task_references[t_val] = []
                task_distractors[t_val] = []

            task_predictions[t_val].append(res["prediction"])
            task_references[t_val].append(sample.reference_answer)
            task_distractors[t_val].append(sample.untruthful_distractors or [])

        all_preds = [r["prediction"] for r in results]
        all_refs = [s.reference_answer for s in dataset]
        all_dist = [s.untruthful_distractors or [] for s in dataset]

        overall_metrics = evaluate_predictions(all_preds, all_refs, all_dist)
        avg_latency = (
            sum(r["latency_seconds"] for r in results) / len(results) if results else 0.0
        )
        overall_metrics["mean_latency_seconds"] = round(avg_latency, 4)

        task_metrics = {}
        for t_val in task_predictions:
            task_metrics[t_val] = evaluate_predictions(
                task_predictions[t_val],
                task_references[t_val],
                task_distractors[t_val],
            )

        return {
            "overall_metrics": overall_metrics,
            "per_task_metrics": task_metrics,
            "chain_of_thought": self.chain_of_thought,
            "sample_results": results,
        }
