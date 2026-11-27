import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from persian_bench.dataset import BenchmarkDataset, BenchmarkSample, TaskType
from persian_bench.evaluator import BenchmarkEvaluator
from persian_bench.models import BaseLLMClient, MockLLMClient, OpenAILLMClient, OllamaLLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persian-bench",
        description="Persian LLM Benchmark — Academic Evaluation Suite for Foundation Models in Persian",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to directory containing benchmark JSONL datasets (default: data)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "reasoning", "knowledge", "factuality"],
        help="Evaluation task axis to benchmark (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        choices=["mock", "openai", "ollama"],
        help="Target model client adapter (default: mock)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model identifier (e.g. gpt-4o-mini, llama3.1:8b)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key for hosted providers (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Enable Chain-of-Thought (CoT) reasoning prompt steering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export full JSON benchmark results",
    )
    return parser


def load_datasets(data_dir: Path, selected_task: str) -> BenchmarkDataset:
    tasks_to_load = (
        [TaskType.REASONING, TaskType.KNOWLEDGE, TaskType.FACTUALITY]
        if selected_task == "all"
        else [TaskType(selected_task)]
    )

    all_samples: List[BenchmarkSample] = []
    for t in tasks_to_load:
        file_path = data_dir / f"{t.value}.jsonl"
        if file_path.exists():
            ds = BenchmarkDataset.load_from_jsonl(file_path)
            all_samples.extend(ds.samples)
        else:
            print(f"[Warning] Dataset file not found: {file_path}", file=sys.stderr)

    return BenchmarkDataset(all_samples)


def get_model_client(args) -> BaseLLMClient:
    if args.model == "mock":
        return MockLLMClient()
    elif args.model == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY env var.")
        model_name = args.model_name or "gpt-4o-mini"
        return OpenAILLMClient(api_key=api_key, model=model_name)
    elif args.model == "ollama":
        model_name = args.model_name or "llama3.1"
        return OllamaLLMClient(model=model_name)
    else:
        raise ValueError(f"Unknown model provider: {args.model}")


def print_results_table(summary: dict):
    overall = summary["overall_metrics"]
    per_task = summary.get("per_task_metrics", {})

    header = f"{'Task':<16} | {'Samples':<8} | {'Exact Match':<12} | {'Token F1':<10} | {'Truthfulness':<14}"
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  PERSIAN LLM BENCHMARK — EMPIRICAL EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print(divider)

    for task_name, metrics in per_task.items():
        print(
            f"{task_name.capitalize():<16} | "
            f"{metrics['samples_evaluated']:<8} | "
            f"{metrics['exact_match'] * 100:>10.1f}% | "
            f"{metrics['macro_f1'] * 100:>8.1f}% | "
            f"{metrics['truthfulness_rate'] * 100:>12.1f}%"
        )

    print(divider)
    print(
        f"{'OVERALL':<16} | "
        f"{overall['samples_evaluated']:<8} | "
        f"{overall['exact_match'] * 100:>10.1f}% | "
        f"{overall['macro_f1'] * 100:>8.1f}% | "
        f"{overall['truthfulness_rate'] * 100:>12.1f}%"
    )
    print("=" * len(header))
    print(f"Mean Generation Latency: {overall.get('mean_latency_seconds', 0.0):.4f}s / query\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    dataset = load_datasets(data_dir, args.task)
    if len(dataset) == 0:
        print(f"Error: No benchmark samples found in '{data_dir}' for task '{args.task}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples across task(s): {args.task}")
    print(f"Initializing adapter for model '{args.model}' (CoT={args.cot})...")

    client = get_model_client(args)
    evaluator = BenchmarkEvaluator(client=client, chain_of_thought=args.cot)

    results = evaluator.evaluate_dataset(dataset)
    print_results_table(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Full benchmark results exported to: {out_path}")


if __name__ == "__main__":
    main()
