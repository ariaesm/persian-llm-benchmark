# Persian LLM Benchmark (`persian-bench`)

An open-source evaluation suite for evaluating language models on Persian reasoning, domain knowledge, and factual consistency.

Developed by **Aria Esmaeilzadeh** (`@ariaesm`).

---

## Overview

`persian-bench` provides an automated harness for testing how well generative language models handle Persian text across three evaluation axes:

1. **Reasoning (`data/reasoning.jsonl`)**: Multi-step arithmetic, logic, and word problems in Persian.
2. **Knowledge (`data/knowledge.jsonl`)**: Cultural, scientific, and historical multiple-choice queries.
3. **Factuality (`data/factuality.jsonl`)**: Detection of common misconceptions and verification against factual references.

---

## Features

- **Persian Text Normalizer**: Unifies Arabic/Persian letter variants (ک/ی), Persian digits (۱۲۳ -> 123), and handles zero-width spaces.
- **Pluggable Architecture**: Built-in mock evaluator for offline testing, plus adapters for OpenAI and Ollama.
- **Metric Suite**: Exact match (EM), token-level precision/recall/F1, and distractor contamination rate.
- **CLI Utility**: Command-line interface with formatted summary tables.

---

## Quickstart

```bash
# Clone and enter the repository
git clone https://github.com/ariaesm/persian-llm-benchmark.git
cd persian-llm-benchmark

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[test,api]"

# Run offline smoke evaluation
persian-bench --task all --model mock
```

---

## Programmatic Usage

```python
from pathlib import Path
from persian_bench.dataset import BenchmarkDataset
from persian_bench.evaluator import BenchmarkEvaluator
from persian_bench.models import MockLLMClient

# Load test dataset
dataset = BenchmarkDataset.load_from_jsonl(Path("data/reasoning.jsonl"))

# Run evaluator
evaluator = BenchmarkEvaluator(client=MockLLMClient())
results = evaluator.evaluate_dataset(dataset)

print("Exact match score:", results["overall_metrics"]["exact_match"])
```

---

## Citation

If you find this benchmark useful in your research or project, please cite:

```bibtex
@software{esmaeilzadeh2026persianbench,
  author = {Aria Esmaeilzadeh},
  title = {Persian LLM Benchmark Suite},
  year = {2026},
  url = {https://github.com/ariaesm/persian-llm-benchmark}
}
```

---

## License

This project is licensed under the MIT License.
