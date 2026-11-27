import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    REASONING = "reasoning"          # Multi-step arithmetic & formal logic
    KNOWLEDGE = "knowledge"          # Domain knowledge, science, language
    FACTUALITY = "factuality"        # Hallucination probing & misinformation resistance


class BenchmarkSample(BaseModel):
    id: str
    task: TaskType
    prompt: str = Field(..., description="Prompt delivered to the language model")
    reference_answer: str = Field(..., description="Canonical correct ground truth answer")
    options: Optional[Dict[str, str]] = Field(default=None, description="Multiple choice options if applicable")
    untruthful_distractors: Optional[List[str]] = Field(default=None, description="Common hallucinations / falsehoods")
    rationale: Optional[str] = Field(default=None, description="Chain-of-thought explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkDataset:
    """
    Standardized dataset loader and manager for Persian evaluation tasks.
    """
    def __init__(self, samples: Optional[List[BenchmarkSample]] = None):
        self.samples = samples or []

    @classmethod
    def load_from_jsonl(cls, path: Path) -> "BenchmarkDataset":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(BenchmarkSample(**data))
        return cls(samples)

    def filter_by_task(self, task: TaskType) -> "BenchmarkDataset":
        filtered = [s for s in self.samples if s.task == task]
        return BenchmarkDataset(filtered)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
