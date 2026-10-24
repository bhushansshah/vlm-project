"""What every benchmark plugs into the shared pipeline.

A ``Dataset`` knows how to download itself, load its questions as ``Sample`` objects,
turn a sample into model input (prompts), and, optionally, how to post-process answers,
write benchmark-specific output files, and grade answers with an LLM judge.
The runner, model backends, judge engine and reports are dataset-agnostic.
"""

from __future__ import annotations

import abc
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping

from ..types import Message


@dataclass(frozen=True)
class Sample:
    question_id: str
    image_path: str
    question: str
    category: str  # the benchmark's task type, used for per-category reports and sampling
    answer: str | None = None  # reference answer, when the benchmark publishes it
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)


PromptBuilder = Callable[[Sample], list[Message]]


@dataclass
class Selection:
    """Which questions to run. Steps are applied in this order."""

    categories: list[str] | None = None
    question_ids: list[str] | None = None
    filters: dict[str, list[str]] | None = None  # meta key -> allowed values, e.g. {"domain": ["av"]}
    per_category: int | None = None  # random N per category (balanced subset)
    limit: int | None = None
    seed: int | None = None  # randomness for per_category and shuffling before limit


def select_samples(samples: list[Sample], sel: Selection) -> list[Sample]:
    out = sorted(samples, key=_natural_key)  # stable order, so a seed always gives the same subset
    if sel.categories:
        out = [s for s in out if s.category in set(sel.categories)]
    if sel.question_ids:
        wanted = set(sel.question_ids)
        out = [s for s in out if s.question_id in wanted]
    for key, allowed in (sel.filters or {}).items():
        allowed_str = {str(v) for v in allowed}
        out = [s for s in out if str(s.meta.get(key)) in allowed_str]

    rng = random.Random(sel.seed if sel.seed is not None else 0)
    if sel.per_category is not None:
        by_cat: dict[str, list[Sample]] = defaultdict(list)
        for s in out:
            by_cat[s.category].append(s)
        out = []
        for cat in sorted(by_cat):
            group = by_cat[cat]
            out += rng.sample(group, min(sel.per_category, len(group)))
        out.sort(key=_natural_key)
    if sel.seed is not None and sel.limit is not None:
        rng.shuffle(out)
    if sel.limit is not None:
        out = out[:sel.limit]
    return out


def _natural_key(s: Sample):
    return (0, int(s.question_id), "") if s.question_id.isdigit() else (1, 0, s.question_id)


class Dataset(abc.ABC):
    name: ClassVar[str]
    categories: ClassVar[tuple[str, ...]]
    #: prompt name -> builder; "default" must exist
    prompts: ClassVar[dict[str, PromptBuilder]]
    #: sample.meta keys to break accuracy down by in reports (in addition to category)
    report_groups: ClassVar[tuple[str, ...]] = ()

    def __init__(self, data_dir: str | Path | None = None):
        root = Path(os.environ.get("VLMBENCH_DATA_DIR", "data"))
        self.data_dir = Path(data_dir) if data_dir else root / self.name

    # ---- data
    @abc.abstractmethod
    def prepare(self, force: bool = False) -> Path:
        """Download / extract the benchmark into ``self.data_dir`` (idempotent)."""

    @abc.abstractmethod
    def load_all(self) -> list[Sample]:
        """All samples; may assume ``prepare`` has run."""

    def is_prepared(self) -> bool:
        return (self.data_dir / "questions.jsonl").exists()

    def load(self, selection: Selection | None = None) -> list[Sample]:
        if not self.is_prepared():
            self.prepare()
        return select_samples(self.load_all(), selection or Selection())

    # ---- prompting
    def build_messages(self, prompt: str, sample: Sample) -> list[Message]:
        if prompt not in self.prompts:
            raise ValueError(f"Unknown prompt {prompt!r} for {self.name}. Available: {sorted(self.prompts)}")
        return self.prompts[prompt](sample)

    # ---- hooks with no-op defaults
    def postprocess(self, sample: Sample, response: str | None) -> dict[str, Any]:
        """Extra fields stored with each response record (e.g. an extracted answer)."""
        return {}

    def summarize(self, samples: dict[str, Sample], records: dict[str, dict]) -> dict[str, Any]:
        """Extra, benchmark-specific entries for summary.json."""
        return {}

    def write_outputs(self, run_dir: Path, samples: dict[str, Sample], records: dict[str, dict]) -> None:
        """Benchmark-specific files, e.g. an evaluation-server submission."""

    # ---- grading (only for benchmarks with public reference answers)
    def judge_prompt(self, sample: Sample, prediction: str) -> str:
        raise NotImplementedError(f"{self.name} has no public reference answers to judge against")
