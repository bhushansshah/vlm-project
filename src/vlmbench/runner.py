"""Run a model over a benchmark: resumable, concurrent (API) or batched (local GPU).

Layout of a run directory::

    runs/<run_name>/
        config.json       # everything needed to reproduce the run
        responses.jsonl   # one record per answered question (append-only, latest wins)
        summary.json      # errors, tokens, latency + dataset-specific diagnostics
        ...               # dataset-specific outputs (e.g. submission.json, predictions.json)
        judgments.jsonl   # written later by `vlmbench judge` (datasets with public answers)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .agents import summarize_traces
from .datasets import Dataset, Sample, Selection
from .models import VLM
from .retry import call_with_retries
from .types import Completion, GenerationConfig, TextPart, usage_totals

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    model: str  # "<provider>:<model>"
    dataset: str = "visualoverload"
    model_options: dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: str = "default"
    agent: str | None = None  # e.g. "tools": wrap the model in a tool-using agent
    agent_options: dict[str, Any] = field(default_factory=dict)
    run_name: str | None = None
    tag: str | None = None  # appended to the automatic run name, e.g. "reason-medium" for a settings variant
    output_dir: str = "runs"
    concurrency: int = 8  # parallel requests for API / server backends
    batch_size: int = 1  # batch size for in-process local backends
    max_retries: int = 5
    retry_errors: bool = False  # re-run questions whose previous attempt errored
    # question selection
    data_dir: str | None = None
    categories: list[str] | None = None
    question_ids: list[str] | None = None
    filters: dict[str, list[str]] | None = None
    per_category: int | None = None
    limit: int | None = None
    sample_seed: int | None = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        safe = self.model.replace("/", "_").replace(":", "__")
        name = f"{self.dataset}__{safe}__{self.prompt}"
        if self.agent:
            name += f"__agent-{self.agent}"
            if self.agent_options.get("tools"):
                name += "-" + "+".join(self.agent_options["tools"])
            if self.agent_options.get("instructions"):
                name += "-" + str(self.agent_options["instructions"])
            if self.agent_options.get("protocol", "text") != "text":
                name += "-" + str(self.agent_options["protocol"])
            if self.agent_options.get("verify"):
                name += "-verify"
        if self.tag:
            name += f"__{self.tag}"
        return name

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.resolved_run_name()

    def selection(self) -> Selection:
        return Selection(categories=self.categories, question_ids=self.question_ids, filters=self.filters,
                         per_category=self.per_category, limit=self.limit, seed=self.sample_seed)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generation"] = self.generation.to_dict()
        return d

    # Fields that change what a response *means*; resuming across a change would mix results.
    IDENTITY_FIELDS = ("dataset", "model", "model_options", "generation", "prompt", "agent", "agent_options")


def read_records(path: Path) -> dict[str, dict]:
    """Latest record per question_id from an append-only JSONL file."""
    records: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["question_id"]] = r
    return records


def load_run_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text())


class Runner:
    def __init__(self, model: VLM, dataset: Dataset, config: RunConfig):
        self.model = model
        self.dataset = dataset
        self.cfg = config
        self.run_dir = config.run_dir()
        self.responses_path = self.run_dir / "responses.jsonl"
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- setup
    def _check_or_write_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "config.json"
        current = {**self.cfg.to_dict(), "backend": self.model.describe()}
        if cfg_path.exists():
            previous = json.loads(cfg_path.read_text())
            # fill in fields added after a run was created, so old runs still resume
            previous.setdefault("dataset", "visualoverload")
            previous.setdefault("agent", None)
            previous.setdefault("agent_options", {})
            previous["generation"] = {**GenerationConfig().to_dict(), **previous.get("generation", {})}
            changed = [k for k in RunConfig.IDENTITY_FIELDS if previous.get(k) != current.get(k)]
            if changed:
                raise RuntimeError(
                    f"{self.run_dir} was created with different {changed}. "
                    "Use a different --run-name, or delete the directory to start over."
                )
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        cfg_path.write_text(json.dumps(current, indent=2, default=str))

    def _pending(self, samples: list[Sample]) -> list[Sample]:
        done = read_records(self.responses_path)
        todo = [s for s in samples
                if s.question_id not in done or (done[s.question_id].get("error") and self.cfg.retry_errors)]
        n_err = sum(1 for s in samples if done.get(s.question_id, {}).get("error"))
        log.info("%d/%d questions done (%d with errors), %d to run",
                 len(samples) - len(todo), len(samples), n_err, len(todo))
        return todo

    # ------------------------------------------------------------- execution
    def _messages(self, s: Sample):
        return self.dataset.build_messages(self.cfg.prompt, s)

    def _record(self, s: Sample, comp: Completion | None, error: str | None,
                attempts: int, latency: float) -> dict:
        prompt = "\n".join(p.text for m in self._messages(s) for p in m.content if isinstance(p, TextPart))
        text = comp.text if comp else None
        return {
            "question_id": s.question_id,
            "category": s.category,
            "prompt": prompt,
            "response": text,
            **self.dataset.postprocess(s, text),
            "finish_reason": comp.finish_reason if comp else None,
            "usage": asdict(comp.usage) if comp else None,
            "latency_s": round(latency, 3),
            "attempts": attempts,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **({"trace": comp.trace} if comp is not None and comp.trace is not None else {}),
        }

    def _write(self, rec: dict) -> None:
        with self._lock, self.responses_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _run_one(self, s: Sample) -> dict:
        t0 = time.perf_counter()
        comp, err, attempts = call_with_retries(
            self.model.generate, self._messages(s), self.cfg.generation, max_retries=self.cfg.max_retries
        )
        rec = self._record(s, comp, err, attempts, time.perf_counter() - t0)
        self._write(rec)
        return rec

    def _run_concurrent(self, todo: list[Sample], pbar: tqdm) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            futures = [pool.submit(self._run_one, s) for s in todo]
            try:
                for fut in as_completed(futures):
                    rec = fut.result()
                    pbar.update(1)
                    if rec["error"]:
                        pbar.set_postfix_str(f"last error: {rec['error'][:60]}")
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise

    def _batches(self, todo: list[Sample]) -> Iterator[list[Sample]]:
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(todo), bs):
            yield todo[i:i + bs]

    def _run_batched(self, todo: list[Sample], pbar: tqdm) -> None:
        for batch in self._batches(todo):
            convs = [self._messages(s) for s in batch]
            t0 = time.perf_counter()
            comps, err, attempts = call_with_retries(
                self.model.generate_batch, convs, self.cfg.generation, max_retries=self.cfg.max_retries
            )
            per_item = (time.perf_counter() - t0) / len(batch)
            if err and len(batch) > 1:  # e.g. OOM on a big batch: fall back to one at a time
                log.warning("Batch failed (%s); retrying items individually", err)
                for s in batch:
                    self._run_one(s)
                    pbar.update(1)
                continue
            for i, s in enumerate(batch):
                self._write(self._record(s, comps[i] if comps else None, err, attempts, per_item))
                pbar.update(1)

    def run(self, samples: list[Sample]) -> dict:
        self._check_or_write_config()
        todo = self._pending(samples)
        if todo:
            with tqdm(total=len(todo), desc=self.cfg.resolved_run_name()) as pbar:
                if self.model.supports_concurrency and self.cfg.concurrency > 1:
                    self._run_concurrent(todo, pbar)
                else:
                    self._run_batched(todo, pbar)
        return finalize(self.run_dir, self.dataset, samples)


# ------------------------------------------------------------------ outputs
def finalize(run_dir: Path, dataset: Dataset, samples: list[Sample] | None = None) -> dict:
    """Write summary.json and the dataset's output files from responses.jsonl."""
    records = read_records(run_dir / "responses.jsonl")
    if samples is None:
        samples = dataset.load()
    by_id = {s.question_id: s for s in samples}
    records = {k: v for k, v in records.items() if k in by_id}
    dataset.write_outputs(run_dir, by_id, records)
    summary = {**summarize(records), **summarize_traces(records), **dataset.summarize(by_id, records)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def cost_summary(totals: dict[str, Any]) -> dict[str, Any]:
    """Cost part of a summary from usage_totals(). usd is None when the provider reports no cost.
    calls = records with usage: one per answer (an agent answer already sums its steps)."""
    return {"usd": totals["cost_usd"], "calls_with_cost": totals["cost_reported_calls"], "calls": totals["calls"]}


def summarize(records: dict[str, dict]) -> dict:
    by_cat: dict[str, Counter] = defaultdict(Counter)
    finish = Counter()
    errors = Counter()
    latencies = []
    for r in records.values():
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["errors"] += bool(r["error"])
        finish[str(r["finish_reason"])] += 1
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1
        else:
            latencies.append(r["latency_s"])
    u = usage_totals(r.get("usage") for r in records.values())
    latencies.sort()
    return {
        "n_questions": len(records),
        "n_errors": sum(c["errors"] for c in by_cat.values()),
        "error_types": dict(errors),
        "per_category": {k: dict(c) for k, c in sorted(by_cat.items())},
        "finish_reasons": dict(finish),
        "tokens": {k: u[k] for k in ("input", "output", "reasoning", "cached")},
        "cost": cost_summary(u),
        "latency_s": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
        },
    }
