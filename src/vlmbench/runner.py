"""Run a model over the benchmark: resumable, concurrent (API) or batched (local GPU).

Layout of a run directory::

    runs/<run_name>/
        config.json       # everything needed to reproduce the run
        responses.jsonl   # one record per attempt-completed question (append-only, latest wins)
        submission.json   # [{"question_id", "response"}] for the official evaluation server
        summary.json      # local diagnostics (errors, parse rates, tokens, latency)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .data import Sample
from .extract import extract, normalize_response
from .models import VLM, NonRetryableError
from .prompts import build_messages
from .types import Completion, GenerationConfig, TextPart

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    model: str  # "<provider>:<model>"
    model_options: dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: str = "default"
    run_name: str | None = None
    output_dir: str = "runs"
    concurrency: int = 8  # parallel requests for API / server backends
    batch_size: int = 1  # batch size for in-process local backends
    max_retries: int = 5
    retry_errors: bool = False  # re-run questions whose previous attempt errored
    # dataset selection
    data_dir: str | None = None
    categories: list[str] | None = None
    difficulties: list[str] | None = None
    question_ids: list[str] | None = None
    limit: int | None = None
    shuffle_seed: int | None = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        safe = self.model.replace("/", "_").replace(":", "__")
        return f"{safe}__{self.prompt}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generation"] = self.generation.to_dict()
        return d

    # Fields that change what a response *means*; resuming across a change would mix results.
    IDENTITY_FIELDS = ("model", "model_options", "generation", "prompt")


def read_records(path: Path) -> dict[str, dict]:
    """Latest record per question_id."""
    records: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["question_id"]] = r
    return records


class Runner:
    def __init__(self, model: VLM, config: RunConfig):
        self.model = model
        self.cfg = config
        self.run_dir = Path(config.output_dir) / config.resolved_run_name()
        self.responses_path = self.run_dir / "responses.jsonl"
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- setup
    def _check_or_write_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "config.json"
        current = {**self.cfg.to_dict(), "backend": self.model.describe()}
        if cfg_path.exists():
            previous = json.loads(cfg_path.read_text())
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
        todo = []
        for s in samples:
            r = done.get(s.question_id)
            if r is None or (r.get("error") and self.cfg.retry_errors):
                todo.append(s)
        n_err = sum(1 for s in samples if done.get(s.question_id, {}).get("error"))
        log.info("%d/%d questions done (%d with errors), %d to run",
                 len(samples) - len(todo), len(samples), n_err, len(todo))
        return todo

    # ------------------------------------------------------------- execution
    def _call_with_retries(self, fn, *args) -> tuple[Any, str | None, int]:
        delay = 2.0
        for attempt in range(1, self.cfg.max_retries + 2):
            try:
                return fn(*args), None, attempt
            except NonRetryableError as e:
                return None, f"non_retryable: {e}", attempt
            except Exception as e:  # noqa: BLE001 - transient network / rate limit / server errors
                if attempt > self.cfg.max_retries:
                    return None, f"{type(e).__name__}: {e}", attempt
                wait = min(delay, 120) * (0.5 + random.random())
                log.warning("Attempt %d failed (%s: %s); retrying in %.0fs",
                            attempt, type(e).__name__, str(e)[:200], wait)
                time.sleep(wait)
                delay *= 2
        raise AssertionError("unreachable")

    def _record(self, s: Sample, comp: Completion | None, error: str | None,
                attempts: int, latency: float) -> dict:
        prompt = "\n".join(
            p.text for m in build_messages(self.cfg.prompt, s) for p in m.content if isinstance(p, TextPart)
        )
        text = comp.text if comp else None
        ex = extract(text, s.question_type, s.options) if text is not None else None
        return {
            "question_id": s.question_id,
            "image_id": s.image_id,
            "category": s.category,
            "question_type": s.question_type,
            "difficulty": s.difficulty,
            "prompt": prompt,
            "response": text,
            "extracted": ex.value if ex else None,
            "parsed": ex.parsed if ex else False,
            "finish_reason": comp.finish_reason if comp else None,
            "usage": asdict(comp.usage) if comp else None,
            "latency_s": round(latency, 3),
            "attempts": attempts,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _write(self, rec: dict) -> None:
        with self._lock, self.responses_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _run_one(self, s: Sample) -> dict:
        t0 = time.perf_counter()
        comp, err, attempts = self._call_with_retries(
            self.model.generate, build_messages(self.cfg.prompt, s), self.cfg.generation
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
            convs = [build_messages(self.cfg.prompt, s) for s in batch]
            t0 = time.perf_counter()
            comps, err, attempts = self._call_with_retries(self.model.generate_batch, convs, self.cfg.generation)
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
        return finalize(self.run_dir, samples)


# ------------------------------------------------------------------ outputs
def finalize(run_dir: Path, samples: list[Sample] | None = None) -> dict:
    """Write submission.json and summary.json from responses.jsonl."""
    records = read_records(run_dir / "responses.jsonl")
    if samples is not None:
        wanted = {s.question_id for s in samples}
        records = {k: v for k, v in records.items() if k in wanted}
    write_submission(records, run_dir / "submission.json")
    summary = summarize(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_submission(records: dict[str, dict], path: Path) -> None:
    """Responses as returned by the model (like the authors' eval.py), with markdown emphasis
    stripped; the server applies its own extraction on top."""
    sub = [{"question_id": qid, "response": normalize_response(r["response"]) or ""}
           for qid, r in records.items()]
    path.write_text(json.dumps(sub, ensure_ascii=False))


def summarize(records: dict[str, dict]) -> dict:
    n = len(records)
    by_cat: dict[str, Counter] = defaultdict(Counter)
    letters: dict[str, Counter] = defaultdict(Counter)
    finish = Counter()
    errors = Counter()
    tok_in = tok_out = tok_reason = 0
    latencies = []
    for r in records.values():
        c = by_cat[r["category"]]
        c["n"] += 1
        c["error"] += bool(r["error"])
        c["parsed"] += bool(r["parsed"])
        finish[str(r["finish_reason"])] += 1
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1
        if r["question_type"] == "choice" and r["parsed"]:
            letters[r["category"]][r["extracted"]] += 1
        u = r.get("usage") or {}
        tok_in += u.get("input_tokens") or 0
        tok_out += u.get("output_tokens") or 0
        tok_reason += u.get("reasoning_tokens") or 0
        if not r["error"]:
            latencies.append(r["latency_s"])
    latencies.sort()
    return {
        "n_questions": n,
        "n_errors": sum(c["error"] for c in by_cat.values()),
        "error_types": dict(errors),
        "parse_rate": round(sum(c["parsed"] for c in by_cat.values()) / n, 4) if n else None,
        "per_category": {
            k: {"n": c["n"], "errors": c["error"], "parse_rate": round(c["parsed"] / c["n"], 4)}
            for k, c in sorted(by_cat.items())
        },
        "choice_letter_distribution": {k: dict(sorted(v.items())) for k, v in sorted(letters.items())},
        "finish_reasons": dict(finish),
        "tokens": {"input": tok_in, "output": tok_out, "reasoning": tok_reason},
        "latency_s": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
        },
        "note": "Ground truth is private: submit submission.json to "
                "https://huggingface.co/spaces/paulgavrikov/visualoverload-submit for accuracy.",
    }
