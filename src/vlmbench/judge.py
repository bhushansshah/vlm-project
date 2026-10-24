"""LLM-as-a-judge grading for benchmarks with public reference answers.

The dataset supplies the judging prompt (``Dataset.judge_prompt``); the judge model is any
backend from ``vlmbench.models``, e.g. ``openrouter:qwen/qwen-2.5-72b-instruct`` (the
default judge in VistaQA's official evaluator). Judgments are appended to
``<run_dir>/judgments.jsonl`` (or another file name, to keep several judges side by side) and re-used on later calls: only new or changed answers,
or ones whose judging failed, are sent to the judge again.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .datasets import Sample, get_dataset
from .models import VLM, NonRetryableError
from .retry import call_with_retries
from .runner import load_run_config, read_records
from .types import GenerationConfig, Message

log = logging.getLogger(__name__)

DEFAULT_JUDGE = "openrouter:qwen/qwen-2.5-72b-instruct"


def _json_objects(text: str) -> list[Any]:
    """Every JSON object embedded in ``text``, in order."""
    decoder, objs, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
            objs.append(obj)
            i = end
        except ValueError:
            i += 1
    return objs


def parse_judgment(raw: str | None) -> tuple[int | None, str]:
    """Parse ``{"correct": 0|1, "reason": ...}`` like the official VistaQA evaluator, with one
    extension: when the judge writes several verdicts (it changed its mind), the last one counts
    (the official parser fails on these and scores them 0). Returns ``(None, why)`` if unparseable."""
    raw = (raw or "").strip()
    verdicts = [o for o in _json_objects(raw) if isinstance(o, dict) and "correct" in o]
    data: dict[str, Any] = verdicts[-1] if verdicts else {}
    if "correct" not in data:
        return None, f"unparseable judge output: {raw[:200]}"
    c = data["correct"]
    correct = int(str(c).strip().lower() in {"1", "true", "yes"}) if isinstance(c, str) else int(int(c) == 1)
    return correct, str(data.get("reason", ""))


class Judge:
    def __init__(self, model: VLM, spec: str, json_mode: bool = True, max_retries: int = 5,
                 extra: dict[str, Any] | None = None):
        self.model = model
        self.spec = spec
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.extra = extra or {}  # provider-specific options, e.g. {"reasoning": {"enabled": False}}

    def _ask(self, prompt: str) -> str | None:
        json_fmt = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        cfg = GenerationConfig(temperature=0.0, max_tokens=512, extra={**self.extra, **json_fmt})
        try:
            return self.model.generate([Message.user(prompt)], cfg).text
        except NonRetryableError:
            if not self.json_mode:
                raise
            # Like the official evaluator: fall back to a plain request if JSON mode is rejected.
            log.info("JSON mode rejected by %s; retrying without it", self.spec)
            self.json_mode = False
            cfg = GenerationConfig(temperature=0.0, max_tokens=512, extra=self.extra)
            return self.model.generate([Message.user(prompt)], cfg).text

    def grade(self, prompt: str) -> dict[str, Any]:
        raw, err, _ = call_with_retries(self._ask, prompt, max_retries=self.max_retries)
        if err:
            return {"correct": None, "reason": "", "judge_raw": None, "error": err}
        correct, reason = parse_judgment(raw)
        return {"correct": correct, "reason": reason, "judge_raw": raw,
                "error": None if correct is not None else reason}


def judge_run(run_dir: Path, judge: Judge, concurrency: int = 8,
              filename: str = "judgments.jsonl") -> dict[str, int]:
    """Grade every answer in a run. Returns counts of what was done."""
    cfg = load_run_config(run_dir)
    dataset = get_dataset(cfg["dataset"], cfg.get("data_dir"))
    samples: dict[str, Sample] = {s.question_id: s for s in dataset.load()}
    responses = read_records(run_dir / "responses.jsonl")
    path = run_dir / filename
    previous = read_records(path)
    lock = threading.Lock()

    def up_to_date(qid: str, resp: dict) -> bool:
        j = previous.get(qid)
        return (j is not None and j["error"] is None and j["judge_model"] == judge.spec
                and j["judged_response"] == resp["response"])

    def write(rec: dict) -> None:
        with lock, path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def work(qid: str) -> None:
        resp = responses[qid]
        base = {"question_id": qid, "judge_model": judge.spec, "judged_response": resp["response"],
                "timestamp": datetime.now(timezone.utc).isoformat()}
        if resp["error"] or not (resp["response"] or "").strip():
            # The model gave no answer: wrong, no judge call needed.
            write({**base, "correct": 0, "reason": "no answer from model", "judge_raw": None, "error": None})
            return
        write({**base, **judge.grade(dataset.judge_prompt(samples[qid], resp["response"]))})

    todo = [q for q, r in responses.items() if q in samples and not up_to_date(q, r)]
    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(tqdm(pool.map(work, todo), total=len(todo), desc=f"judging {run_dir.name}"))
    final = read_records(path)
    return {
        "answers": len(responses),
        "judged_now": len(todo),
        "judge_errors": sum(1 for q in responses if final.get(q, {}).get("error")),
    }
