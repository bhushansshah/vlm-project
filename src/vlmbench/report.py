"""Accuracy reports from judged runs: overall and broken down by category and dataset groups."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets import get_dataset
from .runner import load_run_config, read_records


@dataclass
class Tally:
    correct: int = 0
    graded: int = 0

    def add(self, ok: bool) -> None:
        self.correct += int(ok)
        self.graded += 1

    @property
    def acc(self) -> float | None:
        return 100 * self.correct / self.graded if self.graded else None


@dataclass
class RunScores:
    name: str
    model: str
    prompt: str
    n_answers: int
    n_ungraded: int  # answers whose judgment is missing or failed
    overall: Tally = field(default_factory=Tally)
    groups: dict[str, dict[str, Tally]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(Tally)))
    question_ids: frozenset = frozenset()


def score_run(run_dir: Path, judgments_file: str = "judgments.jsonl") -> tuple[RunScores, tuple[str, ...]]:
    cfg = load_run_config(run_dir)
    dataset = get_dataset(cfg["dataset"], cfg.get("data_dir"))
    samples = {s.question_id: s for s in dataset.load()}
    responses = {q: r for q, r in read_records(run_dir / "responses.jsonl").items() if q in samples}
    judgments = read_records(run_dir / judgments_file)
    if not judgments:
        raise SystemExit(f"{run_dir} has no judgments yet; run `vlmbench judge {run_dir}` first")

    rs = RunScores(name=run_dir.name, model=cfg["model"], prompt=cfg["prompt"],
                   n_answers=len(responses), n_ungraded=0, question_ids=frozenset(responses))
    for qid, resp in responses.items():
        j = judgments.get(qid)
        if j is None or j["correct"] is None or j["judged_response"] != resp["response"]:
            rs.n_ungraded += 1
            continue
        ok = j["correct"] == 1
        s = samples[qid]
        rs.overall.add(ok)
        rs.groups["category"][s.category].add(ok)
        for key in dataset.report_groups:
            rs.groups[key][str(s.meta.get(key))].add(ok)
    return rs, ("category", *dataset.report_groups)


def _fmt(t: Tally | None) -> str:
    return "–" if t is None or t.acc is None else f"{t.acc:.1f}"


def build_report(run_dirs: list[Path], judgments_file: str = "judgments.jsonl") -> tuple[str, dict[str, Any]]:
    """Markdown tables (one row per run) plus a JSON-friendly dict of the same numbers."""
    scored = [score_run(d, judgments_file) for d in run_dirs]
    runs = [rs for rs, _ in scored]
    group_keys = scored[0][1]

    lines = ["# Accuracy report", "",
             "Accuracy (%) = answers judged correct / answers graded.", ""]
    if len({rs.question_ids for rs in runs}) > 1:
        lines += ["> ⚠ The runs answered different sets of questions, so numbers are not directly comparable.", ""]

    lines += ["## Overall", "", "| run | model | prompt | accuracy | correct / graded | ungraded |",
              "|---|---|---|---|---|---|"]
    for rs in runs:
        lines.append(f"| {rs.name} | {rs.model} | {rs.prompt} | **{_fmt(rs.overall)}** | "
                     f"{rs.overall.correct} / {rs.overall.graded} | {rs.n_ungraded} |")

    for key in group_keys:
        values = sorted({v for rs in runs for v in rs.groups[key]})
        title = {"category": "By task type", "hallucination": "By hallucination (1 = target not in image)",
                 "instances": "By number of objects the answer is about"}
        lines += ["", f"## {title.get(key, f'By {key}')}", "",
                  "| model | " + " | ".join(f"{v} (n={runs[0].groups[key][v].graded})" for v in values) + " |",
                  "|---|" + "---|" * len(values)]
        for rs in runs:
            lines.append(f"| {rs.model} | " + " | ".join(_fmt(rs.groups[key].get(v)) for v in values) + " |")

    data = {
        rs.name: {
            "model": rs.model, "prompt": rs.prompt, "n_answers": rs.n_answers, "n_ungraded": rs.n_ungraded,
            "accuracy": rs.overall.acc, "correct": rs.overall.correct, "graded": rs.overall.graded,
            "by": {k: {v: {"accuracy": t.acc, "correct": t.correct, "graded": t.graded}
                       for v, t in sorted(rs.groups[k].items())} for k in group_keys},
        }
        for rs in runs
    }
    return "\n".join(lines) + "\n", data


def write_report(run_dirs: list[Path], out_dir: Path | None, judgments_file: str = "judgments.jsonl") -> str:
    md, data = build_report(run_dirs, judgments_file)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(md)
        (out_dir / "report.json").write_text(json.dumps(data, indent=2))
    return md
