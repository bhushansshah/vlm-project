"""Accuracy reports from judged runs: overall and broken down by category and dataset groups."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets import get_dataset
from .runner import load_run_config, read_records
from .types import usage_totals


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
    answer_usage: dict[str, Any] = field(default_factory=dict)  # usage_totals() of the scored answers
    judge_usage: dict[str, Any] = field(default_factory=dict)  # ... and of their judgments


def answered(run_dir: Path) -> set[str]:
    return set(read_records(run_dir / "responses.jsonl"))


def score_run(run_dir: Path, judgments_file: str = "judgments.jsonl",
              only: set[str] | None = None) -> tuple[RunScores, tuple[str, ...]]:
    """Score one run; ``only`` restricts it to these question ids."""
    cfg = load_run_config(run_dir)
    dataset = get_dataset(cfg["dataset"], cfg.get("data_dir"))
    samples = {s.question_id: s for s in dataset.load()}
    responses = {q: r for q, r in read_records(run_dir / "responses.jsonl").items()
                 if q in samples and (only is None or q in only)}
    local = dataset.grades_locally  # e.g. multiple choice: graded by rule, no judge file
    judgments = {} if local else read_records(run_dir / judgments_file)
    if not local and not judgments:
        raise SystemExit(f"{run_dir} has no judgments yet; run `vlmbench judge {run_dir}` first")

    label = cfg["model"] + (f" + {cfg['agent']} agent" if cfg.get("agent") else "")
    rs = RunScores(name=run_dir.name, model=label, prompt=cfg["prompt"],
                   n_answers=len(responses), n_ungraded=0, question_ids=frozenset(responses))
    for qid, resp in responses.items():
        s = samples[qid]
        if local:
            ok = dataset.grade(s, resp)
        else:
            j = judgments.get(qid)
            if j is None or j["correct"] is None or j["judged_response"] != resp["response"]:
                rs.n_ungraded += 1
                continue
            ok = j["correct"] == 1
        rs.overall.add(ok)
        rs.groups["category"][s.category].add(ok)
        for key in dataset.report_groups:
            rs.groups[key][str(s.meta.get(key))].add(ok)
    rs.answer_usage = usage_totals(r.get("usage") for r in responses.values())
    rs.judge_usage = usage_totals(judgments[q].get("usage") for q in responses if q in judgments)
    return rs, ("category", *dataset.report_groups)


def _fmt(t: Tally | None) -> str:
    return "–" if t is None or t.acc is None else f"{t.acc:.1f}"


def _fmt_cost(u: dict[str, Any]) -> str:
    """$ as reported by the provider; '–' when it reports none; flags a cost that covers only some calls."""
    if not u or u["cost_usd"] is None:
        return "–"
    c = u["cost_usd"]
    s = f"${c:.2f}" if c >= 0.01 or c == 0 else f"${c:.4f}"
    return s if u["cost_reported_calls"] == u["calls"] else f"{s} ({u['cost_reported_calls']} of {u['calls']} calls)"


def build_report(run_dirs: list[Path], judgments_file: str = "judgments.jsonl",
                 common: bool = True) -> tuple[str, dict[str, Any]]:
    """Markdown tables (one row per run) plus a JSON-friendly dict of the same numbers.
    With ``common`` (default), runs that answered different questions (e.g. an agent run on a
    subset vs a full baseline) are all scored on the questions every run answered."""
    sets = [answered(d) for d in run_dirs]
    shared = set.intersection(*sets) if sets else set()
    differ = len({frozenset(s) for s in sets}) > 1
    only = shared if (common and differ) else None
    scored = [score_run(d, judgments_file, only) for d in run_dirs]
    runs = [rs for rs, _ in scored]
    group_keys = scored[0][1]

    lines = ["# Accuracy report", "",
             "Accuracy (%) = answers graded correct (by the LLM judge, or by rule for multiple-choice benchmarks) / answers graded.", ""]
    if only is not None:
        lines += [f"> The runs answered different questions; all are scored on the {len(shared)} questions "
                  "every run answered.", ""]
    elif differ:
        lines += ["> ⚠ The runs answered different sets of questions, so numbers are not directly comparable.", ""]

    lines += ["## Overall", "", "| run | model | prompt | accuracy | correct / graded | ungraded |",
              "|---|---|---|---|---|---|"]
    for rs in runs:
        lines.append(f"| {rs.name} | {rs.model} | {rs.prompt} | **{_fmt(rs.overall)}** | "
                     f"{rs.overall.correct} / {rs.overall.graded} | {rs.n_ungraded} |")

    for key in group_keys:
        values = sorted({v for rs in runs for v in rs.groups[key]})
        title = {"category": "By task type", "hallucination": "By hallucination (1 = target not in image)",
                 "instances": "By number of objects the answer is about", "group": "By group (Perception / Reasoning)"}
        lines += ["", f"## {title.get(key, f'By {key}')}", "",
                  "| model | " + " | ".join(f"{v} (n={runs[0].groups[key][v].graded})" for v in values) + " |",
                  "|---|" + "---|" * len(values)]
        for rs in runs:
            lines.append(f"| {rs.model} | " + " | ".join(_fmt(rs.groups[key].get(v)) for v in values) + " |")

    lines += ["", "## Cost", "",
              "As reported by the provider (OpenRouter reports it; the OpenAI, Anthropic and Gemini APIs and "
              "local models do not: –). Cached = prompt tokens read from the provider's cache.", "",
              "| model | answers cost | judge cost | input tokens / answer | cached share |", "|---|---|---|---|---|"]
    for rs in runs:
        u = rs.answer_usage
        per = f"{u['input'] / u['calls']:,.0f}" if u.get("calls") else "–"
        cached = f"{100 * u['cached'] / u['input']:.0f}%" if u.get("input") and u["cached"] is not None else "–"
        lines.append(f"| {rs.model} | {_fmt_cost(u)} | {_fmt_cost(rs.judge_usage)} | {per} | {cached} |")

    data = {
        rs.name: {
            "model": rs.model, "prompt": rs.prompt, "n_answers": rs.n_answers, "n_ungraded": rs.n_ungraded,
            "accuracy": rs.overall.acc, "correct": rs.overall.correct, "graded": rs.overall.graded,
            "usage": {"answers": rs.answer_usage, "judge": rs.judge_usage},
            "by": {k: {v: {"accuracy": t.acc, "correct": t.correct, "graded": t.graded}
                       for v, t in sorted(rs.groups[k].items())} for k in group_keys},
        }
        for rs in runs
    }
    return "\n".join(lines) + "\n", data


def write_report(run_dirs: list[Path], out_dir: Path | None, judgments_file: str = "judgments.jsonl",
                 common: bool = True) -> str:
    md, data = build_report(run_dirs, judgments_file, common)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(md)
        (out_dir / "report.json").write_text(json.dumps(data, indent=2))
    return md
