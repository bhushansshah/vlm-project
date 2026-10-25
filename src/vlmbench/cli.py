"""Command line interface.

    vlmbench prepare --dataset vistaqa                     # download a benchmark once
    vlmbench run --dataset vistaqa --model openrouter:qwen/qwen3.8-27b --per-category 100 --sample-seed 0
    vlmbench run --config configs/vistaqa/qwen3.8-27b.yaml
    vlmbench judge runs/<run_name>                         # grade answers (datasets with public answers)
    vlmbench report runs/<run_a> runs/<run_b> --out runs/report
    vlmbench finalize runs/<run_name>                      # rebuild summary / output files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from .datasets import DATASETS, get_dataset
from .models import available_providers, load_model
from .judge import DEFAULT_JUDGE
from .runner import RunConfig, Runner, finalize, load_run_config


def _kv(items: list[str] | None) -> dict[str, Any]:
    """Parse ["a=1", "b={x: 2}"] into {"a": 1, "b": {"x": 2}} (values are YAML-typed)."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"Expected key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(val)
    return out


def _filters(items: list[str] | None) -> dict[str, list[str]]:
    """Parse ["domain=av,robotics", "hallucination=1"] into {"domain": ["av", "robotics"], ...}."""
    out: dict[str, list[str]] = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"Expected key=value1,value2, got {item!r}")
        out[key.strip()] = [v.strip() for v in val.split(",") if v.strip()]
    return out


def _question_ids(values: list[str] | None) -> list[str]:
    """IDs given inline, or as files of whitespace-separated IDs (any value that is a file)."""
    ids: list[str] = []
    for v in values or []:
        ids += Path(v).read_text().split() if Path(v).is_file() else [v]
    return ids


def build_config(args: argparse.Namespace) -> RunConfig:
    from .types import GenerationConfig

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    gen = dict(cfg.pop("generation", None) or {})

    # CLI flags override the config file
    overrides = {
        "dataset": args.dataset, "model": args.model, "prompt": args.prompt, "run_name": args.run_name,
        "output_dir": args.output_dir, "concurrency": args.concurrency, "batch_size": args.batch_size,
        "max_retries": args.max_retries, "data_dir": args.data_dir, "categories": args.categories,
        "per_category": args.per_category, "limit": args.limit, "sample_seed": args.sample_seed,
        "agent": args.agent,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.retry_errors:
        cfg["retry_errors"] = True
    if args.question_ids:
        cfg["question_ids"] = _question_ids(args.question_ids)
    if args.filter:
        cfg["filters"] = {**(cfg.get("filters") or {}), **_filters(args.filter)}
    cfg["model_options"] = {**(cfg.get("model_options") or {}), **_kv(args.opt)}
    cfg["agent_options"] = {**(cfg.get("agent_options") or {}), **_kv(args.agent_opt)}

    for k in ("max_tokens", "temperature", "top_p", "seed"):
        if getattr(args, k) is not None:
            gen[k] = getattr(args, k)
    gen["extra"] = {**(gen.get("extra") or {}), **_kv(args.gen_opt)}

    if not cfg.get("model"):
        raise SystemExit("--model (or `model:` in --config) is required")
    return RunConfig(**cfg, generation=GenerationConfig(**gen))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    dataset = get_dataset(cfg.dataset, cfg.data_dir)
    samples = dataset.load(cfg.selection())
    if cfg.question_ids:
        unknown = set(cfg.question_ids) - {s.question_id for s in samples}
        if unknown:
            print(f"WARNING: {len(unknown)} question id(s) not found or filtered out: {sorted(unknown)[:10]}",
                  file=sys.stderr)
    if not samples:
        raise SystemExit("No questions match the selection")
    model = load_model(cfg.model, **cfg.model_options)
    if cfg.agent:
        from .agents import build_agent
        model = build_agent(cfg.agent, model, **cfg.agent_options)
    summary = Runner(model, dataset, cfg).run(samples)
    print(json.dumps(summary, indent=2))
    print(f"\nRun directory: {cfg.run_dir()}")
    if summary["n_errors"]:
        print(f"WARNING: {summary['n_errors']} questions errored; re-run the same command with --retry-errors.")


def cmd_prepare(args: argparse.Namespace) -> None:
    print(f"Dataset ready at {get_dataset(args.dataset, args.data_dir).prepare(force=args.force)}")


def cmd_finalize(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)
    dataset = get_dataset(cfg.get("dataset", "visualoverload"), cfg.get("data_dir"))
    print(json.dumps(finalize(run_dir, dataset), indent=2))


def cmd_judge(args: argparse.Namespace) -> None:
    from .judge import Judge, judge_run

    judge = Judge(load_model(args.judge, **_kv(args.opt)), args.judge, max_retries=args.max_retries,
                  extra=_kv(args.gen_opt))
    for d in args.run_dirs:
        stats = judge_run(Path(d), judge, concurrency=args.concurrency, filename=args.save_as)
        print(f"{d}: {stats}")
        if stats["judge_errors"]:
            print(f"  WARNING: {stats['judge_errors']} answers could not be judged; run the same command again.")


def cmd_report(args: argparse.Namespace) -> None:
    from .report import write_report

    print(write_report([Path(d) for d in args.run_dirs], Path(args.out) if args.out else None, args.judgments,
                       common=not args.all_answers))
    if args.out:
        print(f"Saved {args.out}/report.md and report.json")


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    p = argparse.ArgumentParser(prog="vlmbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    datasets = sorted(DATASETS)

    pp = sub.add_parser("prepare", help="download and extract a benchmark")
    pp.add_argument("--dataset", default="visualoverload", choices=datasets)
    pp.add_argument("--data-dir")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="run a model over a benchmark")
    pr.add_argument("--config", help="YAML run config (CLI flags override it)")
    pr.add_argument("--dataset", choices=datasets, help="benchmark (default: visualoverload)")
    pr.add_argument("--model", help=f"<provider>:<model>, providers: {', '.join(available_providers())}")
    pr.add_argument("-o", "--opt", action="append", metavar="KEY=VALUE",
                    help="backend option, e.g. -o image_detail=high -o base_url=http://node:8000/v1")
    pr.add_argument("--prompt", help="prompt strategy (default: 'default')")
    pr.add_argument("--max-tokens", type=int)
    pr.add_argument("--temperature", type=float)
    pr.add_argument("--top-p", type=float)
    pr.add_argument("--seed", type=int, help="sampling seed passed to the model")
    pr.add_argument("-g", "--gen-opt", action="append", metavar="KEY=VALUE",
                    help="provider-specific generation option, e.g. -g 'reasoning={enabled: false}'")
    pr.add_argument("--agent", help="wrap the model in an agent: 'tools' (tool-augmented ReAct agent)")
    pr.add_argument("-a", "--agent-opt", action="append", metavar="KEY=VALUE",
                    help="agent option, e.g. -a 'tools=[zoom, detect]' -a max_steps=4")
    pr.add_argument("--run-name")
    pr.add_argument("--output-dir")
    pr.add_argument("--concurrency", type=int, help="parallel requests (API / server backends)")
    pr.add_argument("--batch-size", type=int, help="batch size (in-process local backend)")
    pr.add_argument("--max-retries", type=int)
    pr.add_argument("--retry-errors", action="store_true", help="re-run questions that errored before")
    sel = pr.add_argument_group("question selection (combinable; default: all questions)")
    sel.add_argument("--data-dir")
    sel.add_argument("--categories", nargs="+", help="task types / categories to keep")
    sel.add_argument("--question-ids", nargs="+", metavar="ID_OR_FILE",
                     help="question ids, inline or as files of whitespace-separated ids")
    sel.add_argument("--filter", action="append", metavar="KEY=V1,V2",
                     help="keep questions whose metadata matches, e.g. --filter difficulty=hard "
                          "or --filter domain=av,robotics")
    sel.add_argument("--per-category", type=int, help="random N questions per category (balanced subset)")
    sel.add_argument("--limit", type=int)
    sel.add_argument("--sample-seed", "--shuffle-seed", dest="sample_seed", type=int,
                     help="seed for --per-category sampling and shuffling before --limit")
    pr.set_defaults(func=cmd_run)

    pj = sub.add_parser("judge", help="grade a run's answers with an LLM judge")
    pj.add_argument("run_dirs", nargs="+")
    pj.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge model spec (default: {DEFAULT_JUDGE})")
    pj.add_argument("-o", "--opt", action="append", metavar="KEY=VALUE", help="judge backend option")
    pj.add_argument("-g", "--gen-opt", action="append", metavar="KEY=VALUE",
                    help="judge generation option, e.g. -g 'reasoning={enabled: false}'")
    pj.add_argument("--save-as", default="judgments.jsonl",
                    help="judgments file inside each run dir (use another name to keep a second judge's grades)")
    pj.add_argument("--concurrency", type=int, default=8)
    pj.add_argument("--max-retries", type=int, default=5)
    pj.set_defaults(func=cmd_judge)

    prp = sub.add_parser("report", help="accuracy tables for one or more judged runs")
    prp.add_argument("run_dirs", nargs="+")
    prp.add_argument("--out", help="directory to save report.md and report.json")
    prp.add_argument("--judgments", default="judgments.jsonl", help="which judgments file to report on")
    prp.add_argument("--all-answers", action="store_true",
                     help="score each run on all its answers, even if the runs answered different questions "
                          "(default: only the questions every run answered)")
    prp.set_defaults(func=cmd_report)

    pf = sub.add_parser("finalize", help="rebuild summary.json and output files for a run")
    pf.add_argument("run_dir")
    pf.set_defaults(func=cmd_finalize)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "anthropic", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
