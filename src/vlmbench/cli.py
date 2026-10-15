"""Command line interface.

    vlmbench prepare                                   # download + extract the dataset once
    vlmbench run --model openai:gpt-5-mini --limit 20  # quick smoke run
    vlmbench run --config configs/qwen25vl_7b_local.yaml
    vlmbench finalize runs/<run_name>                  # rebuild submission.json / summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from .data import CATEGORIES, DIFFICULTIES, load_samples, prepare
from .models import available_providers, load_model
from .runner import RunConfig, Runner, finalize
from .types import GenerationConfig


def _kv(items: list[str] | None) -> dict[str, Any]:
    """Parse ["a=1", "b={x: 2}"] into {"a": 1, "b": {"x": 2}} (values are YAML-typed)."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"Expected key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(val)
    return out


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    gen = dict(cfg.pop("generation", None) or {})

    # CLI flags override the config file
    overrides = {
        "model": args.model, "prompt": args.prompt, "run_name": args.run_name,
        "output_dir": args.output_dir, "concurrency": args.concurrency, "batch_size": args.batch_size,
        "max_retries": args.max_retries, "data_dir": args.data_dir, "categories": args.categories,
        "difficulties": args.difficulties, "limit": args.limit, "shuffle_seed": args.shuffle_seed,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.retry_errors:
        cfg["retry_errors"] = True
    if args.question_ids:
        cfg["question_ids"] = [q.strip() for q in Path(args.question_ids).read_text().split() if q.strip()]
    cfg["model_options"] = {**(cfg.get("model_options") or {}), **_kv(args.opt)}

    for k in ("max_tokens", "temperature", "top_p", "seed"):
        if getattr(args, k) is not None:
            gen[k] = getattr(args, k)
    gen["extra"] = {**(gen.get("extra") or {}), **_kv(args.gen_opt)}

    if not cfg.get("model"):
        raise SystemExit("--model (or `model:` in --config) is required")
    return RunConfig(**cfg, generation=GenerationConfig(**gen))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    samples = load_samples(
        Path(cfg.data_dir) if cfg.data_dir else None, cfg.categories, cfg.difficulties,
        cfg.question_ids, cfg.limit, cfg.shuffle_seed,
    )
    if not samples:
        raise SystemExit("No questions match the selection")
    model = load_model(cfg.model, **cfg.model_options)
    summary = Runner(model, cfg).run(samples)
    run_dir = Path(cfg.output_dir) / cfg.resolved_run_name()
    print(json.dumps(summary, indent=2))
    print(f"\nRun directory: {run_dir}")
    if summary["n_errors"]:
        print(f"WARNING: {summary['n_errors']} questions errored; re-run with --retry-errors before submitting.")
    if len(samples) < 2720:
        print(f"NOTE: subset of {len(samples)}/2720 questions; the evaluation server scores missing ones as wrong.")
    print(f"Submit {run_dir / 'submission.json'} at https://huggingface.co/spaces/paulgavrikov/visualoverload-submit")


def cmd_prepare(args: argparse.Namespace) -> None:
    path = prepare(Path(args.data_dir) if args.data_dir else None, force=args.force)
    print(f"Dataset ready at {path}")


def cmd_finalize(args: argparse.Namespace) -> None:
    print(json.dumps(finalize(Path(args.run_dir)), indent=2))


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

    pp = sub.add_parser("prepare", help="download and extract the VisualOverload dataset")
    pp.add_argument("--data-dir")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="run a model over the benchmark")
    pr.add_argument("--config", help="YAML run config (CLI flags override it)")
    pr.add_argument("--model", help=f"<provider>:<model>, providers: {', '.join(available_providers())}")
    pr.add_argument("-o", "--opt", action="append", metavar="KEY=VALUE",
                    help="backend option, e.g. -o image_detail=high -o base_url=http://node:8000/v1")
    pr.add_argument("--prompt", help="prompt strategy (default: 'default', the paper's prompt)")
    pr.add_argument("--max-tokens", type=int)
    pr.add_argument("--temperature", type=float)
    pr.add_argument("--top-p", type=float)
    pr.add_argument("--seed", type=int)
    pr.add_argument("-g", "--gen-opt", action="append", metavar="KEY=VALUE",
                    help="provider-specific generation option, e.g. -g 'reasoning={effort: low}'")
    pr.add_argument("--run-name")
    pr.add_argument("--output-dir")
    pr.add_argument("--concurrency", type=int, help="parallel requests (API / server backends)")
    pr.add_argument("--batch-size", type=int, help="batch size (in-process local backend)")
    pr.add_argument("--max-retries", type=int)
    pr.add_argument("--retry-errors", action="store_true", help="re-run questions that errored before")
    pr.add_argument("--data-dir")
    pr.add_argument("--categories", nargs="+", choices=CATEGORIES)
    pr.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES)
    pr.add_argument("--question-ids", help="file with whitespace-separated question ids")
    pr.add_argument("--limit", type=int)
    pr.add_argument("--shuffle-seed", type=int, help="shuffle before --limit (for random subsets)")
    pr.set_defaults(func=cmd_run)

    pf = sub.add_parser("finalize", help="rebuild submission.json and summary.json for a run")
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
