"""Base VLM vs tool agent (v2) vs improved tool agent (v4 + evidence check), with reasoning on (no API calls).

    python analysis/vistaqa_reasoning.py --out runs/vistaqa_reasoning_analysis

Compares, on the questions in --ids (default: the 150-question dev set), for each model:
  reasoning on (medium):  baseline, v2 agent, v4 agent + evidence check (runs tagged "reason-medium")
  reasoning off (earlier): baseline, v2 agent (the full runs), for reference
Sections: accuracy overall / normal / trick and by task type, cost and tokens, tool use, evidence checks.
Writes analysis.md to --out.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vistaqa_baseline import table  # noqa: E402

from vlmbench.datasets import get_dataset  # noqa: E402
from vlmbench.runner import read_records  # noqa: E402
from vlmbench.types import usage_totals  # noqa: E402

MODELS = {"gemma-4-26b-a4b-it": "google_gemma-4-26b-a4b-it", "qwen3.8-27b": "qwen_qwen3.8-27b",
          "muse-glimmer-30b": "meta_muse-glimmer-30b"}
TASKS = ("attribute", "comparison", "counting", "identification", "ocr", "spatial")
TAG = "__reason-medium"


def variants(part: str) -> dict[str, str]:
    """Display name -> run folder name, in table order."""
    base = f"vistaqa__openrouter__{part}__default"
    v4 = "__agent-tools-v4-native-verify" if part.startswith("meta_") else "__agent-tools-v4-verify"
    return {"baseline (no reasoning)": base, "v2 agent (no reasoning)": base + "__agent-tools-v2",
            "baseline": base + TAG, "v2 agent": base + "__agent-tools-v2" + TAG, "v4 agent + check": base + v4 + TAG}


def load(runs_dir: Path, folder: str) -> dict | None:
    d = runs_dir / folder
    if not (d / "judgments.jsonl").exists():
        return None
    return {"responses": read_records(d / "responses.jsonl"), "judge": read_records(d / "judgments.jsonl")}


def acc(run, qs) -> str:
    graded = [q for q in qs if run["judge"].get(q, {}).get("correct") is not None]
    if not graded:
        return "–"
    c = sum(run["judge"][q]["correct"] for q in graded)
    return f"{100 * c / len(graded):.1f}" + ("" if len(graded) == len(qs) else f" ({len(graded)}/{len(qs)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--ids", type=Path, default=Path("analysis/dev150_ids.txt"))
    ap.add_argument("--out", type=Path, default=Path("runs/vistaqa_reasoning_analysis"))
    args = ap.parse_args()

    samples = {s.question_id: s for s in get_dataset("vistaqa").load()}
    qs = args.ids.read_text().split()
    normal = [q for q in qs if not samples[q].meta["hallucination"]]
    trick = [q for q in qs if samples[q].meta["hallucination"]]
    md = ["# VistaQA: baseline vs tool agent (v2) vs improved agent (v4 + evidence check)", "",
          f"{len(qs)} questions ({len(normal)} normal, {len(trick)} trick); accuracy % from the Qwen2.5-72B judge. "
          "Unless marked \"no reasoning\", runs use reasoning effort medium, temperature 0, top_p 1, "
          "max_tokens 10000.", ""]

    overall, by_task, cost, tools, checks = [], [], [], [], []
    for name, part in MODELS.items():
        for vname, folder in variants(part).items():
            run = load(args.runs, folder)
            if run is None:
                continue
            label = f"{name} · {vname}"
            overall.append([label, acc(run, qs), acc(run, normal), acc(run, trick)])
            by_task.append([label] + [acc(run, [q for q in normal if samples[q].category == t]) for t in TASKS])
            u = usage_totals(run["responses"][q].get("usage") for q in qs if q in run["responses"])
            n = u["calls"] or 1
            usd = "–" if u["cost_usd"] is None else f"${u['cost_usd']:.3f}"
            cost.append([label, usd, f"{u['input'] / n / 1000:.1f}k", f"{u['output'] / n:,.0f}",
                         f"{u['reasoning'] / n:,.0f}"])
            recs = [run["responses"][q] for q in qs if q in run["responses"] and run["responses"][q].get("trace")]
            if recs:
                per = Counter(s.get("tool") or "?" for r in recs for s in r["trace"] if "observation" in s)
                tools.append([label, f"{sum(per.values()) / len(recs):.2f}"] +
                             [str(per.get(t, 0)) for t in ("zoom", "grid", "detect", "count", "mark", "ocr",
                                                           "depth", "distance")])
                ver = [r for r in recs if any("verify" in s for s in r["trace"])]
                if ver:
                    rejected = [r for r in ver if any(c["ok"] is False for s in r["trace"] for c in s.get("verify", []))]
                    rq = [r["question_id"] for r in rejected]
                    checks.append([label, f"{len(ver)} / {len(recs)}", str(len(rejected)),
                                   acc(run, [q for q in rq if samples[q].meta["hallucination"]]) +
                                   f" of {sum(samples[q].meta['hallucination'] for q in rq)}",
                                   acc(run, [q for q in rq if not samples[q].meta["hallucination"]]) +
                                   f" of {sum(not samples[q].meta['hallucination'] for q in rq)}"])

    md += ["## Accuracy", ""] + table(["Run", "All", "Normal", "Trick"], overall) + [""]
    md += ["## Normal questions by task type", ""] + table(["Run", *TASKS], by_task) + [""]
    md += ["## Cost and tokens", "", "Cost as reported by OpenRouter (– = not recorded: runs made before cost "
           "tracking). Tokens per question.", ""]
    md += table(["Run", "Cost (answers)", "Input / q", "Output / q", "Reasoning / q"], cost) + [""]
    md += ["## Tool use", ""] + table(["Run", "Calls / q", "zoom", "grid", "detect", "count", "mark", "ocr",
                                       "depth", "distance"], tools) + [""]
    if checks:
        md += ["## Evidence checks (v4)", "",
               "Questions where the agent cited evidence and it was checked; \"rejected\" = at least one box "
               "failed, so the agent was asked to look again. The last two columns are the final accuracy on "
               "those rejected questions.", ""]
        md += table(["Run", "Checked", "Rejected", "Trick accuracy (rejected)", "Normal accuracy (rejected)"],
                    checks) + [""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved {args.out}/analysis.md")


if __name__ == "__main__":
    main()
