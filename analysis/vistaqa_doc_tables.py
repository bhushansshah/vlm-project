"""Tables for the VistaQA write-up: baselines and tool agents, with and without reasoning (no API calls).

    python analysis/vistaqa_doc_tables.py --out runs/vistaqa_doc_tables

Runs (all 1,157 questions, Qwen2.5-72B judge):
  no reasoning (configs/vistaqa):            baseline, v2 agent
  reasoning medium (configs/vistaqa_reasoning, tag "reason-medium"): baseline, v2 agent, v4 agent + check
Cost: runs with reasoning record the cost OpenRouter reports for every call. The earlier runs did not, so
their cost is estimated at list price (tokens x OpenRouter's listed price per token); the judge's cost for
them is estimated from the judge's average cost per answer in the newer runs.
Writes tables.md and tables.json to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vistaqa_baseline import table  # noqa: E402

from vlmbench.datasets import get_dataset  # noqa: E402
from vlmbench.runner import read_records  # noqa: E402
from vlmbench.types import usage_totals  # noqa: E402

# short name -> (run-name part, list price $ per million input / output tokens)
MODELS = {"Gemma 4 26B": ("google_gemma-4-26b-a4b-it", 0.0675, 0.225),
          "Qwen3.8 27B": ("qwen_qwen3.8-27b", 0.20, 2.50),
          "Muse Glimmer 30B": ("meta_muse-glimmer-30b", 0.30, 1.20)}
TASKS = ("attribute", "comparison", "counting", "identification", "ocr", "spatial")
TASK_NAMES = {"ocr": "OCR"}
R = "__reason-medium"


def folders(part: str) -> dict[str, str]:
    base = f"vistaqa__openrouter__{part}__default"
    v4 = "__agent-tools-v4-native-verify" if part.startswith("meta_") else "__agent-tools-v4-verify"
    return {"base": base, "v2": base + "__agent-tools-v2", "base_r": base + R, "v2_r": base + "__agent-tools-v2" + R,
            "v4_r": base + v4 + R}


def load(d: Path) -> dict:
    return {"responses": read_records(d / "responses.jsonl"), "judge": read_records(d / "judgments.jsonl")}


def score(run, qs) -> tuple[int, int]:
    c = sum(run["judge"][q]["correct"] == 1 for q in qs if q in run["judge"])
    return c, len(qs)


def pct(run, qs) -> str:
    c, n = score(run, qs)
    return f"{100 * c / n:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/vistaqa_doc_tables"))
    args = ap.parse_args()

    samples = {s.question_id: s for s in get_dataset("vistaqa").load()}
    allq = sorted(samples, key=int)
    groups = {"normal": [q for q in allq if not samples[q].meta["hallucination"]],
              "trick": [q for q in allq if samples[q].meta["hallucination"]]}
    runs = {m: {k: load(args.runs / f) for k, f in folders(p).items()} for m, (p, _, _) in MODELS.items()}
    for m, rs in runs.items():  # every run must be complete and fully graded
        for k, r in rs.items():
            missing = [q for q in allq if r["judge"].get(q, {}).get("correct") is None]
            assert not missing, f"{m} {k}: {len(missing)} questions not graded"

    data: dict = {"accuracy": {}, "cost": {}}
    for m, rs in runs.items():
        data["accuracy"][m] = {k: {g: score(r, qs) for g, qs in [("all", allq), *groups.items()]} |
                               {f"{g}/{t}": score(r, [q for q in qs if samples[q].category == t])
                                for g, qs in groups.items() for t in TASKS} for k, r in rs.items()}

    # ---- cost: reported (reasoning runs) or list-price estimate (earlier runs)
    judge_per_answer = []
    for m, rs in runs.items():
        for k in ("base_r", "v2_r", "v4_r"):
            u = usage_totals(rs[k]["judge"][q].get("usage") for q in allq)
            judge_per_answer.append(u["cost_usd"] / u["calls"])
    judge_avg = sum(judge_per_answer) / len(judge_per_answer)
    for m, (part, p_in, p_out) in MODELS.items():
        data["cost"][m] = {}
        for k, r in runs[m].items():
            u = usage_totals(r["responses"][q].get("usage") for q in allq)
            j = usage_totals(r["judge"][q].get("usage") for q in allq)
            list_price = (u["input"] * p_in + u["output"] * p_out) / 1e6
            data["cost"][m][k] = {
                "answers": u["cost_usd"] if u["cost_usd"] is not None else list_price,
                "answers_estimated": u["cost_usd"] is None, "answers_list_price": list_price,
                "judge": j["cost_usd"] if j["cost_usd"] is not None else judge_avg * len(allq),
                "judge_estimated": j["cost_usd"] is None,
                "input_per_q": u["input"] / len(allq), "output_per_q": u["output"] / len(allq),
                "reasoning_per_q": u["reasoning"] / len(allq)}

    # ---- markdown preview (the doc tables are built from tables.json)
    md = ["# VistaQA tables", ""]
    for label, ks in [("Baselines", ("base", "base_r")), ("v2 agent", ("v2", "v2_r", "base_r")),
                      ("v4 agent", ("v4_r", "v2_r", "base_r"))]:
        rows = [[m] + [f"{100 * data['accuracy'][m][k]['all'][0] / 1157:.1f}%" for k in ks] for m in MODELS]
        md += [f"## {label}: overall", ""] + table(["Model", *ks], rows) + [""]
        for g in groups:
            rows = [[TASK_NAMES.get(t, t.capitalize()), m] +
                    [pct(runs[m][k], [q for q in groups[g] if samples[q].category == t]) for k in ks]
                    for t in TASKS for m in MODELS]
            md += [f"### {label}: {g}", ""] + table(["Category", "Model", *ks], rows) + [""]
    rows = [[m, k, f"${c['answers']:.2f}" + (" (est.)" if c["answers_estimated"] else ""),
             f"${c['answers_list_price']:.2f}", f"${c['judge']:.2f}" + (" (est.)" if c["judge_estimated"] else ""),
             f"{c['input_per_q'] / 1000:.1f}k", f"{c['output_per_q']:.0f}", f"{c['reasoning_per_q']:.0f}"]
            for m in MODELS for k, c in data["cost"][m].items()]
    md += ["## Cost", ""] + table(["Model", "Run", "Answers", "List price", "Judge", "Input/q", "Output/q",
                                   "Reasoning/q"], rows) + [""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text("\n".join(md) + "\n")
    (args.out / "tables.json").write_text(json.dumps(data, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
