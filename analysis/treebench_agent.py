"""Tool agent vs base VLM on TreeBench (no API calls).

    python analysis/treebench_agent.py --out runs/treebench_agent_analysis \
        --pair "Qwen3.8 (think low)=treebench__openrouter__qwen_qwen3.8-27b__default__think-low:treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb1__think-low"

Each --pair is "label=<base run>:<agent run>" (folders under --runs-dir); compared on the questions both
answered. Writes analysis.md and analysis.json to --out:
1. Accuracy, all questions and without the tuning questions, with fixed / broken counts and an exact
   McNemar test.
2. By category (and Perception / Reasoning).
3. Tool use: calls per question, how often each tool was used, accuracy with and without each tool.
4. Perspective questions: how often viewpoint was used, and accuracy when it was.
5. Evidence: TreeBench box mIoU of the cited evidence, and how often a tool call covered a target.
6. Tokens and cost as reported by the provider.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from vlmbench.datasets import get_dataset
from vlmbench.datasets.treebench import evidence_scores
from vlmbench.runner import read_records
from vlmbench.types import usage_totals


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}" if n else "–"


def tools_used(rec: dict) -> list[str]:
    return [s["tool"] for s in rec.get("trace") or [] if s.get("tool") and "observation" in s]


def load_run(runs_dir: Path, name: str, dataset, samples) -> dict[str, dict]:
    recs = read_records(runs_dir / name / "responses.jsonl")
    if not recs:
        raise SystemExit(f"no responses in {runs_dir / name}")
    for q, r in recs.items():
        r["_ok"] = dataset.grade(samples[q], r)
    return recs


def analyse_pair(label, base_name, agent_name, base, agent, samples, sizes, tuning) -> tuple[list[str], dict]:
    qs = sorted(set(base) & set(agent))
    md = [f"## {label}", "", f"Base: `{base_name}`  ", f"Agent: `{agent_name}`  ",
          f"Questions answered by both: {len(qs)}", ""]
    out: dict = {"label": label, "base": base_name, "agent": agent_name, "n": len(qs)}

    # 1. accuracy
    md += ["| questions | n | base | agent | change | fixed | broken | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for name, sel in [("all", qs), ("without tuning set", [q for q in qs if q not in tuning])]:
        if not sel:
            continue
        b = sum(base[q]["_ok"] for q in sel)
        a = sum(agent[q]["_ok"] for q in sel)
        fixed = sum(agent[q]["_ok"] and not base[q]["_ok"] for q in sel)
        broken = sum(base[q]["_ok"] and not agent[q]["_ok"] for q in sel)
        p = mcnemar(fixed, broken)
        md.append(f"| {name} | {len(sel)} | {pct(b, len(sel))} | {pct(a, len(sel))} | "
                  f"{100 * (a - b) / len(sel):+.1f} | {fixed} | {broken} | {p:.3g} |")
        out[name] = {"n": len(sel), "base": b, "agent": a, "fixed": fixed, "broken": broken, "p": p}

    # 2. by category / group
    md += ["", "| category | n | base | agent | fixed | broken |", "|---|---|---|---|---|---|"]
    out["by_category"] = {}
    for key in ("group", "category"):
        groups = defaultdict(list)
        for q in qs:
            groups[samples[q].meta["group"] if key == "group" else samples[q].category].append(q)
        for g, sel in sorted(groups.items()):
            b = sum(base[q]["_ok"] for q in sel)
            a = sum(agent[q]["_ok"] for q in sel)
            fixed = sum(agent[q]["_ok"] and not base[q]["_ok"] for q in sel)
            broken = sum(base[q]["_ok"] and not agent[q]["_ok"] for q in sel)
            name = f"**{g}**" if key == "group" else g.split("/")[1]
            md.append(f"| {name} | {len(sel)} | {pct(b, len(sel))} | {pct(a, len(sel))} | {fixed} | {broken} |")
            out["by_category"][g] = {"n": len(sel), "base": b, "agent": a, "fixed": fixed, "broken": broken}

    # 3. tool use
    calls = {q: tools_used(agent[q]) for q in qs}
    n_calls = [len(c) for c in calls.values()]
    per_tool = Counter(t for c in calls.values() for t in set(c))
    md += ["", f"Tool calls per question: mean {sum(n_calls) / len(qs):.2f}; "
               f"no tool used on {sum(c == 0 for c in n_calls)} of {len(qs)} questions.", "",
           "| tool | questions using it | agent accuracy when used | base accuracy on those questions |",
           "|---|---|---|---|"]
    out["tools"] = {}
    for t, n in per_tool.most_common():
        sel = [q for q in qs if t in calls[q]]
        a = sum(agent[q]["_ok"] for q in sel)
        b = sum(base[q]["_ok"] for q in sel)
        md.append(f"| {t} | {n} | {pct(a, n)} | {pct(b, n)} |")
        out["tools"][t] = {"questions": n, "agent": a, "base": b}
    by_calls = defaultdict(list)
    for q in qs:
        by_calls[min(len(calls[q]), 4)].append(q)
    md += ["", "| tool calls | questions | agent accuracy | base accuracy |", "|---|---|---|---|"]
    for k in sorted(by_calls):
        sel = by_calls[k]
        md.append(f"| {k if k < 4 else '4+'} | {len(sel)} | {pct(sum(agent[q]['_ok'] for q in sel), len(sel))} | "
                  f"{pct(sum(base[q]['_ok'] for q in sel), len(sel))} |")

    # 4. perspective questions
    pt = [q for q in qs if samples[q].category.endswith("Perspective Transform")]
    if pt:
        vp = [q for q in pt if "viewpoint" in calls[q]]
        no_vp = [q for q in pt if "viewpoint" not in calls[q]]
        md += ["", f"Perspective questions: {len(pt)}; viewpoint used on {len(vp)} "
                   f"(agent {pct(sum(agent[q]['_ok'] for q in vp), len(vp))}%, base on the same "
                   f"{pct(sum(base[q]['_ok'] for q in vp), len(vp))}%); not used on {len(no_vp)} "
                   f"(agent {pct(sum(agent[q]['_ok'] for q in no_vp), len(no_vp))}%)."]
        out["perspective"] = {"n": len(pt), "viewpoint_used": len(vp),
                              "agent_when_used": sum(agent[q]["_ok"] for q in vp),
                              "base_when_used": sum(base[q]["_ok"] for q in vp)}

    # 5. evidence
    ev = [evidence_scores(samples[q], agent[q], sizes[q]) for q in qs]
    ev = [e for e in ev if e is not None]
    if ev:
        cited = [e for e in ev if e["cited"]]
        miou = 100 * sum(e["miou"] for e in ev) / len(ev)
        md += ["", f"Evidence: boxes cited in {len(cited)} of {len(ev)} answers; TreeBench mIoU {miou:.1f} "
                   f"({(100 * sum(e['miou'] for e in cited) / len(cited)) if cited else 0:.1f} when cited); a tool call covered a target "
                   f"on {pct(sum(e['inspected_target'] for e in ev), len(ev))}% of questions."]
        hit = [q for q, e in zip(qs, ev) if e["inspected_target"]]
        miss = [q for q, e in zip(qs, ev) if not e["inspected_target"]]
        md.append(f"Accuracy when a tool looked at a target: {pct(sum(agent[q]['_ok'] for q in hit), len(hit))}% "
                  f"({len(hit)}); when none did: {pct(sum(agent[q]['_ok'] for q in miss), len(miss))}% ({len(miss)}).")
        out["evidence"] = {"answers": len(ev), "cited": len(cited), "miou": miou,
                           "inspected": len(hit), "acc_inspected": sum(agent[q]["_ok"] for q in hit),
                           "acc_not_inspected": sum(agent[q]["_ok"] for q in miss)}

    # 6. cost
    ub = usage_totals(base[q].get("usage") for q in qs)
    ua = usage_totals(agent[q].get("usage") for q in qs)
    fmt = lambda u: "–" if u["cost_usd"] is None else f"${u['cost_usd']:.2f}"
    md += ["", "| | input tokens / question | output tokens / question | cost (provider-reported) |", "|---|---|---|---|",
           f"| base | {ub['input'] / len(qs):,.0f} | {ub['output'] / len(qs):,.0f} | {fmt(ub)} |",
           f"| agent | {ua['input'] / len(qs):,.0f} | {ua['output'] / len(qs):,.0f} | {fmt(ua)} |", ""]
    out["usage"] = {"base": ub, "agent": ua}
    return md, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--pair", action="append", required=True, help='"label=<base run>:<agent run>"')
    ap.add_argument("--tuning", default="analysis/treebench_tuning_ids.txt")
    ap.add_argument("--out")
    args = ap.parse_args()

    ds = get_dataset("treebench")
    samples = {s.question_id: s for s in ds.load()}
    sizes = {}
    for q, s in samples.items():
        with Image.open(s.image_path) as im:
            sizes[q] = im.size
    tuning = set(Path(args.tuning).read_text().split()) if Path(args.tuning).exists() else set()
    runs_dir = Path(args.runs_dir)

    md = ["# TreeBench: tool agent vs base model", "",
          "Accuracy in %. Fixed = right with the agent, wrong without; broken = the reverse. "
          f"Tuning set: {len(tuning)} questions used to develop the agent (reported separately).", ""]
    data = []
    for spec in args.pair:
        label, _, runs = spec.partition("=")
        base_name, _, agent_name = runs.partition(":")
        base = load_run(runs_dir, base_name, ds, samples)
        agent = load_run(runs_dir, agent_name, ds, samples)
        pmd, pdata = analyse_pair(label, base_name, agent_name, base, agent, samples, sizes, tuning)
        md += pmd
        data.append(pdata)
    text = "\n".join(md) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.md").write_text(text)
        (out / "analysis.json").write_text(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
