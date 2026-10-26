"""Tool agent vs base VLM on VistaQA (no API calls).

    python analysis/vistaqa_agent.py --out runs/vistaqa_agent_analysis

For each model, pairs the base run with its tool-agent run and compares them on the same questions:
1. Overall accuracy (all questions, and without the 30 prompt-tuning questions), with fixed / broken counts.
2. By task type, normal vs trick.
3. By domain and number of answer objects.
4. By size of the answer object.
5. Counting errors (exact / too high / too low).
6. Tool use: how often each tool was called, and accuracy by number of calls.
7. Tokens and list-price cost.
Writes analysis.md to --out.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from vistaqa_baseline import add_question_facts, bin_of, first_number, fmt_frac, pct, quantile_bins, table
from vlmbench.datasets import get_dataset
from vlmbench.runner import read_records

# short name -> (run-name model part, OpenRouter list price in $ per million input / output tokens)
MODELS = {
    "gemma-4-26b-a4b-it": ("google_gemma-4-26b-a4b-it", 0.0675, 0.225),
    "qwen3.8-27b": ("qwen_qwen3.8-27b", 0.20, 2.50),
    "muse-glimmer-30b": ("meta_muse-glimmer-30b", 0.30, 1.20),
}
AGENT_SUFFIX = "__agent-tools-v2"
TASKS = ("attribute", "comparison", "counting", "identification", "ocr", "spatial")


def load_pairs(runs_dir: Path) -> dict[str, dict]:
    pairs = {}
    for name, (part, p_in, p_out) in MODELS.items():
        base = runs_dir / f"vistaqa__openrouter__{part}__default"
        agent = base.with_name(base.name + AGENT_SUFFIX)
        if not (agent / "judgments.jsonl").exists():
            print(f"skipping {name}: no judged agent run at {agent}")
            continue
        pairs[name] = {
            "base": {"responses": read_records(base / "responses.jsonl"), "judge": read_records(base / "judgments.jsonl")},
            "agent": {"responses": read_records(agent / "responses.jsonl"), "judge": read_records(agent / "judgments.jsonl")},
            "price": (p_in, p_out),
        }
    return pairs


def acc(run, qs) -> float:
    return 100 * sum(run["judge"][q]["correct"] for q in qs) / len(qs) if qs else float("nan")


def cell(p, qs) -> str:
    """'base -> agent (+gain)' accuracy on these questions."""
    if not qs:
        return "–"
    b, a = acc(p["base"], qs), acc(p["agent"], qs)
    return f"{b:.1f} → {a:.1f} ({a - b:+.1f})"


def common(pairs) -> list[str]:
    qs = None
    for p in pairs.values():
        both = {q for q in p["agent"]["judge"] if q in p["base"]["judge"]}
        qs = both if qs is None else qs & both
    return sorted(qs, key=int)


def overall_section(pairs, qs, tuning) -> list[str]:
    rest = [q for q in qs if q not in tuning]
    rows = []
    for name, p in pairs.items():
        fixed = sum(p["agent"]["judge"][q]["correct"] and not p["base"]["judge"][q]["correct"] for q in qs)
        broke = sum(p["base"]["judge"][q]["correct"] and not p["agent"]["judge"][q]["correct"] for q in qs)
        rows.append([name, f"{acc(p['base'], qs):.1f}%", f"{acc(p['agent'], qs):.1f}%",
                     f"{acc(p['agent'], qs) - acc(p['base'], qs):+.1f}", cell(p, rest), str(fixed), str(broke)])
    return ["## 1. Overall", "",
            f"{len(qs)} questions, same judge (Qwen2.5-72B) for both. \"Fixed\" = base wrong, agent right; "
            "\"broken\" = the other way round.", ""] + table(
        ["Model", "Base", "Agent", "Gain", f"Without the {len(tuning)} tuning questions", "Fixed", "Broken"], rows) + [""]


def group_section(title, note, pairs, groups: dict[str, list[str]]) -> list[str]:
    rows = [[f"{g} (n={len(qs)})"] + [cell(p, qs) for p in pairs.values()] for g, qs in groups.items()]
    return [f"## {title}", "", note, ""] + table(["Group"] + list(pairs), rows) + [""]


def task_section(pairs, qs, facts) -> list[str]:
    out = []
    for trick, label in [(0, "normal"), (1, "trick")]:
        groups = {t: [q for q in qs if facts[q]["sample"].category == t and facts[q]["sample"].meta["hallucination"] == trick]
                  for t in TASKS}
        out += group_section(f"2{'ab'[trick]}. By task type, {label} questions",
                             "Accuracy %: base → agent (gain).", pairs, groups)
    return out


def domain_section(pairs, qs, facts) -> list[str]:
    domains = sorted({facts[q]["sample"].meta["domain"] for q in qs})
    groups = {d: [q for q in qs if facts[q]["sample"].meta["domain"] == d] for d in domains}
    out = group_section("3a. By domain", "Accuracy %: base → agent (gain).", pairs, groups)
    inst = defaultdict(list)
    for q in qs:
        inst[facts[q]["sample"].meta["instances"]].append(q)
    out += group_section("3b. By number of answer objects", "Trick questions have none.", pairs,
                         {g: inst[g] for g in ("none", "one", "many") if inst[g]})
    return out


def object_size_section(pairs, qs, facts) -> list[str]:
    real = [q for q in qs if facts[q]["obj_frac"] is not None]
    edges = quantile_bins([facts[q]["obj_side_px"] for q in real], 5)
    bounds = [None] + edges + [None]
    groups = defaultdict(list)
    for q in real:
        groups[bin_of(facts[q]["obj_side_px"], edges)].append(q)
    named = {}
    for b in range(5):
        lo, hi = bounds[b], bounds[b + 1]
        named[f"< {hi:.0f} px" if lo is None else f">= {lo:.0f} px" if hi is None else f"{lo:.0f} – {hi:.0f} px"] = groups[b]
    return group_section("4. By size of the answer object",
                         "Normal questions only. Size = width of the median answer object (square root of its mask area). "
                         "Each group holds about a fifth of the questions.", pairs, named)


def counting_section(pairs, qs, facts) -> list[str]:
    cq = [q for q in qs if facts[q]["sample"].category == "counting"
          and facts[q]["sample"].meta["hallucination"] == 0 and facts[q]["gt_count"] is not None]
    rows = []
    for name, p in pairs.items():
        for which in ("base", "agent"):
            got = {q: first_number(p[which]["responses"][q]["response"]) for q in cq}
            diffs = [got[q] - facts[q]["gt_count"] for q in cq if got[q] is not None]
            n = len(diffs)
            rows.append([f"{name} {which}", pct(sum(d == 0 for d in diffs), n), pct(sum(d > 0 for d in diffs), n),
                         pct(sum(d < 0 for d in diffs), n), f"{statistics.mean(abs(d) for d in diffs):.1f}"])
    return ["## 5. Counting errors", "",
            f"{len(cq)} normal counting questions with a number in the reference. Count = first number in the answer. "
            "The mean error is sensitive to long answers whose first number is not the count (e.g. \"x=250\").",
            ""] + table(["Run", "Exact", "Too high", "Too low", "Mean abs. error"], rows) + [""]


def n_calls(rec) -> int:
    return sum("observation" in s for s in rec.get("trace") or [])


def tool_section(pairs, qs, facts) -> list[str]:
    out = ["## 6. Tool use", ""]
    rows = []
    for name, p in pairs.items():
        recs = [p["agent"]["responses"][q] for q in qs]
        tools = Counter(s.get("tool") or "?" for r in recs for s in r["trace"] if "observation" in s)
        errs = sum(bool(s.get("error")) for r in recs for s in r["trace"] if "observation" in s)
        total = sum(tools.values())
        rows.append([name, pct(sum(n_calls(r) > 0 for r in recs), len(recs)), f"{total / len(recs):.2f}",
                     *(str(tools.get(t, 0)) for t in ("zoom", "grid", "detect", "ocr", "depth")), pct(errs, total),
                     str(sum(r.get("finish_reason") == "no_answer_tag" for r in recs))])
    out += table(["Model", "Used a tool", "Calls / question", "zoom", "grid", "detect", "ocr", "depth",
                  "Failed calls", "No answer tag"], rows) + [""]

    # which tool is used most per task type (share of questions where the tool was called at least once)
    out += ["### Share of questions where each tool was called, by task (all models together)", ""]
    rows = []
    for t in TASKS:
        tq = [q for q in qs if facts[q]["sample"].category == t]
        used = Counter()
        for p in pairs.values():
            for q in tq:
                used.update({s.get("tool") for s in p["agent"]["responses"][q]["trace"] if "observation" in s})
        n = len(tq) * len(pairs)
        rows.append([t] + [f"{100 * used[k] / n:.0f}%" for k in ("zoom", "grid", "detect", "ocr", "depth")])
    out += table(["Task", "zoom", "grid", "detect", "ocr", "depth"], rows) + [""]

    out += ["### Accuracy by number of tool calls", "",
            "Agent accuracy, and the base model's accuracy on the same questions. Harder questions tend to get more "
            "calls, so this shows where the agent helps, not that more calls cause the gain.", ""]
    buckets = [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 99)]
    rows = []
    for name, p in pairs.items():
        row = [name]
        for _, lo, hi in buckets:
            b = [q for q in qs if lo <= n_calls(p["agent"]["responses"][q]) <= hi]
            row.append(f"{cell(p, b)}, n={len(b)}" if b else "–")
        rows.append(row)
    return out + table(["Model"] + [f"{b} calls" for b, _, _ in buckets], rows) + [""]


def cost_section(pairs, qs) -> list[str]:
    rows = []
    for name, p in pairs.items():
        p_in, p_out = p["price"]
        row = [name]
        for which in ("base", "agent"):
            u = [p[which]["responses"][q].get("usage") or {} for q in qs]
            tin = sum(x.get("input_tokens") or 0 for x in u)
            tout = sum((x.get("output_tokens") or 0) for x in u)
            row += [f"{tin / len(qs) / 1000:.1f}k", f"${(tin * p_in + tout * p_out) / 1e6:.2f}"]
        rows.append(row)
    return ["## 7. Tokens and cost", "",
            "Input tokens per question and list-price cost of the model's answers for the whole run (judge not "
            "included). OpenRouter's actual bill can be lower because of prompt caching.", ""] + table(
        ["Model", "Base input / q", "Base cost", "Agent input / q", "Agent cost"], rows) + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--tuning-ids", type=Path, default=Path("analysis/tuning_ids.txt"))
    ap.add_argument("--out", type=Path, default=Path("runs/vistaqa_agent_analysis"))
    args = ap.parse_args()

    pairs = load_pairs(args.runs)
    facts = add_question_facts(get_dataset("vistaqa"))
    qs = common(pairs)
    tuning = set(args.tuning_ids.read_text().split()) if args.tuning_ids.exists() else set()
    md = ["# VistaQA: tool agent (v2 instructions) vs base VLM", ""]
    md += overall_section(pairs, qs, tuning)
    md += task_section(pairs, qs, facts)
    md += domain_section(pairs, qs, facts)
    md += object_size_section(pairs, qs, facts)
    md += counting_section(pairs, qs, facts)
    md += tool_section(pairs, qs, facts)
    md += cost_section(pairs, qs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved {args.out}/analysis.md")


if __name__ == "__main__":
    main()
