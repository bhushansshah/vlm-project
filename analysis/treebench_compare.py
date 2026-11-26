"""Compare two TreeBench runs question by question (e.g. a student and a candidate teacher model).
No API calls; works on runs that are still going (only questions both runs answered are compared).

    python analysis/treebench_compare.py --out runs/treebench_compare_4b_vs_27b \
        --a treebench__vllm__Qwen_Qwen3.5-4B__default__think-h100 \
        --b treebench__vllm__Qwen_Qwen3.5-27B__default__think-h100

--a / --b take one or more run folders each (merged, as in treebench_failures.py). Writes compare.md and
compare.json to --out (or prints the markdown):
1. Accuracy overall, per group and per category for both runs, on the same questions.
2. Paired outcomes (both right, only A, only B, neither) with an exact McNemar test.
3. The perspective left/right check and the grounding check (treebench_failures.py) for each run.
4. Thinking length and latency, right vs wrong.
5. Distillation yield: questions where B is right and A is wrong (B's trace teaches A something new),
   per category, plus a few examples with the end of both thinking texts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from treebench_failures import analyse, merge_runs  # same folder

from vlmbench.datasets import get_dataset


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}" if n else "–"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def compare(samples, recs_a, recs_b, dataset, n_examples: int) -> dict:
    common = sorted(set(recs_a) & set(recs_b))
    a = analyse(samples, {q: recs_a[q] for q in common}, dataset, 0)
    b = analyse(samples, {q: recs_b[q] for q in common}, dataset, 0)
    ra = {r["qid"]: r for r in a["rows"]}
    rb = {r["qid"]: r for r in b["rows"]}

    def group_of(cat: str) -> str:
        return cat.split("/")[0]

    table = defaultdict(lambda: {"n": 0, "a": 0, "b": 0, "both": 0, "only_a": 0, "only_b": 0, "neither": 0})
    for q in common:
        cat = ra[q]["category"]
        for key in ("All", group_of(cat), cat):
            t = table[key]
            t["n"] += 1
            t["a"] += ra[q]["correct"]
            t["b"] += rb[q]["correct"]
            t["both"] += ra[q]["correct"] and rb[q]["correct"]
            t["only_a"] += ra[q]["correct"] and not rb[q]["correct"]
            t["only_b"] += rb[q]["correct"] and not ra[q]["correct"]
            t["neither"] += not ra[q]["correct"] and not rb[q]["correct"]
    for t in table.values():
        t["p_mcnemar"] = round(mcnemar_exact(t["only_a"], t["only_b"]), 4)

    def lengths(rows: dict, recs: dict) -> dict:
        out = {}
        for label, want in (("correct", True), ("wrong", False)):
            qs = [q for q in common if rows[q]["correct"] == want]
            out[label] = {
                "n": len(qs),
                "median_output_tokens": median((recs[q].get("usage") or {}).get("output_tokens") for q in qs),
                "median_reasoning_tokens": median(rows[q]["reasoning_tokens"] for q in qs),
                "median_latency_s": median(recs[q].get("latency_s") for q in qs),
            }
        return out

    only_b = [q for q in common if rb[q]["correct"] and not ra[q]["correct"]]
    examples = []
    for q in only_b[:n_examples]:
        examples.append({"qid": q, "category": ra[q]["category"], "question": ra[q]["question"],
                         "answer": ra[q]["answer"], "pred_a": ra[q]["pred"], "pred_b": rb[q]["pred"],
                         "thinking_tail_a": ra[q]["thinking_tail"], "thinking_tail_b": rb[q]["thinking_tail"]})
    return {
        "n_common": len(common), "only_in_a": len(set(recs_a) - set(recs_b)), "only_in_b": len(set(recs_b) - set(recs_a)),
        "table": dict(table),
        "outcomes": {"a": a["outcomes"], "b": b["outcomes"]},
        "perspective": {"a": a["perspective"], "b": b["perspective"]},
        "grounding": {"a": a["groups"], "b": b["groups"]},
        "lengths": {"a": lengths(ra, recs_a), "b": lengths(rb, recs_b)},
        "distill_yield": {"only_b_right": len(only_b),
                          "per_category": dict(Counter(ra[q]["category"] for q in only_b).most_common())},
        "examples": examples,
    }


def markdown(c: dict, name_a: str, name_b: str) -> str:
    L = [f"# TreeBench: A vs B on {c['n_common']} shared questions", "",
         f"- **A** = `{name_a}`", f"- **B** = `{name_b}`"]
    if c["only_in_a"] or c["only_in_b"]:
        L.append(f"- Not compared (answered by one run only): {c['only_in_a']} in A, {c['only_in_b']} in B")
    L += ["", "## Accuracy and paired outcomes", "",
          "| Questions | n | A | B | both right | only A | only B | neither | McNemar p |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    keys = ["All", "Perception", "Reasoning"] + sorted(k for k in c["table"] if "/" in k)
    for k in keys:
        if k not in c["table"]:
            continue
        t = c["table"][k]
        L.append(f"| {k} | {t['n']} | {pct(t['a'], t['n'])}% | {pct(t['b'], t['n'])}% | {t['both']} | {t['only_a']} "
                 f"| {t['only_b']} | {t['neither']} | {t['p_mcnemar']} |")
    L += ["", "## How the misses break down", "", "| Outcome | A | B |", "| --- | --- | --- |"]
    for o in sorted(set(c["outcomes"]["a"]) | set(c["outcomes"]["b"])):
        L.append(f"| {o} | {c['outcomes']['a'].get(o, 0)} | {c['outcomes']['b'].get(o, 0)} |")
    L += ["", "## Perspective (four-direction options): left/right scored separately", "",
          "| | A | B |", "| --- | --- | --- |"]
    pa, pb = c["perspective"]["a"], c["perspective"]["b"]
    for label, key in [("questions", "n"), ("whole answer right", "accuracy"), ("front/back right", "front_back_ok"),
                       ("left/right right", "left_right_ok"), ("right if left/right mirrored", "if_left_right_mirrored")]:
        L.append(f"| {label} | {pa[key]} ({pct(pa[key], pa['n'])}%) | {pb[key]} ({pct(pb[key], pb['n'])}%) |"
                 if key != "n" else f"| {label} | {pa['n']} | {pb['n']} |")
    L += ["", "## Thinking length and latency (medians)", "",
          "| Run | answers | n | output tokens | thinking tokens | latency (s) |", "| --- | --- | --- | --- | --- | --- |"]
    for run in ("a", "b"):
        for label in ("correct", "wrong"):
            s = c["lengths"][run][label]
            L.append(f"| {run.upper()} | {label} | {s['n']} | {s['median_output_tokens']} | {s['median_reasoning_tokens']} "
                     f"| {None if s['median_latency_s'] is None else round(s['median_latency_s'])} |")
    L += ["", "## Grounding: boxes in the thinking that match both target boxes (IoU ≥ 0.3)", "",
          "| Group | A: cites boxes | A: targets found | B: cites boxes | B: targets found |", "| --- | --- | --- | --- | --- |"]
    for g, sa in c["grounding"]["a"].items():
        sb = c["grounding"]["b"][g]
        L.append(f"| {g} | {sa['cites_boxes']}/{sa['n']} | {sa['targets_found']}/{sa['scored']} "
                 f"| {sb['cites_boxes']}/{sb['n']} | {sb['targets_found']}/{sb['scored']} |")
    y = c["distill_yield"]
    L += ["", f"## Distillation yield: {y['only_b_right']} questions where B is right and A is wrong", ""]
    L += [f"- {cat}: {n}" for cat, n in y["per_category"].items()]
    for e in c["examples"]:
        L += ["", f"### {e['qid']} ({e['category']}): gold {e['answer']}, A {e['pred_a']}, B {e['pred_b']}", "",
              f"> {e['question'][:400]}", "", "**A, end of thinking:**", "", "```", e["thinking_tail_a"].strip(), "```",
              "", "**B, end of thinking:**", "", "```", e["thinking_tail_b"].strip(), "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, help="run folder(s) of model A (e.g. the student)")
    ap.add_argument("--b", nargs="+", required=True, help="run folder(s) of model B (e.g. the teacher)")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, help="folder for compare.md and compare.json (default: print markdown)")
    ap.add_argument("--examples", type=int, default=4, help="B-right / A-wrong examples to show")
    args = ap.parse_args()

    dataset = get_dataset("treebench")
    samples = {s.question_id: s for s in dataset.load_all()}
    recs_a = merge_runs([args.runs_dir / r for r in args.a])
    recs_b = merge_runs([args.runs_dir / r for r in args.b])
    c = compare(samples, recs_a, recs_b, dataset, args.examples)
    md = markdown(c, " + ".join(args.a), " + ".join(args.b))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "compare.md").write_text(md)
        (args.out / "compare.json").write_text(json.dumps(c, indent=1, ensure_ascii=False))
        print(f"wrote {args.out / 'compare.md'} ({c['n_common']} shared questions)")
    else:
        print(md)


if __name__ == "__main__":
    main()
