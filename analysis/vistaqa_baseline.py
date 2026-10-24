"""Extra analyses of the VistaQA base-VLM runs (no API calls).

    python analysis/vistaqa_baseline.py runs/vistaqa__openrouter__* --out runs/vistaqa_analysis

1. Object size: accuracy vs how big the answer object is (from the dataset's mask areas).
2. Counting errors: how far off the counts are (over / under, by how much).
3. Picture size: accuracy vs image resolution and image tokens, within domains.
4. Judge agreement (if a second judge's file is given with --second-judge).
5. Judges vs hand labels (analysis/judge_handcheck.csv: 100 random answers graded by hand).
Writes analysis.md (+ the per-question table analysis_rows.csv) to --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from vlmbench.datasets import get_dataset
from vlmbench.runner import load_run_config, read_records

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                "fifteen sixteen seventeen eighteen nineteen twenty").split()
_NUM = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r"|no|none)\b", re.I)


def first_number(text: str | None) -> int | None:
    """First count mentioned in an answer ("no"/"none" = 0)."""
    m = _NUM.search(text or "")
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else 0 if w in ("no", "none") else NUMBER_WORDS.index(w)


def pct(c: int, n: int) -> str:
    return f"{100 * c / n:.1f}% ({c}/{n})" if n else "–"


def load_runs(run_dirs: list[Path], second_judge: str | None):
    runs = {}
    for d in run_dirs:
        cfg = load_run_config(d)
        name = cfg["model"].split("/")[-1].split(":")[0]
        runs[name] = {
            "responses": read_records(d / "responses.jsonl"),
            "judge": read_records(d / "judgments.jsonl"),
            "judge2": read_records(d / second_judge) if second_judge else {},
        }
    return runs


def add_question_facts(ds) -> dict[str, dict]:
    """Per question: image size, answer-object sizes (from mask areas), reference count."""
    facts = {}
    for s in ds.load():
        raw = json.loads((ds.data_dir / "raw" / "VistaQA" / f"{s.question_id}.json").read_text())
        w, h = s.meta["width"], s.meta["height"]
        areas = sorted(a["area"] for a in raw["annotations"])
        facts[s.question_id] = {
            "sample": s,
            "pixels": w * h,
            # typical answer object: median object, as a share of the image and as a side length in px
            "obj_frac": statistics.median(areas) / (w * h) if areas else None,
            "obj_side_px": math.sqrt(statistics.median(areas)) if areas else None,
            "gt_count": first_number(s.answer) if s.category == "counting" else None,
        }
    return facts


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def quantile_bins(values: list[float], k: int) -> list[float]:
    vs = sorted(values)
    return [vs[min(len(vs) - 1, int(len(vs) * i / k))] for i in range(1, k)]


def bin_of(x: float, edges: list[float]) -> int:
    return sum(x >= e for e in edges)


def fmt_frac(x: float) -> str:
    return f"{100 * x:.2g}%"


def object_size_section(runs, facts) -> list[str]:
    real = [q for q, f in facts.items() if f["obj_frac"] is not None]
    out = ["## 1. Accuracy vs size of the answer object", "",
           "Normal questions only (trick questions have no object). Object size = the median answer object's mask area. "
           "Each group holds about a fifth of the questions.", ""]
    for key, label, fmt in [("obj_frac", "share of the image", fmt_frac),
                            ("obj_side_px", "width in pixels (square root of its area)", lambda x: f"{x:.0f} px")]:
        edges = quantile_bins([facts[q][key] for q in real], 5)
        bounds = [None] + edges + [None]
        groups = defaultdict(list)
        for q in real:
            groups[bin_of(facts[q][key], edges)].append(q)
        rows = []
        for b in range(5):
            lo, hi = bounds[b], bounds[b + 1]
            name = (f"< {fmt(hi)}" if lo is None else f">= {fmt(lo)}" if hi is None else f"{fmt(lo)} – {fmt(hi)}")
            qs = groups[b]
            rows.append([name] + [pct(sum(r["judge"][q]["correct"] for q in qs), len(qs)) for r in runs.values()])
        out += [f"### By object size as {label}", ""] + table(["Object size"] + list(runs), rows) + [""]
    return out


def counting_section(runs, facts) -> list[str]:
    qs = [q for q, f in facts.items()
          if f["sample"].category == "counting" and f["sample"].meta["hallucination"] == 0 and f["gt_count"] is not None]
    out = ["## 2. Counting errors", "",
           f"Normal counting questions with a number in the reference answer: {len(qs)}. "
           "The predicted count is the first number in the model's answer.", ""]
    summary, direction = [], []
    buckets = [("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7+", 7, 10**6)]
    for name, r in runs.items():
        parsed = {q: first_number(r["responses"][q]["response"]) for q in qs}
        got = [q for q in qs if parsed[q] is not None]
        diffs = {q: parsed[q] - facts[q]["gt_count"] for q in got}
        exact = sum(d == 0 for d in diffs.values())
        over = sum(d > 0 for d in diffs.values())
        under = sum(d < 0 for d in diffs.values())
        off1 = sum(abs(d) == 1 for d in diffs.values())
        zero = sum(parsed[q] == 0 for q in got)
        mae = statistics.mean(abs(d) for d in diffs.values()) if diffs else float("nan")
        rel = statistics.median(abs(d) / facts[q]["gt_count"] for q, d in diffs.items() if d != 0) if over + under else 0
        agree = sum((diffs[q] == 0) == bool(r["judge"][q]["correct"]) for q in got)
        summary.append([name, pct(len(got), len(qs)), pct(exact, len(got)), pct(over, len(got)), pct(under, len(got)),
                        pct(off1, over + under), pct(zero, len(got)), f"{mae:.1f}", f"{100 * rel:.0f}%",
                        pct(agree, len(got))])
        row = [name]
        for _, lo, hi in buckets:
            b = [q for q in got if lo <= facts[q]["gt_count"] <= hi]
            row.append(f"{sum(diffs[q] == 0 for q in b)} / {sum(diffs[q] > 0 for q in b)} / {sum(diffs[q] < 0 for q in b)}")
        direction.append(row)
    out += table(["Model", "Gave a number", "Exact", "Too high", "Too low", "Off by 1 (of wrong)",
                  "Said zero / none", "Mean abs. error", "Median error when wrong", "Exact-match agrees with judge"],
                 summary)
    n_b = {name: sum(1 for q in qs if lo <= facts[q]["gt_count"] <= hi) for name, lo, hi in buckets}
    out += ["", "### Exact / too high / too low, by the true count", ""]
    out += table(["Model"] + [f"true count {b} (n={n_b[b]})" for b, _, _ in buckets], direction) + [""]
    return out


def picture_size_section(runs, facts) -> list[str]:
    out = ["## 3. Accuracy vs picture size", "",
           "All 1,157 questions. Robotics images are all 320×180, so they are reported apart.", ""]
    mp_bins = [("< 0.1 MP (robotics)", 0, 1e5), ("0.1 – 1 MP", 1e5, 1e6), ("1 – 2 MP", 1e6, 2e6),
               ("2 – 3.5 MP", 2e6, 3.5e6), (">= 3.5 MP", 3.5e6, 1e12)]
    rows, tok_rows = [], []
    for label, lo, hi in mp_bins:
        qs = [q for q, f in facts.items() if lo <= f["pixels"] < hi]
        doms = defaultdict(int)
        for q in qs:
            doms[facts[q]["sample"].meta["domain"]] += 1
        top = ", ".join(f"{d} {n}" for d, n in sorted(doms.items(), key=lambda x: -x[1])[:3])
        rows.append([label, str(len(qs)), top] + [pct(sum(r["judge"][q]["correct"] for q in qs), len(qs)) for r in runs.values()])
        tok_rows.append([label] + [str(int(statistics.median(r["responses"][q]["usage"]["input_tokens"] for q in qs)))
                                   if qs else "–" for r in runs.values()])
    out += table(["Image size", "Questions", "Main domains"] + list(runs), rows)
    out += ["", "### Median input tokens per question (mostly the image)", ""]
    out += table(["Image size"] + list(runs), tok_rows)

    out += ["", "### Within each domain: smaller half vs larger half of the images", "",
            "Splitting at the domain's median image size removes the domain effect "
            "(robotics left out: all its images are the same size).", ""]
    rows = []
    for dom in ["av", "indoor", "math", "outdoor", "science"]:
        qs = sorted((q for q, f in facts.items() if f["sample"].meta["domain"] == dom), key=lambda q: facts[q]["pixels"])
        small, large = qs[:len(qs) // 2], qs[len(qs) // 2:]
        if facts[small[-1]]["pixels"] == facts[large[0]]["pixels"] and len({facts[q]["pixels"] for q in qs}) < 3:
            continue
        cells = []
        for r in runs.values():
            a = 100 * sum(r["judge"][q]["correct"] for q in small) / len(small)
            b = 100 * sum(r["judge"][q]["correct"] for q in large) / len(large)
            cells.append(f"{a:.0f}% → {b:.0f}%")
        med = facts[large[0]]["pixels"] / 1e6
        rows.append([dom, f"{med:.1f} MP", f"{len(small)}/{len(large)}"] + cells)
    out += table(["Domain", "Split at", "n small/large"] + [f"{m} (small → large)" for m in runs], rows) + [""]
    return out


def judge_agreement_section(runs, second_judge: str) -> list[str]:
    out = ["## 4. Agreement between the two judges", ""]
    rows, disagree_examples = [], []
    for name, r in runs.items():
        qs = [q for q in r["judge"] if q in r["judge2"]
              and r["judge"][q]["correct"] is not None and r["judge2"][q]["correct"] is not None]
        a = [r["judge"][q]["correct"] for q in qs]
        b = [r["judge2"][q]["correct"] for q in qs]
        n = len(qs)
        agree = sum(x == y for x, y in zip(a, b))
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree / n - pe) / (1 - pe) if pe < 1 else 1.0
        only1 = sum(x and not y for x, y in zip(a, b))
        only2 = sum(y and not x for x, y in zip(a, b))
        rows.append([name, str(n), f"{100 * pa:.1f}%", f"{100 * pb:.1f}%", f"{100 * agree / n:.1f}%", f"{kappa:.2f}",
                     str(only1), str(only2)])
    j1 = next(iter(runs.values()))["judge"]
    j2 = next(iter(runs.values()))["judge2"]
    name1 = next(iter(j1.values()))["judge_model"]
    name2 = next(iter(j2.values()))["judge_model"]
    out += [f"Judge 1: `{name1}`. Judge 2: `{name2}`.", ""]
    out += table(["Model", "Answers", "Accuracy (judge 1)", "Accuracy (judge 2)", "Same verdict", "Cohen's kappa",
                  "Only judge 1 says correct", "Only judge 2 says correct"], rows) + [""]
    return out


def handcheck_section(path: Path) -> list[str]:
    rows = [r for r in csv.DictReader(path.open()) if r["my_label"] != ""]
    n = len(rows)
    out = ["## 5. Judges vs hand labels", "",
           f"{n} random answers (about a third per model) graded by hand against the reference answer with the "
           f"official rules ({path}; one more was too ambiguous to label).", ""]
    trows = []
    for col, label in [("judge_qwen72b", "Qwen2.5-72B (judge 1)"), ("judge_haiku", "Claude Haiku 5.5 (judge 2)")]:
        agree = sum(r[col] == r["my_label"] for r in rows)
        lenient = [r["n"] for r in rows if r[col] == "1" and r["my_label"] == "0"]
        strict = [r["n"] for r in rows if r[col] == "0" and r["my_label"] == "1"]
        trows.append([label, pct(agree, n), f"{len(lenient)} (#{', #'.join(lenient)})" if lenient else "0",
                      f"{len(strict)} (#{', #'.join(strict)})" if strict else "0",
                      str(sum(r[col] == "1" for r in rows))])
    trows.append(["Hand labels", "–", "–", "–", str(sum(r["my_label"] == "1" for r in rows))])
    out += table(["Grader", "Agrees with hand labels", "Says correct when wrong", "Says wrong when correct",
                  f"Correct answers in the sample (of {n})"], trows) + [""]
    return out


def write_rows(path: Path, runs, facts) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "task", "domain", "trick", "image_pixels", "obj_frac", "obj_side_px", "gt_count",
                    "model", "response", "correct_judge1", "correct_judge2", "input_tokens"])
        for q, fa in sorted(facts.items(), key=lambda x: int(x[0])):
            s = fa["sample"]
            for name, r in runs.items():
                resp = r["responses"].get(q)
                if resp is None:
                    continue
                w.writerow([q, s.category, s.meta["domain"], s.meta["hallucination"], fa["pixels"],
                            fa["obj_frac"], fa["obj_side_px"], fa["gt_count"], name, resp["response"],
                            r["judge"].get(q, {}).get("correct"), r["judge2"].get(q, {}).get("correct"),
                            (resp.get("usage") or {}).get("input_tokens")])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--second-judge", help="file name of a second judge's judgments inside each run dir")
    p.add_argument("--handcheck", type=Path, default=Path("analysis/judge_handcheck.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/vistaqa_analysis"))
    args = p.parse_args()

    runs = load_runs(args.run_dirs, args.second_judge)
    facts = add_question_facts(get_dataset("vistaqa"))
    md = ["# VistaQA base-VLM: extra analyses", ""]
    md += object_size_section(runs, facts)
    md += counting_section(runs, facts)
    md += picture_size_section(runs, facts)
    if args.second_judge:
        md += judge_agreement_section(runs, args.second_judge)
    if args.handcheck.exists():
        md += handcheck_section(args.handcheck)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.md").write_text("\n".join(md) + "\n")
    write_rows(args.out / "analysis_rows.csv", runs, facts)
    print("\n".join(md))
    print(f"\nSaved {args.out}/analysis.md and analysis_rows.csv")


if __name__ == "__main__":
    main()
