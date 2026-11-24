"""Why does a no-tools VLM fail on TreeBench? Failure analysis from saved answers and thinking text
(no API calls).

    python analysis/treebench_failures.py --out runs/treebench_qwen35_4b_failures \
        treebench__vllm__Qwen_Qwen3.5-4B__default__think-fp16 \
        treebench__vllm__Qwen_Qwen3.5-4B__default__think-fp16-part2

Several run folders (under --runs-dir) are merged into one set of answers. That covers a run split
across jobs; for a question in more than one, the latest record wins. Works on a run that is still
going. Writes analysis.md and analysis.json to --out (or prints the markdown if --out is not given):
1. Accuracy by category, and how the misses break down (wrong letter, no letter, cut off, error).
2. Perspective questions with the four-direction options: front/back and left/right correctness
   scored separately, and the accuracy if every left/right were mirrored. A left/right rate well
   under 50% means the model answers in the camera's frame instead of the reference person's.
3. Grounding: boxes the model writes in its thinking (Qwen-style [x1, y1, x2, y2] on a 0-1000
   scale) compared with TreeBench's target boxes. If the targets were found but the answer is
   wrong, the error is in the reasoning, not in locating the objects.
4. Thinking length and hesitation ("wait" count), right vs wrong answers.
5. A few wrong answers per category, with the end of the thinking, to read by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from vlmbench.datasets import get_dataset
from vlmbench.datasets.choice import extract_choice_letter
from vlmbench.runner import read_records

PT = "Reasoning/Perspective Transform"
DIRECTIONS = {"Front left": ("F", "L"), "Front right": ("F", "R"), "Left rear": ("B", "L"), "Right rear": ("B", "R")}
BOX = re.compile(r"\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]")


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}" if n else "–"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_runs(run_dirs: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for d in run_dirs:
        for qid, r in read_records(d / "responses.jsonl").items():
            if qid not in merged or (r.get("timestamp") or "") >= (merged[qid].get("timestamp") or ""):
                merged[qid] = r
    return merged


def outcome(rec: dict, letter: str | None, correct: bool) -> str:
    if correct:
        return "correct"
    if rec.get("error"):
        return "error"
    if rec.get("finish_reason") == "length":
        return "cut off"
    if letter is None:
        return "no letter"
    return "wrong letter"


def analyse(samples: dict, records: dict, dataset, n_examples: int) -> dict:
    rows = []
    for qid, rec in sorted(records.items()):
        s = samples[qid]
        opts = s.meta["options"]
        letter = extract_choice_letter(rec.get("response"), opts)
        ok = dataset.grade(s, rec)
        thinking = rec.get("reasoning") or ""
        W, H = Image.open(s.image_path).size
        gts = [[b[0] * 1000 / W, b[1] * 1000 / H, b[2] * 1000 / W, b[3] * 1000 / H] for b in s.meta["target_boxes"]]
        boxes = [list(map(int, m)) for m in BOX.findall(thinking)]
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1] and max(b) <= 1000]
        best = [max(iou(g, b) for b in boxes) for g in gts] if boxes and gts else None
        rows.append({
            "qid": qid, "category": s.category, "answer": s.answer, "pred": letter, "correct": ok,
            "outcome": outcome(rec, letter, ok), "options": list(opts),
            "reasoning_tokens": (rec.get("usage") or {}).get("reasoning_tokens"),
            "waits": len(re.findall(r"\bwait\b", thinking.lower())),
            "cites_boxes": bool(boxes),
            "targets_found": None if best is None else all(v >= 0.3 for v in best),
            "best_iou": None if best is None else sum(best) / len(best),
            "question": s.question, "thinking_tail": thinking[-700:],
        })

    # 1. accuracy and outcomes
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    categories = {c: {"n": len(g), "correct": sum(r["correct"] for r in g),
                      "outcomes": dict(Counter(r["outcome"] for r in g if not r["correct"]))}
                  for c, g in sorted(by_cat.items())}

    # 2. perspective: front/back vs left/right
    pt = [r for r in rows if r["category"] == PT and sorted(r["options"]) == sorted(DIRECTIONS) and r["pred"]]
    combos = Counter()
    for r in pt:
        g = DIRECTIONS[r["options"][ord(r["answer"]) - 65]]
        p = DIRECTIONS[r["options"][ord(r["pred"]) - 65]]
        combos[("front/back " + ("ok" if p[0] == g[0] else "flipped"), "left/right " + ("ok" if p[1] == g[1] else "flipped"))] += 1
    n_pt = len(pt)
    perspective = {
        "n": n_pt,
        "accuracy": sum(r["correct"] for r in pt),
        "front_back_ok": combos[("front/back ok", "left/right ok")] + combos[("front/back ok", "left/right flipped")],
        "left_right_ok": combos[("front/back ok", "left/right ok")] + combos[("front/back flipped", "left/right ok")],
        "if_left_right_mirrored": combos[("front/back ok", "left/right flipped")],
        "combos": {f"{a}, {b}": v for (a, b), v in combos.most_common()},
        "other_option_sets": sum(1 for r in rows if r["category"] == PT) - sum(
            1 for r in rows if r["category"] == PT and sorted(r["options"]) == sorted(DIRECTIONS)),
    }

    # 3. grounding and 4. thinking length
    def group_stats(g: list[dict]) -> dict:
        with_boxes = [r for r in g if r["targets_found"] is not None]
        toks = [r["reasoning_tokens"] for r in g if r["reasoning_tokens"]]
        return {
            "n": len(g),
            "cites_boxes": sum(r["cites_boxes"] for r in g),
            "scored": len(with_boxes),
            "targets_found": sum(r["targets_found"] for r in with_boxes),
            "mean_best_iou": round(statistics.mean(r["best_iou"] for r in with_boxes), 3) if with_boxes else None,
            "median_reasoning_tokens": statistics.median(toks) if toks else None,
            "mean_waits": round(statistics.mean(r["waits"] for r in g), 1) if g else None,
        }

    groups = {}
    for name, sel in [("perspective", lambda r: r["category"] == PT), ("other", lambda r: r["category"] != PT)]:
        for label, want in [("correct", True), ("wrong", False)]:
            groups[f"{name} / {label}"] = group_stats([r for r in rows if sel(r) and r["correct"] == want])

    examples = {c: [r for r in g if not r["correct"]][:n_examples] for c, g in sorted(by_cat.items())}
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)), "categories": categories,
            "perspective": perspective, "groups": groups, "examples": examples, "rows": rows}


def markdown(a: dict, run_names: list[str]) -> str:
    L = ["# TreeBench failure analysis", "", "Runs: " + ", ".join(f"`{n}`" for n in run_names), "",
         f"**{a['correct']}/{a['n']} correct ({pct(a['correct'], a['n'])}%)** of {a['n']} answered "
         f"(TreeBench has 405). Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(a["outcomes"].items())), "",
         "## Accuracy by category", "", "| Category | n | Acc % | Misses | Miss types |", "|---|---|---|---|---|"]
    for c, d in a["categories"].items():
        miss = d["n"] - d["correct"]
        L.append(f"| {c} | {d['n']} | {pct(d['correct'], d['n'])} | {miss} | "
                 + ", ".join(f"{k} {v}" for k, v in d["outcomes"].items()) + " |")
    p = a["perspective"]
    L += ["", "## Perspective questions: front/back vs left/right", "",
          f"On the {p['n']} answered perspective questions with the four-direction options "
          f"({p['other_option_sets']} others use other option sets):", "",
          f"- Accuracy: {p['accuracy']}/{p['n']} ({pct(p['accuracy'], p['n'])}%); chance is 25%",
          f"- Front/back right: {p['front_back_ok']}/{p['n']} ({pct(p['front_back_ok'], p['n'])}%)",
          f"- Left/right right: {p['left_right_ok']}/{p['n']} ({pct(p['left_right_ok'], p['n'])}%); chance is 50%, "
          "so much lower means the camera's left/right is being used",
          f"- Accuracy if every left/right were mirrored: {p['if_left_right_mirrored']}/{p['n']} "
          f"({pct(p['if_left_right_mirrored'], p['n'])}%)", "",
          "| Front/back, left/right | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in p["combos"].items()]
    L += ["", "## Grounding and thinking length", "",
          "Boxes the model cites in its thinking vs the target boxes (found = every target matched at IoU ≥ 0.3).", "",
          "| Group | n | Cite boxes | Targets found (of scored) | Mean best IoU | Median thinking tokens | Mean 'wait' |",
          "|---|---|---|---|---|---|---|"]
    for g, d in a["groups"].items():
        L.append(f"| {g} | {d['n']} | {d['cites_boxes']} | {d['targets_found']}/{d['scored']} | {d['mean_best_iou']} | "
                 f"{d['median_reasoning_tokens']} | {d['mean_waits']} |")
    L += ["", "## Wrong answers to read", ""]
    for c, ex in a["examples"].items():
        if not ex:
            continue
        L += [f"### {c}", ""]
        for r in ex:
            tail = r["thinking_tail"].strip().replace("\n", "\n> ")
            L += [f"**{r['qid']}** gold {r['answer']}, said {r['pred']}: {r['question']}", "", f"> …{tail}", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders under --runs-dir, merged into one set of answers")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, help="folder for analysis.md and analysis.json (default: print markdown)")
    ap.add_argument("--examples", type=int, default=3, help="wrong answers shown per category")
    args = ap.parse_args()

    dataset = get_dataset("treebench")
    samples = {s.question_id: s for s in dataset.load_all()}
    records = merge_runs([args.runs_dir / r for r in args.runs])
    a = analyse(samples, records, dataset, args.examples)
    md = markdown(a, args.runs)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "analysis.md").write_text(md)
        (args.out / "analysis.json").write_text(json.dumps(a, indent=1, ensure_ascii=False))
        print(f"wrote {args.out / 'analysis.md'} and analysis.json ({a['n']} questions)")
    else:
        print(md)


if __name__ == "__main__":
    main()
