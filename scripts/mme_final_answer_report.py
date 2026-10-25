"""Batch report restricted to responses that produced a final answer.

Truncated (finish_reason=length), skipped, and stop-without-answer examples are
tracked separately and excluded from the accuracy denominators.
Usage: python scripts/mme_final_answer_report.py [run_dir]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/mme_gemma4_reasoning_paid_32k_batch1")
out = run / "final_answer_report"
out.mkdir(exist_ok=True)
rows = json.loads((run / "per_example.json").read_text())


def category(r):
    if r["status"] == "skipped":
        return "skipped"
    if r.get("finish_reason") != "stop":
        return "truncated"
    return "answered" if r.get("explicit_prediction") else "no_final_answer"


def pct(a, b):
    return round(100 * a / b, 2) if b else None


per_example = []
for r in rows:
    c = category(r)
    pred = r.get("explicit_prediction") or ""
    per_example.append({
        "sample_id": r["sample_id"], "question_id": r["Question_id"], "domain": r["Subtask"],
        "type": r["Task"], "category": r["Category"], "task": "/".join(r["task_key"]),
        "outcome": c, "ground_truth": r["Ground truth"], "final_answer": pred if c == "answered" else "",
        "correct": (pred == r["Ground truth"]) if c == "answered" else "",
        "official_prediction": r.get("prediction", ""), "official_correct": r.get("correct", ""),
        "finish_reason": r.get("finish_reason"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
        "latency_s": r.get("latency_s"),
    })


def aggregate(key):
    g = defaultdict(lambda: defaultdict(int))
    for e in per_example:
        a = g[e[key]]
        a["total"] += 1
        a[e["outcome"]] += 1
        if e["outcome"] == "answered":
            a["correct"] += e["correct"]
            a["official_correct"] += bool(e["official_correct"])
    res = []
    for k, a in g.items():
        res.append({key: k, "total": a["total"], "answered": a["answered"], "correct": a["correct"],
                    "accuracy_answered_pct": pct(a["correct"], a["answered"]),
                    "official_accuracy_same_subset_pct": pct(a["official_correct"], a["answered"]),
                    "truncated": a["truncated"], "no_final_answer": a["no_final_answer"], "skipped": a["skipped"],
                    "truncated_pct": pct(a["truncated"], a["total"]),
                    "accuracy_all_selected_pct": pct(a["correct"], a["total"])})
    return sorted(res, key=lambda x: -(x["accuracy_answered_pct"] or -1))


def write_csv(name, data):
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


domains, tasks = aggregate("domain"), aggregate("task")
write_csv("per_example.csv", per_example)
write_csv("per_domain.csv", domains)
write_csv("per_task.csv", tasks)
(out / "truncated_examples.json").write_text(json.dumps(
    [e for e in per_example if e["outcome"] != "answered"], indent=1))

ans = [e for e in per_example if e["outcome"] == "answered"]
cnt = lambda o: sum(e["outcome"] == o for e in per_example)
macro = sum(t["accuracy_answered_pct"] for t in tasks if t["answered"]) / sum(1 for t in tasks if t["answered"])
summary = {
    "total": len(per_example), "answered": len(ans), "truncated": cnt("truncated"),
    "no_final_answer": cnt("no_final_answer"), "skipped": cnt("skipped"),
    "correct": sum(e["correct"] for e in ans),
    "accuracy_answered_pct": pct(sum(e["correct"] for e in ans), len(ans)),
    "macro_task_accuracy_answered_pct": round(macro, 2),
    "accuracy_all_selected_pct": pct(sum(e["correct"] for e in ans), len(per_example)),
    "official_accuracy_same_subset_pct": pct(sum(bool(e["official_correct"]) for e in ans), len(ans)),
}
(out / "summary.json").write_text(json.dumps(summary, indent=1))

md = [f"# Batch 1 — accuracy on responses with a final answer\n",
      f"Answered **{summary['answered']}** / {summary['total']} · truncated {summary['truncated']} · "
      f"no final answer {summary['no_final_answer']} · skipped {summary['skipped']}\n",
      f"**Accuracy (answered only): {summary['accuracy_answered_pct']}%** ({summary['correct']}/{summary['answered']}) · "
      f"macro task {summary['macro_task_accuracy_answered_pct']}% · "
      f"if truncated/skipped counted wrong: {summary['accuracy_all_selected_pct']}%\n",
      "Final answers use the explicit extractor (last stated answer letter); the official parser is shown only for reference.\n",
      "## By domain\n", "| Domain | Answered/Total | Accuracy | Truncated | Skipped |", "|---|---:|---:|---:|---:|"]
md += [f"| {d['domain']} | {d['answered']}/{d['total']} | {d['accuracy_answered_pct']}% | {d['truncated']} | {d['skipped']} |" for d in domains]
md += ["\n## By task\n", "| Task | Answered/Total | Accuracy | Truncated |", "|---|---:|---:|---:|"]
md += [f"| {t['task']} | {t['answered']}/{t['total']} | {t['accuracy_answered_pct']}% | {t['truncated']} |" for t in tasks]
(out / "report.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
