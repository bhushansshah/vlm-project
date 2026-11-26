"""Build the SFT arms' training files from data/train_v1/sft and the trace runs (plan, section 4).

    python training/make_sft_arms.py --out data/sft_arms

- a2_teacher.jsonl: Qwen3.5-27B thinking traces, correct answers only, finished (not cut off), thinking text at
  most --max-trace-tokens (estimated as words x 1.3 when the usage record has no reasoning count).
- a1_bare.jsonl: the SAME questions as a2, target = the gold letter, thinking off (empty think block).
  Holding the questions fixed means a1 and a2 differ only in the target.
- a3_self.jsonl: Qwen3.5-4B's own thinking traces under the same filter (its own set of questions).
Each line: {"question_id", "image", "prompt", "target", "thinking", "category", "source"}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vlmbench.datasets import get_dataset
from vlmbench.runner import read_records


def traces(run: Path, samples: dict, dataset, max_tokens: int) -> dict[str, dict]:
    keep = {}
    for qid, r in read_records(run / "responses.jsonl").items():
        s = samples.get(qid)
        thinking = (r.get("reasoning") or "").strip()
        answer = (r.get("response") or "").strip()
        if s is None or not thinking or not answer or not dataset.grade(s, r):
            continue
        n = (r.get("usage") or {}).get("reasoning_tokens") or int(len(thinking.split()) * 1.3)
        if n > max_tokens or "ground truth" in thinking.lower():
            continue
        keep[qid] = {"target": f"{thinking}\n</think>\n\n{answer}", "n_thinking_tokens": n}
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/train_v1/sft"))
    ap.add_argument("--teacher-run", type=Path, default=Path("runs/train_v1__teacher-27b-think__sft"))
    ap.add_argument("--self-run", type=Path, default=Path("runs/train_v1__self-4b-think__sft"))
    ap.add_argument("--max-trace-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("data/sft_arms"))
    ap.add_argument("--arms", nargs="+", default=["a1_bare", "a2_teacher", "a3_self"],
                    help="which files to write (build an arm only once its trace run has finished)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset("tbstyle", args.data_dir)
    samples = {s.question_id: s for s in ds.load_all()}

    def row(s, target, thinking):
        return {"question_id": s.question_id, "image": s.image_path, "prompt": s.meta["default_prompt"],
                "target": target, "thinking": thinking, "category": s.category}

    report = {"sft_pool": len(samples)}
    arms = {}
    if (args.teacher_run / "responses.jsonl").exists():
        t = traces(args.teacher_run, samples, ds, args.max_trace_tokens)
        arms["a2_teacher"] = [row(samples[q], v["target"], True) for q, v in t.items()]
        arms["a1_bare"] = [row(samples[q], samples[q].answer, False) for q in t]
        report["teacher_answered"] = len(read_records(args.teacher_run / "responses.jsonl"))
    if (args.self_run / "responses.jsonl").exists():
        t = traces(args.self_run, samples, ds, args.max_trace_tokens)
        arms["a3_self"] = [row(samples[q], v["target"], True) for q, v in t.items()]
        report["self_answered"] = len(read_records(args.self_run / "responses.jsonl"))
    for name, rows in arms.items():
        if name not in args.arms:
            continue
        with (args.out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        report[name] = {"n": len(rows), "per_category": dict(Counter(r["category"] for r in rows))}
    (args.out / f"report_{'_'.join(sorted(args.arms))}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
