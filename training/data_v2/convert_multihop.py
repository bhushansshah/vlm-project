"""MultihopSpatial (ECCV 2026, Apache-2.0) person-perspective questions, in TreeBench's prompt format.

Human-labelled perspective practice to sit beside the generated SA-1B questions: "From the perspective of the
man wearing red clothes, which object is behind him?" with four object options, 1 to 3 reasoning hops, on
COCO images (~640 px). Only the 2,475 training questions that take a person's perspective are kept; their
options are shuffled (seeded) so answer letters are balanced, the "provide the bounding box" request is
dropped, and the answer's box (xywh) becomes `target_boxes` [x1, y1, x2, y2].

    python training/data_v2/convert_multihop.py --raw data/train_v2/raw/multihopspatial \
        --out data/train_v2/extra/multihopspatial.jsonl --images data/train_v2/images/multihopspatial
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PT = "Reasoning/Perspective Transform"
LETTERS = "ABCD"
DIRECT = "Answer with the option's letter from the given choices directly."
PERSPECTIVE = re.compile(r"perspective|viewpoint|point of view", re.I)
CHOICE = re.compile(r"<choice>(.*?)</choice>", re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.images.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.raw / "parquet" / "train-00000-of-00001.parquet")
    kept, skipped = [], Counter()
    for r in table.to_pylist():
        q = r["question"]
        if not PERSPECTIVE.search(q):
            skipped["not a person-perspective question"] += 1
            continue
        m = CHOICE.search(q)
        if not m:
            skipped["no choices"] += 1
            continue
        opts = [o.strip().rstrip(",").strip() for o in re.split(r"\(([a-d])\)\s*", m.group(1))[2::2]]
        ans = re.match(r"\(([a-d])\)", r["answer"].strip())
        if len(opts) != 4 or not ans:
            skipped["unparsed options or answer"] += 1
            continue
        gold = opts["abcd".index(ans.group(1))]
        stem = re.sub(r"\s+", " ", CHOICE.sub("", q).split("And provide")[0]).strip()
        rng.shuffle(opts)
        x, y, w, h = r["bbox"]
        img_path = args.images / Path(r["image"]["path"] or f"{r['id']}.jpg").name
        if not img_path.exists():
            img_path.write_bytes(r["image"]["bytes"])
        listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        kept.append({"question_id": f"mhs-{r['id']}", "image": str(img_path.resolve()), "question": stem,
                     "options": opts, "category": PT, "default_prompt": f"{stem}\nOptions:\n{listed}\n{DIRECT}",
                     "answer": LETTERS[opts.index(gold)], "target_boxes": [[x, y, x + w, y + h]],
                     "source": f"multihopspatial:{r['hop']}", "base_image": str(img_path.resolve()),
                     "evidence": {"human_label": True, "hop": r["hop"]}})
    with args.out.open("w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(json.dumps({"kept": len(kept), "skipped": skipped, "hops": Counter(k["source"] for k in kept),
                      "answer_letters": Counter(k["answer"] for k in kept)}, indent=1, default=dict))


if __name__ == "__main__":
    main()
