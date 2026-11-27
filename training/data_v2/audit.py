"""Draw generated questions with their boxes, for checking labels by eye.

Box 1 is red (the reference, or object A), box 2 blue, further boxes green. Writes one JPEG per item and an
index.jsonl (question, options, label, evidence) next to them.

    python training/data_v2/audit.py --questions data/train_v2/dev/questions.jsonl --n 40 \
        --out data/train_v2/checks/audit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLOURS = [(255, 0, 0), (0, 90, 255), (0, 200, 0), (0, 200, 0), (0, 200, 0), (0, 200, 0), (0, 200, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--category", help="only this TreeBench category (substring)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.questions.open()]
    rows = [r for r in rows if "+ocr" not in r["source"] and (not args.category or args.category in r["category"])]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "index.jsonl").open("w") as f:
        for k, r in enumerate(rows):
            img = Image.open(r["image"]).convert("RGB")
            d = ImageDraw.Draw(img)
            lw = max(3, img.width // 300)
            for i, b in enumerate(r["target_boxes"]):
                d.rectangle(b, outline=COLOURS[min(i, len(COLOURS) - 1)], width=lw)
                d.text((b[0], max(0, b[1] - 12 * lw)), str(i + 1), fill=COLOURS[min(i, len(COLOURS) - 1)],
                       font_size=12 * lw)
            img.thumbnail((1600, 1600))
            name = f"{k:03d}_{r['question_id']}.jpg"
            img.save(args.out / name, quality=88)
            f.write(json.dumps({"file": name, "question": r["question"], "options": r["options"],
                                "answer": r["answer"], "category": r["category"], "evidence": r["evidence"]}) + "\n")
    print(f"{len(rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
