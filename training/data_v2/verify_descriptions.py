"""Round-trip check of the referring descriptions (describe.py): can a different model find the object from the text?

A question names its objects only in words ("the man in a light blue shirt walking, on the right side of the
image"). A description is kept only if a second, larger VLM (Qwen3.5-9B, thinking off) that sees the whole photo
with no box drawn, and only that text, points at the intended object:

- its box overlaps the intended box (IoU >= 0.3) or contains the intended box's centre, and
- among all objects of that class in the photo (from the labels), the intended one is the best match.

That catches descriptions that are wrong (colours, clothing) and ones that fit another object too. The model
is asked for a box as JSON; Qwen3.5 gives coordinates on a 0-1000 scale of the image (checked with --calibrate on
a few items before the full run). Resumable.

    python training/data_v2/verify_descriptions.py --descriptions data/train_v2/descriptions.jsonl \
        --labels 'data/train_v2/annotations/labels_*.jsonl' --out data/train_v2/verified.jsonl \
        --base-url http://127.0.0.1:18452/v1
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from PIL import Image

from generate import iou, where

PROMPT = ("Find {phrase} in this photo. Reply with its bounding box only, as JSON: "
          "{{\"bbox_2d\": [x1, y1, x2, y2]}}.")
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def encode(path: str, max_side: int = 2048) -> tuple[str, tuple[int, int], tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    size = img.size
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode(), size, img.size


def parse_box(text: str, size: tuple[int, int], scale: str, sent: tuple[int, int] | None = None) -> list[float] | None:
    # the box list itself (not the "2" in "bbox_2d", nor the JSON array around the object)
    m = re.search(r"bbox_2d\"?\s*:\s*\[([^\]]+)\]", text or "") or re.search(r"\[\s*(-?\d[^\[\]]*)\]", text or "")
    nums = [float(x) for x in NUM.findall(m.group(1))] if m else []
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    w, h = size
    if scale == "1000":
        x1, x2, y1, y2 = x1 / 1000 * w, x2 / 1000 * w, y1 / 1000 * h, y2 / 1000 * h
    elif sent is not None:  # pixels of the image as sent (it may have been shrunk)
        sx, sy = w / sent[0], h / sent[1]
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def judge(pred, target, others) -> tuple[bool, str, float]:
    """Pass when the predicted box lands on the intended object and on no other object of its class better."""
    if pred is None:
        return False, "no box in reply", 0.0
    score = iou(pred, target)
    cx, cy = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    hit = score >= 0.3 or (pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]
                           and (pred[2] - pred[0]) * (pred[3] - pred[1]) <= 6 * (target[2] - target[0]) * (target[3] - target[1]))
    if not hit:
        return False, "points elsewhere", score
    best_other = max((iou(pred, o) for o in others), default=0.0)
    if best_other > score:
        return False, "matches another object of the same class better", score
    return True, "ok", score


def rescore(paths: list[str], out: Path) -> dict:
    """Merge shard files and re-judge every stored reply with the current parser (the first full run had a
    parser bug: it read the "2" in "bbox_2d" as a coordinate). The first record per object wins."""
    seen, rows = set(), []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["image"], r["object"])
            if key in seen:
                continue
            seen.add(key)
            r["pred_box"] = parse_box(r["raw"], tuple(r["size"]), "1000", tuple(r["sent_size"]))
            r["passed"], r["reason"], r["iou"] = judge(r["pred_box"], r["target_box"], r["others"])
            rows.append(r)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"checked": len(rows), "passed": sum(r["passed"] for r in rows), "reasons": reasons}


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rescore":  # verify_descriptions.py rescore OUT SHARD...
        print(json.dumps(rescore(sys.argv[3:], Path(sys.argv[2])), indent=1))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18452/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--scale", choices=["1000", "pixels"], default="1000")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--follow", type=Path, help="keep checking new descriptions until this file exists")
    ap.add_argument("--only", type=Path, help="file of image#object keys: check only these, in this order")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=2048, help="longest side of the photo sent to the checker")
    ap.add_argument("--no-where", action="store_true",
                    help="send the description alone, without its position in the image: questions no longer say "
                         "where an object is, so the description has to single it out by itself")
    args = ap.parse_args()
    objects = {}
    for path in glob.glob(args.labels):
        for line in open(path):
            r = json.loads(line)
            objects[r["image"]] = r["objects"]
    done = set()
    if args.out.exists():
        done = {(d["image"], d["object"]) for d in map(json.loads, args.out.open())}
    order = None
    if args.only:
        order = {k: i for i, k in enumerate(args.only.read_text().split())}
    client = OpenAI(base_url=args.base_url, api_key="none", timeout=900)
    lock = threading.Lock()
    f = args.out.open("a")

    def one(d):
        objs = {o["id"]: o for o in objects[d["image"]]}
        target = objs[d["object"]]["box"]
        others = [o["box"] for o in objs.values() if o["id"] != d["object"] and o["label"] == d["label"]
                  and o["score"] >= 0.2]
        b64, size, sent = encode(d["image"], args.max_side)
        phrase = d["description"] if args.no_where else f"{d['description']} {where(target, *size)}"
        try:
            r = client.chat.completions.create(
                model=args.model, temperature=0.0, max_tokens=80,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    {"type": "text", "text": PROMPT.format(phrase=phrase)}]}])
            raw = r.choices[0].message.content or ""
        except Exception as e:
            raw = f"ERROR {e!r}"
        pred = parse_box(raw, size, args.scale, sent)
        ok, why, score = judge(pred, target, others)
        rec = {"image": d["image"], "object": d["object"], "label": d["label"], "phrase": phrase, "raw": raw,
               "pred_box": pred, "target_box": target, "size": size, "sent_size": sent, "others": others, "iou": round(score, 3), "passed": ok, "reason": why}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    while True:
        finished = args.follow is None or args.follow.exists()  # read before the descriptions: no race
        descs = [d for d in map(json.loads, args.descriptions.open()) if d.get("description")]
        todo = [d for d in descs if (d["image"], d["object"]) not in done and d["image"] in objects
                and int(hashlib.sha1(Path(d["image"]).name.encode()).hexdigest(), 16) % args.num_shards == args.shard]
        if order is not None:
            todo = sorted((d for d in todo if f"{d['image']}#{d['object']}" in order),
                          key=lambda d: order[f"{d['image']}#{d['object']}"])
        if args.limit:
            todo = todo[:max(0, args.limit - len(done))]
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, todo))
        done |= {(d["image"], d["object"]) for d in todo}
        if finished or (args.limit and len(done) >= args.limit):
            break
        if not todo:
            time.sleep(60)
    f.close()
    rows = [json.loads(line) for line in args.out.open()]
    print(json.dumps({"checked": len(rows), "passed": sum(r["passed"] for r in rows),
                      "reasons": {k: sum(r["reason"] == k for r in rows) for k in {r["reason"] for r in rows}}},
                     indent=1))


if __name__ == "__main__":
    main()
