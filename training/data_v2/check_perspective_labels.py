"""How often do v2's automatic perspective labels agree with TreeBench's human answers?

A measurement only: nothing here trains on TreeBench. "readable" = every question whose reference pose was read,
labelled from the raw facing angle; "kept" = those the v2 rule (labels.robust_quadrant) would keep. The rule's
history: a 20-degree axis margin first (7 of 10 kept right here), then 30 after this check's errors, then the
current rule after a hand check of generated (non-TreeBench) items. For each TreeBench perspective question with the four
quadrant options and at least two target boxes, the reference is the gold box that matches a detected person
(IoU >= 0.5); the other gold box is the target. The labeller then answers from facing + depth, exactly as it
labels v2 training questions.

    python training/data_v2/check_perspective_labels.py --treebench /home/bhsshah/vlm-project/data/treebench \
        --out data/train_v2/checks/perspective_vs_treebench.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from labels import (QUADRANTS, box_nearness, depth_map, detect, facings, load_models, point_xz, quadrant,
                    robust_quadrant)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treebench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.treebench / "questions.jsonl").open()]
    rows = [r for r in rows if tuple(r["options"]) == QUADRANTS and len(r["target_boxes"]) >= 2]
    m = load_models()
    counts, per_item = Counter(), []
    for r in rows:
        img = Image.open(args.treebench / r["image"]).convert("RGB")
        gold = r["options"]["ABCD".index(r["answer"])]
        people = [o for o in detect(m, img) if o["label"] == "person" and o["score"] >= 0.3]
        matches = [i for i, b in enumerate(r["target_boxes"]) if any(iou(b, p["box"]) >= 0.5 for p in people)]
        rec = {"question_id": r["question_id"], "gold": gold, "status": None}
        counts["questions"] += 1
        if len(matches) != 1 or len(r["target_boxes"]) != 2:
            rec["status"] = "reference not identified (no single person box among two gold boxes)"
            counts["skipped_reference"] += 1
            per_item.append(rec)
            continue
        ref_box = r["target_boxes"][matches[0]]
        tgt_box = r["target_boxes"][1 - matches[0]]
        est = facings(m, img, [ref_box])[0]
        if est is None:
            rec["status"] = "pose unreadable"
            counts["skipped_pose"] += 1
            per_item.append(rec)
            continue
        dmap = depth_map(m, img)
        ref_xz = point_xz(box_nearness(dmap, ref_box)["near"], ref_box, img.size)
        tgt_xz = point_xz(box_nearness(dmap, tgt_box)["near"], tgt_box, img.size)
        q_any, bearing = quadrant(ref_xz, est["angle"], tgt_xz, 0.0)
        q_kept, _ = robust_quadrant(ref_xz, est, tgt_xz)
        rec.update({"facing": est, "bearing": round(bearing, 1), "label_any": q_any, "label_kept": q_kept})
        counts["readable"] += 1
        counts["readable_right"] += q_any == gold
        counts["readable_lr_right"] += q_any.split()[-1 if q_any.startswith("Front") else 0].lower() == \
            gold.split()[-1 if gold.startswith("Front") else 0].lower()
        if q_kept:
            counts["kept"] += 1
            counts["kept_right"] += q_kept == gold
        rec["status"] = "ok"
        per_item.append(rec)
    summary = dict(counts)
    if counts["readable"]:
        summary["accuracy_readable"] = round(counts["readable_right"] / counts["readable"], 3)
        summary["left_right_accuracy_readable"] = round(counts["readable_lr_right"] / counts["readable"], 3)
    if counts["kept"]:
        summary["accuracy_kept"] = round(counts["kept_right"] / counts["kept"], 3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "items": per_item}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
