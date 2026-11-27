"""Turn image labels (annotate.py) into TreeBench-style multiple-choice questions for data v2.

Two steps, with referring descriptions (describe.py) in between:

    python training/data_v2/generate.py candidates --labels 'data/train_v2/annotations/labels_*.jsonl' \
        --out data/train_v2/candidates.jsonl
    python training/data_v2/describe.py ...            # writes data/train_v2/descriptions.jsonl
    python training/data_v2/generate.py assemble --candidates data/train_v2/candidates.jsonl \
        --descriptions data/train_v2/descriptions.jsonl --out data/train_v2

Categories (TreeBench's names) and how each answer is decided, all from automatic labels with margins so
borderline cases are left out rather than guessed:

- Reasoning/Perspective Transform: "From the perspective of <person>, in which direction is <object>?" with
  TreeBench's four options in its order (Front left, Front right, Left rear, Right rear). The person's facing
  comes from body keypoints, positions from depth (labels.quadrant). Targets within 30 degrees of an axis are
  dropped. References facing the camera, where left and right flip, are at least half of the items.
- Reasoning/Ordering: "From left to right, which <class> is closest to the camera?" (or farthest, or from
  right to left) over 3-7 same-class objects spaced clearly apart, options "The first one" ... Only groups
  where no other uncertain detection of that class exists, so the count is not off by a missed object.
- Reasoning/Comparison: "Which is closer to the camera, <A> or <B>?" (depth margin) and "Which is closer to
  <person>, <A> or <B>?" (3D distance ratio >= 1.5), with TreeBench-like options ("The <A>", "The <B>",
  "They are about the same distance", "It cannot be determined").
- Perception/OCR: ~10% rendered copies of the SA-1B questions, drawn as TreeBench draws its OCR items: the
  question, " Options:", and one line per option, in white DejaVu Sans 36 px straight onto the photo from
  (50, 50), wrapped at the photo width minus 100 px, lines 36 px apart; the photo keeps its size and the text
  block is added as the last target box. The prompt is TreeBench's OCR prompt.

Objects are named by their description only ("the man in a beige jacket"), never by where they are in the
image: a position ("in the lower right of the image") would tell the model where to look.

Every item keeps `target_boxes` (pixel [x1, y1, x2, y2], the objects the question is about) and an `evidence`
record (facing angle, bearing, nearness values) so labels can be audited. Splits are disjoint by image.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from labels import QUADRANTS, robust_quadrant, snapped_facing

PT, ORD, CMP, OCR = ("Reasoning/Perspective Transform", "Reasoning/Ordering", "Reasoning/Comparison",
                     "Perception/OCR")
LETTERS = "ABCDEFGHIJK"
DIRECT = "Answer with the option's letter from the given choices directly."
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
# items per category: (sft, rl, dev); OCR copies are drawn from each split's other items
TARGETS = {PT: (1200, 1200, 150), ORD: (600, 600, 80), CMP: (600, 600, 80)}
# extra item files (already in TreeBench's format) by name: (sft, rl, dev); split by image like the rest
EXTRA_TARGETS = {"multihopspatial": (1000, 1000, 100)}
OCR_SHARE = 0.10
MIN_SCORE = 0.5  # detections used as question objects
UNCERTAIN = (0.2, 0.5)  # detections in this band make an ordering group unusable
PLURAL = {"person": "people", "bus": "buses", "bench": "benches", "skis": "pairs of skis", "sheep": "sheep",
          "wine glass": "wine glasses", "knife": "knives", "mouse": "mice", "couch": "couches",
          "glass": "glasses", "potted plant": "potted plants", "traffic light": "traffic lights"}
SKIP_CLASSES = {"dining table", "bed", "toilet", "sink", "refrigerator", "oven", "tv", "laptop", "keyboard",
                "mouse", "remote", "cell phone", "book", "scissors", "toothbrush", "hair drier", "teddy bear",
                "spoon", "fork", "knife", "tie"}  # rarely the subject of a dense outdoor scene, or too tiny


def plural(name: str) -> str:
    return PLURAL.get(name, name + "s")


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def usable(o, img_area: float, min_frac: float = 2e-4, max_frac: float = 0.25) -> bool:
    return (o["score"] >= MIN_SCORE and o["label"] not in SKIP_CLASSES
            and min_frac <= area(o["box"]) / img_area <= max_frac)


# ------------------------------------------------------------------------------------------- candidates
def cell(box, w, h) -> tuple[int, int]:
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    return min(int(cx * 3), 2), min(int(cy * 3), 2)


def distinct(o, rec) -> bool:
    """No other object of the same class (even an uncertain one) right next to this one: none whose centre
    lies within the box grown by half its width and height on every side. With the region words and the
    appearance description, that should keep "the woman in an orange sweater in the lower left" to one
    person; the hand check measures how often it does not. Tried first: no same-class object in the same
    ninth of the image (no perspective question survived in crowded scenes), then within a full box-size
    (61 perspective references in the first 1,567 SA-1B images; half a box-size gives 175)."""
    x1, y1, x2, y2 = o["box"]
    w, h = (x2 - x1) / 2, (y2 - y1) / 2
    for p in rec["objects"]:
        if p["id"] == o["id"] or p["label"] != o["label"] or p["score"] < UNCERTAIN[0]:
            continue
        cx, cy = (p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2
        if x1 - w <= cx <= x2 + w and y1 - h <= cy <= y2 + h:
            return False
    return True


def apart(a, b) -> bool:
    """Two objects clearly separate: their boxes do not touch (b's box grown by a quarter of a's size on each
    side), and in 3D they are at least 15% of a's distance from the camera apart. Without this a handbag the
    person is carrying passed as a perspective target ("is her handbag front left or rear right of her?")."""
    x1, y1, x2, y2 = a["box"]
    mx, my = (x2 - x1) / 4, (y2 - y1) / 4
    bx1, by1, bx2, by2 = b["box"]
    if bx1 < x2 + mx and bx2 > x1 - mx and by1 < y2 + my and by2 > y1 - my:
        return False
    return math.dist(a["xz"], b["xz"]) >= 0.15 * a["xz"][1]


def perspective(rec, rng) -> list[dict]:
    img_area = rec["width"] * rec["height"]
    objs = [o for o in rec["objects"] if usable(o, img_area) and distinct(o, rec)]
    refs = [o for o in objs if o["label"] == "person" and snapped_facing(o.get("facing")) is not None]
    out = []
    for ref in refs:
        targets = [t for t in objs if t["id"] != ref["id"] and apart(ref, t)]
        rng.shuffle(targets)
        for t in targets:
            q, bearing = robust_quadrant(ref["xz"], ref["facing"], t["xz"], ref["box"][3] - ref["box"][1])
            if q is None:
                continue
            f = ref["facing"]
            out.append({"category": PT, "kind": "perspective", "objects": [ref["id"], t["id"]], "answer": q,
                        "evidence": {"toward_camera": snapped_facing(f) == 180.0, "signed": f["signed"],
                                     "face": f["face"], "orient_angle": f["angle"], "bearing": round(bearing, 1),
                                     "ref_near": ref["near"], "target_near": t["near"]}})
            break  # one target per reference
    return out


def could_be_member(u, sure) -> bool:
    """Could an uncertain detection be one more member of the row (so counting would be off by one)? Not if
    it duplicates a member (IoU > 0.3), sits outside the row's height band, or is far smaller or larger
    than the members. (Rejecting a group for any uncertain detection kept 41 of 782 groups.)"""
    if any(iou(u["box"], s["box"]) > 0.3 for s in sure):
        return False
    ys = [(s["box"][1] + s["box"][3]) / 2 for s in sure]
    mh = sum(s["box"][3] - s["box"][1] for s in sure) / len(sure)
    ma = sum(area(s["box"]) for s in sure) / len(sure)
    cy = (u["box"][1] + u["box"][3]) / 2
    return min(ys) - mh <= cy <= max(ys) + mh and ma / 4 <= area(u["box"]) <= ma * 4


def ordering(rec, rng) -> list[dict]:
    img_area = rec["width"] * rec["height"]
    by_cls = defaultdict(list)
    for o in rec["objects"]:
        by_cls[o["label"]].append(o)
    out = []
    for cls, group in by_cls.items():
        if cls in SKIP_CLASSES:
            continue
        sure = [o for o in group if usable(o, img_area, 1e-4)]
        unsure = [o for o in group if UNCERTAIN[0] <= o["score"] < UNCERTAIN[1]]
        if not 3 <= len(sure) <= 7 or any(could_be_member(u, sure) for u in unsure):
            continue
        sure.sort(key=lambda o: (o["box"][0] + o["box"][2]) / 2)
        cx = [(o["box"][0] + o["box"][2]) / 2 for o in sure]
        mean_w = sum(o["box"][2] - o["box"][0] for o in sure) / len(sure)
        if min(b - a for a, b in zip(cx, cx[1:])) < 0.6 * mean_w:  # left-to-right order must be unmistakable
            continue
        near = [o["near"] for o in sure]
        for which in ("closest", "farthest"):
            vals = near if which == "closest" else [-v for v in near]
            order = sorted(range(len(sure)), key=lambda i: -vals[i])
            if vals[order[0]] - vals[order[1]] < 0.06:
                continue
            direction = rng.choice(["left to right", "right to left"])
            pos = order[0] if direction == "left to right" else len(sure) - 1 - order[0]
            ids = [o["id"] for o in sure] if direction == "left to right" else [o["id"] for o in reversed(sure)]
            out.append({"category": ORD, "kind": f"ordering_{which}", "objects": ids, "class": cls,
                        "direction": direction, "answer_index": pos, "n": len(sure),
                        "evidence": {"near_left_to_right": [round(v, 3) for v in near],
                                     "margin": round(vals[order[0]] - vals[order[1]], 3)}})
            break
    return out


def comparison(rec, rng) -> list[dict]:
    img_area = rec["width"] * rec["height"]
    objs = [o for o in rec["objects"] if usable(o, img_area) and distinct(o, rec)]
    out = []
    pairs = [(a, b) for i, a in enumerate(objs) for b in objs[i + 1:]
             if iou(a["box"], b["box"]) < 0.05 and a["label"] != b["label"]]
    rng.shuffle(pairs)
    for a, b in pairs[:20]:
        if abs(a["near"] - b["near"]) >= 0.12 and min(a["near_p25"], b["near_p25"]) >= 0:
            closer = a if a["near"] > b["near"] else b
            # the closer one's lower quartile must still beat the other's upper quartile: no overlap in depth
            other = b if closer is a else a
            if closer["near_p25"] > other["near_p75"]:
                out.append({"category": CMP, "kind": "closer_to_camera", "objects": [a["id"], b["id"]],
                            "answer_object": closer["id"],
                            "evidence": {"near": {a["id"]: a["near"], b["id"]: b["near"]}}})
                break
    refs = [o for o in objs if o["label"] == "person"]
    rng.shuffle(refs)
    for ref in refs[:3]:
        cands = [o for o in objs if o["id"] != ref["id"] and apart(ref, o)]
        rng.shuffle(cands)
        for a, b in [(cands[i], cands[i + 1]) for i in range(0, len(cands) - 1, 2)]:
            if a["label"] == b["label"]:
                continue
            da, db = math.dist(ref["xz"], a["xz"]), math.dist(ref["xz"], b["xz"])
            if max(da, db) / max(min(da, db), 1e-6) >= 1.5:
                out.append({"category": CMP, "kind": "closer_to_person", "objects": [ref["id"], a["id"], b["id"]],
                            "answer_object": a["id"] if da < db else b["id"],
                            "evidence": {"dist_3d": {a["id"]: round(da, 3), b["id"]: round(db, 3)}}})
                return out
    return out


def cmd_candidates(args) -> None:
    rng = random.Random(args.seed)
    out = []
    for path in sorted(glob.glob(args.labels)):
        for line in open(path):
            rec = json.loads(line)
            if not Path(rec["image"]).exists():  # dropped after labelling (looked like a test image)
                continue
            for fn in (perspective, ordering, comparison):
                for c in fn(rec, rng):
                    c.update({"image": rec["image"], "width": rec["width"], "height": rec["height"],
                              "boxes": {str(o["id"]): o["box"] for o in rec["objects"] if o["id"] in c["objects"]},
                              "labels": {str(o["id"]): o["label"] for o in rec["objects"] if o["id"] in c["objects"]}})
                    out.append(c)
    # objects that need a referring description (ordering groups are referred to by class and position)
    need = {}
    for c in out:
        ids = c["objects"] if c["category"] != ORD else []
        for i in ids:
            need[f"{c['image']}#{i}"] = {"image": c["image"], "object": i, "label": c["labels"][str(i)],
                                         "box": c["boxes"][str(i)]}
    with open(args.out, "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    Path(args.out).with_suffix(".objects.jsonl").write_text("".join(json.dumps(v) + "\n" for v in need.values()))
    print(json.dumps({"candidates": len(out), "by_kind": Counter(c["kind"] for c in out),
                      "objects_to_describe": len(need)}, indent=1, default=dict))


# ------------------------------------------------------------------------------------------- assemble
def where(box, w, h) -> str:
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    row = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    if row == "middle" and col == "center":
        return "in the center of the image"
    if row == "middle":
        return f"on the {col} side of the image"
    if col == "center":
        return f"in the {row} part of the image"
    return f"in the {row} {col} of the image"


def refer(c, i, desc) -> str:
    """How a question names an object: its description alone. Not its position in the image (where(), used
    before): that gave away where to look. The descriptions used are the ones that passed the round-trip check
    without a position (verify_descriptions.py --no-where), so each still singles out one object."""
    return desc.get(f"{c['image']}#{i}") or ""


OCR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
OCR_PROMPT = "Recognize the question and options in the image and answer it."


def render_ocr(src: dict, out_dir: Path) -> dict:
    """An OCR copy drawn as TreeBench's OCR items are (measured on its 68): white DejaVu Sans 36 px written
    straight onto the photo from (50, 50); the question followed by " Options:", then one line per option, all
    wrapped at the photo width minus 100 px, lines 36 px apart. The photo keeps its size, so the other target
    boxes do not move; the text block is added as the last target box, as TreeBench does."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src["image"]).convert("RGB")
    font = ImageFont.truetype(OCR_FONT, 36)
    limit = img.width - 100

    def wrap(text: str) -> list[str]:
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if cur and font.getlength(trial) > limit:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + [cur]

    lines = wrap(src["question"] + " Options:")
    for i, o in enumerate(src["options"]):
        lines += wrap(f"{LETTERS[i]}. {o}")
    draw = ImageDraw.Draw(img)
    for j, line in enumerate(lines):
        draw.text((50, 50 + 36 * j), line, fill=(255, 255, 255), font=font)
    path = out_dir / f"{src['question_id']}.jpg"
    img.save(path, quality=95)
    text_box = [50.0, 50.0, 50.0 + max(font.getlength(x) for x in lines), 50.0 + 36.0 * len(lines)]
    return dict(src, question_id=src["question_id"] + "-ocr", image=str(path), category=OCR,
                source=src["source"] + "+ocr", default_prompt=f"{OCR_PROMPT}\n{DIRECT}",
                target_boxes=src["target_boxes"] + [text_box])


def short_name(c, i, desc) -> str:
    d = desc.get(f"{c['image']}#{i}") or f"the {c['labels'][str(i)]}"
    return "The " + d[4:] if d.lower().startswith("the ") else d[:1].upper() + d[1:]


def build_item(c, desc, rng) -> dict | None:
    k = c["kind"]
    ids = c["objects"]
    if k == "perspective":
        ref, tgt = (refer(c, i, desc) for i in ids)
        if not ref or not tgt:
            return None
        q = rng.choice([f"From the perspective of {ref}, in which direction is {tgt}?",
                        f"From the viewpoint of {ref}, what is the relative direction of {tgt}?",
                        f"From the point of view of {ref}, where is {tgt} located relative to them?"])
        options, answer = list(QUADRANTS), QUADRANTS.index(c["answer"])
    elif k.startswith("ordering"):
        n, cls = c["n"], c["class"]
        which = "closest to" if k == "ordering_closest" else "farthest from"
        q = f"From {c['direction']}, which {cls} is the one {which} the camera?"
        if rng.random() < 0.5:
            q = f"Counting the {plural(cls)} from {c['direction']}, which one is {which} the camera?"
        options = [f"The {ORDINALS[j]} one" for j in range(n)]
        answer = c["answer_index"]
    elif k == "closer_to_camera":
        a, b = ids
        ra, rb = refer(c, a, desc), refer(c, b, desc)
        if not ra or not rb:
            return None
        q = f"Which object is closer to the camera, {ra}, or {rb}?"
        options = [short_name(c, a, desc), short_name(c, b, desc), "They are about the same distance",
                   "It cannot be determined"]
        answer = 0 if c["answer_object"] == a else 1
    elif k == "closer_to_person":
        p, a, b = ids
        rp, ra, rb = refer(c, p, desc), refer(c, a, desc), refer(c, b, desc)
        if not (rp and ra and rb):
            return None
        q = f"Considering the real distances in the scene, which is closer to {rp}: {ra}, or {rb}?"
        options = [short_name(c, a, desc), short_name(c, b, desc), "They are about the same distance",
                   "It cannot be determined"]
        answer = 0 if c["answer_object"] == a else 1
    else:
        return None
    if k in ("closer_to_camera", "closer_to_person"):  # any of the four positions, so no letter is always wrong
        right = options[answer]
        rng.shuffle(options)
        answer = options.index(right)
    listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    qid = "v2-" + hashlib.sha1(f"{c['image']}|{k}|{ids}".encode()).hexdigest()[:12]
    return {"question_id": qid, "image": c["image"], "question": q, "options": options, "category": c["category"],
            "default_prompt": f"{q}\nOptions:\n{listed}\n{DIRECT}", "answer": LETTERS[answer],
            "target_boxes": [c["boxes"][str(i)] for i in ids], "source": f"sa1b-generated:{k}", "base_image": c["image"],
            "evidence": c["evidence"], "confidence": round(confidence(c), 4)}


def confidence(c) -> float:
    """How far a label is from flipping (larger = safer), used to pick the safest labels first:
    perspective = degrees from the nearest quadrant boundary; ordering and closer-to-camera = the depth margin;
    closer-to-person = the 3D distance ratio minus 1."""
    ev = c["evidence"]
    if c["kind"] == "perspective":
        return min(abs(abs(ev["bearing"]) - a) for a in (0.0, 90.0, 180.0)) / 45.0
    if c["kind"].startswith("ordering"):
        return ev["margin"] / 0.06
    if c["kind"] == "closer_to_camera":
        a, b = ev["near"].values()
        return abs(a - b) / 0.12
    if c["kind"] == "closer_to_person":
        a, b = ev["dist_3d"].values()
        return max(a, b) / max(min(a, b), 1e-6) - 1.0
    return 0.0


def split_of(image: str, seed: int, dev_share: float = 0.06) -> str:
    h = int(hashlib.sha1(f"{seed}:{image}".encode()).hexdigest(), 16) % 1000 / 1000
    return "dev" if h < dev_share else ("sft" if h < dev_share + (1 - dev_share) / 2 else "rl")


def balanced(items: list[dict], n: int, key, rng) -> list[dict]:
    """Up to n items, as even as possible across `key` values; within each value the most confident labels
    first (random order among equals; human-labelled items have no confidence and stay random)."""
    groups = defaultdict(list)
    for r in items:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
        g.sort(key=lambda r: r.get("confidence", 0.0))  # stable: ties keep the shuffled order; best last
    out = []
    while len(out) < n and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def cmd_assemble(args) -> None:
    rng = random.Random(args.seed)
    passed = None
    if args.verified:  # only descriptions a second model could trace back to the right object
        passed = {f"{v['image']}#{v['object']}" for v in map(json.loads, open(args.verified)) if v["passed"]}
    desc, n_desc, n_not_real = {}, 0, 0
    for line in open(args.descriptions):
        d = json.loads(line)
        n_desc += 1
        n_not_real += "NOT REAL" in (d.get("raw") or "").upper()
        key = f"{d['image']}#{d['object']}"
        if d.get("description") and (passed is None or key in passed):
            desc[key] = d["description"]
    items = [it for c in map(json.loads, open(args.candidates)) if (it := build_item(c, desc, rng))]
    per_image = defaultdict(list)
    for it in items:
        per_image[it["image"]].append(it)
    out = {"sft": [], "rl": [], "dev": []}
    pools = defaultdict(lambda: defaultdict(list))
    for img, its in per_image.items():
        rng.shuffle(its)
        for it in its[:3]:  # at most three questions per image
            pools[split_of(img, args.seed)][it["category"]].append(it)
    for cat, sizes in TARGETS.items():
        for split, n in zip(("sft", "rl", "dev"), sizes):
            cands = pools[split][cat]
            if cat == PT:  # even quadrants; at least half with the reference facing the camera
                toward = [r for r in cands if r["evidence"]["toward_camera"]]
                away = [r for r in cands if not r["evidence"]["toward_camera"]]
                pick = balanced(toward, max(n // 2, n - len(away)), lambda r: r["answer"], rng)
                pick += balanced(away, n - len(pick), lambda r: r["answer"], rng)
            else:
                pick = balanced(cands, n, lambda r: r["answer"], rng)
            out[split] += pick
    for path in args.extra or []:
        name = Path(path).stem
        sizes = EXTRA_TARGETS.get(name)
        if not sizes:
            raise SystemExit(f"no EXTRA_TARGETS entry for {name}")
        by_split = defaultdict(list)
        for r in map(json.loads, open(path)):
            by_split[split_of(r["base_image"], args.seed)].append(r)
        for split, n in zip(("sft", "rl", "dev"), sizes):
            out[split] += balanced(by_split[split], n, lambda r: r["answer"], rng)
    chosen = {r["question_id"] for v in out.values() for r in v}
    reserve = Path(args.out) / "reserve"
    reserve.mkdir(parents=True, exist_ok=True)
    n_reserve = {}
    for split in ("sft", "rl"):  # left-over questions, same images as that split (never dev images)
        rest = [r for cat in pools[split] for r in pools[split][cat] if r["question_id"] not in chosen]
        n_reserve[split] = len(rest)
        with (reserve / f"{split}.jsonl").open("w") as f:
            for r in rest:
                f.write(json.dumps(r) + "\n")
    ocr_dir = Path(args.out) / "images" / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    for split in out:
        # OCR copies only of the SA-1B questions: TreeBench's OCR photos are all high resolution, and its 36 px
        # text would cover most of a 640 px MultihopSpatial photo
        pool = [r for r in out[split] if r["source"].startswith("sa1b-generated")]
        src = rng.sample(pool, min(len(pool), int(len(out[split]) * OCR_SHARE)))
        out[split] += [render_ocr(r, ocr_dir) for r in src]
        rng.shuffle(out[split])
        d = Path(args.out) / split
        d.mkdir(parents=True, exist_ok=True)
        with (d / "questions.jsonl").open("w") as f:
            for r in out[split]:
                f.write(json.dumps(r) + "\n")
    imgs = {k: {r["base_image"] for r in v} for k, v in out.items()}
    report = {"sizes": {k: len(v) for k, v in out.items()},
              "per_category": {k: dict(Counter(r["category"] for r in v)) for k, v in out.items()},
              "per_kind": {k: dict(Counter(r["source"] for r in v)) for k, v in out.items()},
              "answer_letters": {k: dict(Counter(r["answer"] for r in v)) for k, v in out.items()},
              "perspective_toward_camera_share": {k: round(sum(r["evidence"].get("toward_camera", False)
                                                                 for r in v if r["category"] == PT)
                                                             / max(1, sum(r["category"] == PT for r in v)), 3)
                                                  for k, v in out.items()},
              "images": {k: len(v) for k, v in imgs.items()},
              "image_overlap": {"sft&rl": len(imgs["sft"] & imgs["rl"]), "sft&dev": len(imgs["sft"] & imgs["dev"]),
                                "rl&dev": len(imgs["rl"] & imgs["dev"])},
              "candidates_built": len(items), "objects_described": n_desc, "objects_not_real": n_not_real,
              "descriptions_used": len(desc), "reserve": n_reserve}
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--labels", required=True, help="glob of annotate.py outputs")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("assemble")
    a.add_argument("--candidates", required=True)
    a.add_argument("--descriptions", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--verified", help="verify_descriptions.py output: use only descriptions that passed")
    a.add_argument("--extra", nargs="*", help="item files already in TreeBench format (see EXTRA_TARGETS)")
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"candidates": cmd_candidates, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
