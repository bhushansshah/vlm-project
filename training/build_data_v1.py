"""Build training data v1 for TreeBench-style SFT and RL (docs/plans/treebench_sft_rl_plan.md, section 3).

Every item is converted to TreeBench's own prompt format (question, "Options:", lettered options, "Answer with
the option's letter ...") and labelled with the TreeBench category it practises, so per-category results line
up with the test set. TreeBench itself is never read except to drop training images that look like a TreeBench
image (perceptual hash).

    python training/build_data_v1.py --out data/train_v1

Sources (all downloaded to ~/vlm_cache/train_data, none from SA-1B):
- SpatialMQA (COCO): "if you were the man, where is X?" -> Perspective Transform.
- STVQA-7K (Visual Genome): orientation -> Perspective; depth/distance/size -> Comparison; reach ->
  Contact and Occlusion; relation -> Spatial Containment; count -> Ordering; existence/instance_location ->
  Object Retrieval. Already multiple choice.
- TreeVGR-RL-37K: VisDrone counting (MC, high-res) -> Ordering; V*-train colour questions -> Attributes
  (4 options: the true colour and 3 others from a fixed vocabulary). Target boxes kept.
- Rendered OCR: copies of other items with the question and options drawn above the image, asked with
  TreeBench's OCR prompt -> OCR.

Output: <out>/{sft,rl,dev}/questions.jsonl in TreeBench's schema plus "source" and "base_image" (absolute
image paths; run them with `vlmbench run --dataset tbstyle --data-dir <out>/<split>`), and <out>/images/.
Splits are disjoint by image. Seeded; re-running gives the same files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CACHE = Path.home() / "vlm_cache" / "train_data"
LETTERS = "ABCDEFGH"
DIRECT = "Answer with the option's letter from the given choices directly."
OCR_PROMPT = "Recognize the question and options in the image and answer it.\n" + DIRECT
PT, ORD, CMP, CON, SPC = ("Reasoning/Perspective Transform", "Reasoning/Ordering", "Reasoning/Comparison",
                          "Reasoning/Contact and Occlusion", "Reasoning/Spatial Containment")
ATT, RET, OCR = "Perception/Attributes", "Perception/Object Retrieval", "Perception/OCR"
STVQA_CAT = {"orientation": PT, "depth": CMP, "distance": CMP, "size": CMP, "reach": CON, "relation": SPC,
             "count": ORD, "existence": RET, "instance_location": RET}
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray",
           "silver", "gold", "beige"]

# items per category: (sft, rl, dev). OCR items are rendered copies drawn from the other categories' pools.
TARGETS = {PT: (800, 800, 100), ORD: (450, 450, 70), CMP: (450, 450, 55), CON: (250, 250, 50),
           SPC: (250, 250, 35), ATT: (300, 300, 35), RET: (150, 150, 25), OCR: (300, 300, 80)}


def prompt(question: str, options: list[str]) -> str:
    listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\nOptions:\n{listed}\n{DIRECT}"


def item(qid, image, question, options, answer_idx, category, source, boxes=()):
    return {"question_id": qid, "image": str(image), "question": question, "options": list(options),
            "answer": LETTERS[answer_idx], "category": category, "source": source,
            "default_prompt": prompt(question, options), "target_boxes": [list(map(float, b)) for b in boxes],
            "base_image": str(image)}


def shuffled(options: list[str], answer: str, rng: random.Random) -> tuple[list[str], int]:
    opts = list(options)
    rng.shuffle(opts)
    return opts, opts.index(answer)


def load_spatialmqa(rng):
    root = CACHE / "SpatialMQA"
    out = {"train": [], "dev": []}
    for split in ("train", "dev"):
        for i, line in enumerate((root / f"{split}.jsonl").open()):
            r = json.loads(line)
            if r["answer"] not in r["options"]:
                continue
            opts, k = shuffled(r["options"], r["answer"], rng)
            out[split].append(item(f"smqa-{split}-{i}", root / "images" / r["image"], r["question"], opts, k, PT,
                                   "spatialmqa"))
    return out


def load_stvqa(rng, img_dir: Path):
    out = {"train": [], "dev": []}
    img_dir.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("dev", "val")):
        df = pd.read_parquet(CACHE / "STVQA-7K" / "data" / f"{name}-00000-of-00001.parquet")
        for i, r in df.iterrows():
            cat = STVQA_CAT.get(r["category"])
            m = re.findall(r"\(([A-H])\)\s*(.+)", r["question_with_options"])
            if not cat or not m or r["answer_only"] not in LETTERS[:len(m)]:
                continue
            options = [t.strip() for _, t in m]
            answer = options[LETTERS.index(r["answer_only"])]
            path = img_dir / f"{r['image_id']}.jpg"
            if not path.exists():
                img = r["images"]
                data = img["bytes"] if isinstance(img, dict) else img
                Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=95)
            opts, k = shuffled(options, answer, rng)
            out[split].append(item(f"stvqa-{name}-{i}", path, r["question_only"], opts, k, cat,
                                   f"stvqa-{r['category']}"))
    return out


def load_treevgr(rng):
    root = CACHE / "TreeVGR-RL-37K"
    df = pd.read_parquet(root / "vstar30k_visdrone6k_x1y1x2y2.parquet")
    items = []
    for i, r in df.iterrows():
        q = r["problem"].replace("<image>\n", "").strip()
        boxes = [t["bbox"] for t in r["target_instances"]]
        image = root / r["images"][0]
        if "(A)" in q:  # VisDrone counting, already multiple choice
            stem, _, rest = q.partition("Options:")
            m = re.findall(r"\(([A-H])\)\s*([^\n(]+)", rest or q)
            if not m or r["answer"] not in LETTERS[:len(m)]:
                continue
            options = [t.strip() for _, t in m]
            answer = options[LETTERS.index(r["answer"])]
            opts, k = shuffled(options, answer, rng)
            items.append(item(f"tvgr-{i}", image, stem.strip(), opts, k, ORD, "treevgr-visdrone", boxes))
            continue
        m = re.match(r"What is the colou?r of the (.+)\?$", q)
        a = re.search(r"\bis (\w+)\.?$", r["answer"])
        if m and a and a.group(1).lower() in COLOURS:
            true = a.group(1).lower()
            options = [true] + rng.sample([c for c in COLOURS if c != true], 3)
            opts, k = shuffled(options, true, rng)
            items.append(item(f"tvgr-{i}", image, q, opts, k, ATT, "treevgr-vstar-colour", boxes))
    return items


def render_ocr(src: dict, out_dir: Path) -> dict:
    """Draw the question and options above the image, as TreeBench's OCR category does."""
    img = Image.open(src["image"]).convert("RGB")
    w = img.width
    size = max(14, w // 40)
    font = ImageFont.load_default(size=size)
    lines, text = [], src["question"]
    words, cur = text.split(), ""
    for word in words:  # wrap the question to the image width
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) > w - 2 * size and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(src["options"])]
    banner = int(len(lines) * size * 1.35 + size)
    canvas = Image.new("RGB", (w, img.height + banner), "white")
    draw = ImageDraw.Draw(canvas)
    for j, line in enumerate(lines):
        draw.text((size, size // 2 + int(j * size * 1.35)), line, fill="black", font=font)
    canvas.paste(img, (0, banner))
    path = out_dir / f"{src['question_id']}.jpg"
    canvas.save(path, quality=92)
    r = dict(src, question_id=src["question_id"] + "-ocr", image=str(path), category=OCR,
             source=src["source"] + "+ocr", default_prompt=OCR_PROMPT,
             target_boxes=[[b[0], b[1] + banner, b[2], b[3] + banner] for b in src["target_boxes"]])
    return r


def phash_filter(items: list[dict], treebench_dir: Path) -> tuple[list[dict], int]:
    """Drop items whose image is a near-duplicate (pHash Hamming distance <= 10) of a TreeBench image."""
    import imagehash

    tb = [imagehash.phash(Image.open(p)) for p in sorted((treebench_dir / "images").glob("*.jpg"))]
    cache: dict[str, bool] = {}
    kept = []
    for r in items:
        if r["image"] not in cache:
            h = imagehash.phash(Image.open(r["image"]))
            cache[r["image"]] = any(h - t <= 10 for t in tb)
        if not cache[r["image"]]:
            kept.append(r)
    return kept, sum(cache.values())


def split_by_image(items: list[dict], dev_share: float, seed: int):
    """Assign each image to one of three buckets so a photo never appears in two splits."""
    out = defaultdict(list)
    for r in items:
        h = int(hashlib.sha1(f"{seed}:{r['image']}".encode()).hexdigest(), 16) % 1000 / 1000
        out["dev" if h < dev_share else ("sft" if h < dev_share + (1 - dev_share) / 2 else "rl")].append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/train_v1"))
    ap.add_argument("--treebench", type=Path, default=Path("data/treebench"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    (args.out / "images" / "ocr").mkdir(parents=True, exist_ok=True)

    smqa = load_spatialmqa(rng)
    stv = load_stvqa(rng, args.out / "images" / "stvqa")
    tvgr = load_treevgr(rng)
    # SpatialMQA and STVQA ship their own dev splits (other images); TreeVGR is split here by image.
    tv = split_by_image(tvgr, 0.05, args.seed)
    pools = {"train": smqa["train"] + stv["train"] + tv["sft"] + tv["rl"], "dev": smqa["dev"] + stv["dev"] + tv["dev"]}
    report = {"loaded": {k: len(v) for k, v in pools.items()}}
    for k in pools:
        pools[k], n_dup = phash_filter(pools[k], args.treebench)
        report[f"treebench_lookalike_images_dropped_{k}"] = n_dup

    by_cat = {k: defaultdict(list) for k in pools}
    for k, rows in pools.items():
        for r in rows:
            by_cat[k][r["category"]].append(r)
    # train pool -> sft and rl, disjoint by image
    train_split = split_by_image(pools["train"], 0.0, args.seed + 1)
    sft_imgs = {r["image"] for r in train_split["sft"]}
    out = {"sft": [], "rl": [], "dev": []}
    for cat, (n_sft, n_rl, n_dev) in TARGETS.items():
        if cat == OCR:
            continue
        cands = by_cat["train"][cat]
        rng.shuffle(cands)
        out["sft"] += [r for r in cands if r["image"] in sft_imgs][:n_sft]
        out["rl"] += [r for r in cands if r["image"] not in sft_imgs][:n_rl]
        dev = by_cat["dev"][cat]
        rng.shuffle(dev)
        out["dev"] += dev[:n_dev]
    # the sources' own dev splits reuse a few training images: drop those dev items
    train_imgs = {r["base_image"] for r in out["sft"] + out["rl"]}
    report["dev_dropped_shared_image"] = sum(r["base_image"] in train_imgs for r in out["dev"])
    out["dev"] = [r for r in out["dev"] if r["base_image"] not in train_imgs]
    # rendered OCR copies of items already in each split (same image, so no cross-split leak)
    for k, (i) in (("sft", 0), ("rl", 1), ("dev", 2)):
        src = rng.sample(out[k], TARGETS[OCR][i])
        out[k] += [render_ocr(r, args.out / "images" / "ocr") for r in src]

    for k, rows in out.items():
        rng.shuffle(rows)
        (args.out / k).mkdir(exist_ok=True)
        # one folder per split, readable by vlmbench's "tbstyle" dataset (--dataset tbstyle --data-dir <out>/<k>)
        with (args.out / k / "questions.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(dict(r, image=str(Path(r["image"]).resolve()))) + "\n")
    imgs = {k: {r["base_image"] for r in v} for k, v in out.items()}
    report["image_overlap"] = {"sft&rl": len(imgs["sft"] & imgs["rl"]), "sft&dev": len(imgs["sft"] & imgs["dev"]),
                               "rl&dev": len(imgs["rl"] & imgs["dev"])}
    report["sizes"] = {k: len(v) for k, v in out.items()}
    report["per_category"] = {k: dict(Counter(r["category"] for r in v)) for k, v in out.items()}
    report["answer_letters"] = {k: dict(Counter(r["answer"] for r in v)) for k, v in out.items()}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
