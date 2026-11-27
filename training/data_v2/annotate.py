"""Label images for data v2 (see labels.py): one JSON line per image, plus a small depth map per image.

Shardable, resumable (images already in the shard's output are skipped):

    python training/data_v2/annotate.py --manifest data/train_v2/images/sa1b/manifest.txt \
        --out data/train_v2/annotations --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from labels import annotate, load_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, help="file listing image paths")
    ap.add_argument("--images-dir", type=Path, help="or: every .jpg in this folder (e.g. while a download runs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--wait-for", type=Path, help="keep watching --images-dir until this file exists")
    args = ap.parse_args()
    def listing() -> list[str]:
        if args.manifest:
            paths = [p for p in args.manifest.read_text().split() if p]
        else:
            # only finished files: the downloader writes each JPEG whole, but give it a few seconds
            paths = sorted(str(p.resolve()) for p in args.images_dir.glob("*.jpg") if time.time() - p.stat().st_mtime > 10)
        # shard by file name, so a growing folder never moves an image to another shard (no double work)
        paths = [p for p in paths if int(hashlib.sha1(Path(p).name.encode()).hexdigest(), 16) % args.num_shards == args.shard]
        return paths[:args.limit] if args.limit else paths

    (args.out / "depth").mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"labels_{args.shard:02d}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["image"] for line in out_path.open()}
    m = load_models()
    t0, n = time.time(), 0
    with out_path.open("a") as f:
        while True:
            todo = [p for p in listing() if p not in done]
            for p in todo:
                done.add(p)
                if not Path(p).exists():  # removed by the downloader's test-image check
                    continue
                try:
                    rec, small = annotate(m, Image.open(p))
                except Exception as e:  # a broken image must not stop the shard
                    print(f"skip {p}: {e!r}", flush=True)
                    continue
                depth_file = args.out / "depth" / (Path(p).stem + ".npy")
                np.save(depth_file, small)
                f.write(json.dumps({"image": p, "depth": str(depth_file.resolve()), **rec}) + "\n")
                n += 1
                if n % 100 == 0:
                    f.flush()
                    print(f"{n} images, {(time.time() - t0) / n:.2f} s/image", flush=True)
            f.flush()
            # --wait-for: keep picking up new images until the downloader's report file exists
            if not args.wait_for or (args.wait_for.exists() and not todo):
                break
            if not todo:
                time.sleep(60)
    print(f"done: {n} new images in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
