"""Download a few SA-1B parts (official links), keep the JPEGs, and drop images that look like a test image.

SA-1B is TreeBench's own image source (about 2250 x 1500 px, dense scenes). Its licence must be accepted at
https://ai.meta.com/datasets/segment-anything-downloads, which gives a links file (`file_name<TAB>cdn_link`,
one line per part of ~11K images, ~11 GB). Only the JPEGs are kept; SA-1B's own mask files are skipped.

Every kept image is compared with TreeBench's 405 images and V* Bench's 191 (both SA-1B) by perceptual hash;
a Hamming distance <= 10 (v1's rule) drops it.

    python training/data_v2/fetch_sa1b.py --links data/train_v2/raw/sa1b_links.txt --parts 2 \
        --out data/train_v2/images/sa1b --treebench data/treebench
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image


def test_hashes(treebench: Path, vstar_cache: Path):
    import imagehash
    from huggingface_hub import snapshot_download

    paths = sorted((treebench / "images").glob("*.jpg"))
    vstar = Path(snapshot_download("craigwu/vstar_bench", repo_type="dataset", allow_patterns=["*.jpg"],
                                   cache_dir=str(vstar_cache)))
    paths += sorted(vstar.rglob("*.jpg"))
    return [imagehash.phash(Image.open(p)) for p in paths], len(paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path, required=True)
    ap.add_argument("--parts", type=int, default=2, help="how many SA-1B parts to download")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--treebench", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import imagehash

    lines = [ln.rstrip("\n").split("\t") for ln in args.links.open() if ln.strip()]
    parts = [(n, u) for n, u in lines if n.endswith(".tar")]
    chosen = random.Random(args.seed).sample(parts, args.parts)
    args.out.mkdir(parents=True, exist_ok=True)
    tests, n_tests = test_hashes(args.treebench, args.out.parent / "_vstar")
    report = {"parts": [n for n, _ in chosen], "test_images_hashed": n_tests, "kept": 0, "dropped_lookalike": 0}
    dropped = []
    for name, url in chosen:
        print(f"downloading {name}", flush=True)
        # the parts are gzip-compressed despite the .tar name: "r|*" detects the compression
        with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|*") as tar:
            for member in tar:
                if not member.name.endswith(".jpg"):
                    continue
                dst = args.out / Path(member.name).name
                if not dst.exists():
                    with tar.extractfile(member) as src:
                        dst.write_bytes(src.read())
                h = imagehash.phash(Image.open(dst))
                if any(h - t <= 10 for t in tests):
                    dropped.append(dst.name)
                    dst.unlink()
                    report["dropped_lookalike"] += 1
                else:
                    report["kept"] += 1
    report["dropped_files"] = dropped
    (args.out / "manifest.txt").write_text("\n".join(str(p.resolve()) for p in sorted(args.out.glob("*.jpg"))) + "\n")
    (args.out / "fetch_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_files"}, indent=1))


if __name__ == "__main__":
    main()
