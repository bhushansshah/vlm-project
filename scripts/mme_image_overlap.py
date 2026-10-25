"""Repair byte-identical cross-batch images, preserving every task quota."""
from __future__ import annotations
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
import mme_realworld as mme


def repair(seed=42):
    data = mme.DATA
    first = json.loads((data / "batch_1.json").read_text())
    second = json.loads((data / "batch_2.json").read_text())
    m1 = json.loads((data / "images_batch_1.json").read_text())
    m2 = json.loads((data / "images_batch_2.json").read_text())
    annotations = json.loads((data / "annotations.json").read_text())
    unique = Counter(r["Question_id"] for r in annotations)
    first_hashes = {v["sha256"] for v in m1.values()}
    used_ids = {r["Question_id"] for r in first + second}
    used_paths = {r["Image"] for r in first + second}
    used_hashes = first_hashes | {v["sha256"] for v in m2.values()}
    before = Counter(tuple(r["task_key"]) for r in second)
    replacements = []
    backup = data / "batch_2.before_byte_dedup.json"
    if not backup.exists():
        mme.dump(backup, second)

    def rank(old, index):
        return hashlib.sha256(f"{seed}:{old['source_index']}:{index}".encode()).digest()

    def eligible_source(source, key):
        return (list(mme.task_key(source)) == key and source["Image"] not in used_paths
                and source["Question_id"] not in used_ids and unique[source["Question_id"]] == 1)

    def install(old, index, source, image_bytes, sha, shard, image_index=None, reason="Duplicate-byte replacement"):
        sid = f"mme-{index:05d}"
        old_meta = m2[old["sample_id"]]
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            mime = Image.MIME[image.format]
            image.verify()
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
        path = data / "images" / (sid + suffix)
        path.write_bytes(image_bytes)
        second[second.index(old)] = {**source, "source_index": index,
                                    "task_key": old["task_key"], "sample_id": sid}
        del m2[old["sample_id"]]
        m2[sid] = {"path": str(path.relative_to(data)), "width": width, "height": height,
                   "mime": mime, "bytes": len(image_bytes), "sha256": sha, "source_shard": shard,
                   "source_index": index, "source_image_index": index if image_index is None else image_index}
        used_ids.add(source["Question_id"])
        used_paths.add(source["Image"])
        used_hashes.add(sha)
        replacements.append({"batch": 2, "old_sample_id": old["sample_id"], "new_sample_id": sid,
             "task_key": old["task_key"], "old_sha256": old_meta["sha256"], "new_sha256": sha,
             "seed": seed, "source_shard": shard, "reason": reason})

    def replace(old, allow_augment=True):
        key = old["task_key"]
        available = [r for r in annotations if eligible_source(r, key)]
        if available:
            shards = [m2[old["sample_id"]]["source_shard"]] + sorted({m2[r["sample_id"]]["source_shard"]
                         for r in second if r["task_key"] == key})
            for name in dict.fromkeys(shards):
                print(f"Searching {name} for an unused image in {key}", flush=True)
                path = data / "shards" / Path(name).name
                mme.download(f"https://huggingface.co/datasets/{mme.REPO}/resolve/{mme.REVISION}/{name}", path)
                parquet = pq.ParquetFile(path)
                best = None
                for chunk in parquet.iter_batches(batch_size=4):
                    for row in chunk.to_pylist():
                        index = row["index"]
                        source = annotations[index]
                        if not eligible_source(source, key):
                            continue
                        score = rank(old, index)
                        if best and score >= best[0]:
                            continue
                        assert (row["question"], row["multi-choice options"], row["answer"]) == (
                            source["Text"], source["Answer choices"], source["Ground truth"])
                        image_bytes = base64.b64decode(row["bytes"], validate=True)
                        sha = hashlib.sha256(image_bytes).hexdigest()
                        if sha not in used_hashes:
                            best = (score, index, source, image_bytes, sha)
                parquet.close()
                path.unlink()
                if best:
                    _, index, source, image_bytes, sha = best
                    install(old, index, source, image_bytes, sha, name)
                    return
        if not allow_augment:
            raise RuntimeError(f"No free native image found for {key}")
        # A scarce task can have all its images already assigned to other tasks.
        # Move an owner in an abundant task to a new image, then reuse its freed
        # native image for the scarce task's canonical question on that image.
        relevant = {r["Image"]: (i, r) for i, r in enumerate(annotations)
                    if list(mme.task_key(r)) == key and unique[r["Question_id"]] == 1}
        owners = [r for r in second if r["Image"] in relevant and r["task_key"] != key
                  and m2[r["sample_id"]]["sha256"] not in first_hashes]
        def availability(owner):
            return len({r["Image"] for r in annotations if eligible_source(r, owner["task_key"])})
        owners.sort(key=lambda r: (-availability(r), rank(old, r["source_index"])))
        if not owners or not availability(owners[0]):
            raise RuntimeError(f"No feasible image reassignment for {key}")
        owner = owners[0]
        freed = dict(m2[owner["sample_id"]])
        image_bytes = (data / freed["path"]).read_bytes()
        index, source = relevant[owner["Image"]]
        assert source["Question_id"] not in used_ids
        replace(owner, allow_augment=False)
        assert freed["sha256"] not in {v["sha256"] for sid, v in m2.items() if sid != old["sample_id"]}
        install(old, index, source, image_bytes, freed["sha256"], freed["source_shard"],
                image_index=freed.get("source_image_index", freed["source_index"]),
                reason="Same-task replacement using native image freed by an abundant-task reassignment")

    duplicates = [r["sample_id"] for r in second if m2[r["sample_id"]]["sha256"] in first_hashes]
    for sid in duplicates:
        replace(next(r for r in second if r["sample_id"] == sid))
    assert Counter(tuple(r["task_key"]) for r in second) == before
    assert len({r["Question_id"] for r in first + second}) == 2000
    assert len({r["Image"] for r in first + second}) == 2000
    assert len({v["sha256"] for v in m2.values()}) == 1000
    assert not first_hashes & {v["sha256"] for v in m2.values()}
    mme.dump(data / "batch_2.json", second)
    mme.dump(data / "images_batch_2.json", m2)
    audit = json.loads((data / "sampling_audit.json").read_text())
    audit.setdefault("byte_duplicate_replacements", []).extend(replacements)
    audit.update(verified_cross_batch_image_sha256_overlap=0,
                 image_disjointness_basis="Native image SHA-256 hashes and annotation paths")
    mme.dump(data / "sampling_audit.json", audit)
    print(json.dumps({"replacements": replacements, "cross_batch_image_hash_overlap": 0}, indent=2), flush=True)


if __name__ == "__main__":
    repair()
