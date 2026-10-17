"""Dataset loading: VisualOverload here, TreeBench via ``treebench.py`` (same on-disk layout).

The HF dataset stores the full ~4K painting with every one of its ~18 questions
(~9.4 GB decoded). ``prepare`` extracts each painting to disk once and writes a
lightweight ``questions.jsonl`` index, so runs load instantly and every pipeline
(including tool-using agents that crop / detect on files) works from image paths.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

HF_DATASET = "paulgavrikov/visualoverload"
DATASETS = ("visualoverload", "treebench")
CATEGORIES = ("activity", "attributes", "counting", "ocr", "reasoning", "scene")  # VisualOverload
DIFFICULTIES = ("easy", "medium", "hard")


def default_data_dir(dataset: str = "visualoverload") -> Path:
    return Path(os.environ.get("VLMBENCH_DATA_DIR", "data")) / dataset


@dataclass(frozen=True)
class Sample:
    question_id: str
    image_path: str
    question: str
    question_type: str  # "choice" | "counting" | "ocr"
    options: tuple[str, ...]  # empty for freeform questions
    difficulty: str
    category: str
    default_prompt: str
    answer: str | None = None  # public ground truth (TreeBench); None when private (VisualOverload)
    target_boxes: tuple[tuple[float, ...], ...] = ()  # evidence boxes [x1, y1, x2, y2], if released

    @property
    def image_id(self) -> str:
        return Path(self.image_path).stem

    def meta(self) -> dict:
        d = asdict(self)
        d["options"] = list(self.options)
        d["target_boxes"] = [list(b) for b in self.target_boxes]
        return d


def _parse_options(raw) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return tuple(ast.literal_eval(raw))  # stored as a stringified Python list


def prepare(data_dir: Path | None = None, force: bool = False, dataset: str = "visualoverload") -> Path:
    """Download the dataset from the HF hub and extract images + question index."""
    if dataset == "treebench":
        from . import treebench
        return treebench.prepare(Path(data_dir or default_data_dir(dataset)), force=force)
    if dataset != "visualoverload":
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {DATASETS}")

    from datasets import Image as HFImage
    from datasets import load_dataset
    from tqdm import tqdm

    data_dir = Path(data_dir or default_data_dir())
    index = data_dir / "questions.jsonl"
    if index.exists() and not force:
        log.info("Dataset already prepared at %s", data_dir)
        return data_dir

    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split="test").cast_column("image", HFImage(decode=False))

    tmp = index.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w") as f:
        for row in tqdm(ds, desc="Extracting VisualOverload"):
            img = row["image"]
            name = Path(img.get("path") or f"{row['question_id']}.jpg").name
            dest = img_dir / name
            if not dest.exists():
                dest.write_bytes(img["bytes"])
            elif dest.stat().st_size != len(img["bytes"]):
                raise RuntimeError(f"Two different images share the file name {name}")
            rec = {
                "question_id": row["question_id"],
                "image": f"images/{name}",
                "question": row["question"],
                "question_type": row["question_type"],
                "options": list(_parse_options(row["options"])),
                "difficulty": row["difficulty"],
                "category": row["category"],
                "default_prompt": row["default_prompt"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(index)
    log.info("Prepared %d questions over %d images in %s", n, len(list(img_dir.iterdir())), data_dir)
    return data_dir


def load_samples(
    data_dir: Path | None = None,
    categories: Iterable[str] | None = None,
    difficulties: Iterable[str] | None = None,
    question_ids: Iterable[str] | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    dataset: str = "visualoverload",
) -> list[Sample]:
    """Load (a filtered subset of) the benchmark. Prepares the dataset on first use."""
    data_dir = Path(data_dir or default_data_dir(dataset))
    if not (data_dir / "questions.jsonl").exists():
        prepare(data_dir, dataset=dataset)

    cats = set(categories) if categories else None
    diffs = set(difficulties) if difficulties else None
    qids = set(question_ids) if question_ids else None

    samples = []
    with (data_dir / "questions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if cats and r["category"] not in cats:
                continue
            if diffs and r["difficulty"] not in diffs:
                continue
            if qids and r["question_id"] not in qids:
                continue
            samples.append(Sample(
                question_id=r["question_id"],
                image_path=str((data_dir / r["image"]).resolve()),
                question=r["question"],
                question_type=r["question_type"],
                options=tuple(r["options"]),
                difficulty=r["difficulty"],
                category=r["category"],
                default_prompt=r["default_prompt"],
                answer=r.get("answer"),
                target_boxes=tuple(tuple(b) for b in r.get("target_boxes") or ()),
            ))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return samples
