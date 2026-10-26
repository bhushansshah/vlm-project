"""TreeBench (Wang et al., arXiv 2507.07999; HF ``HaochenWang/TreeBench``): 405 multiple-choice
questions on dense SA-1B photos (~2-7 MP), in 10 categories split into Perception and Reasoning.
Answers and the target objects' boxes are public, so answers are graded locally by rule (no LLM
judge) and the boxes are kept for evidence analysis.

The HF release is one TSV with base64 images; ``prepare`` extracts it into ``images/`` plus a
``questions.jsonl`` index. Two deliberate differences from the official ``inference_treebench.py``:

* "Perception/OCR" questions print the question and options *on the image*; the text question
  only says "Recognize the question and options in the image and answer it.". The official
  script means to withhold the options for these, but compares ``category == "OCR"`` against
  values like "Perception/OCR", so it always appends them. We withhold them, as intended; the
  options are still stored so answers can be read.
* Options come from the "multi-choice options" field (what the official prompt uses). The
  per-letter A-K columns drop or merge options in 11 rows, making 4 answers unselectable.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..types import ImagePart, Message
from .base import Dataset, Sample
from .choice import LETTERS, extract_choice_letter

HF_DATASET = "HaochenWang/TreeBench"
HF_FILE = "TreeBench.tsv"
DIRECT_ANSWER = "Answer with the option's letter from the given choices directly."
CATEGORIES = (
    "Perception/Attributes", "Perception/Material", "Perception/OCR", "Perception/Object Retrieval",
    "Perception/Physical State", "Reasoning/Comparison", "Reasoning/Contact and Occlusion",
    "Reasoning/Ordering", "Reasoning/Perspective Transform", "Reasoning/Spatial Containment",
)


def _parse_option_text(text: str) -> list[str]:
    """Split "A. x\\nB. y ..." into ["x", "y", ...], requiring the letters in order. Handles the
    release's variants: all options on one line ("A. x B. y"), and "A x" without the period."""
    starts, pos = [], 0
    for letter in LETTERS:
        m = (re.compile(rf"(?:^|(?<=\s)){letter}[.)]\s").search(text, pos)
             or re.compile(rf"(?:^|(?<=\n))\s*{letter}\s").search(text, pos))
        if not m:
            break
        starts.append((m.start(), m.end()))
        pos = m.end()
    return [text[s_end:(starts[i + 1][0] if i + 1 < len(starts) else len(text))].strip()
            for i, (_, s_end) in enumerate(starts)]


def _options(row: dict) -> list[str]:
    opts = _parse_option_text(row.get("multi-choice options") or "")
    return opts or [row[L].strip() for L in LETTERS if (row.get(L) or "").strip()]


def _prompt(question: str, options: list[str], options_in_image: bool) -> str:
    if options_in_image:
        return f"{question}\n{DIRECT_ANSWER}"
    listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\nOptions:\n{listed}\n{DIRECT_ANSWER}"


def default_prompt(s: Sample) -> list[Message]:
    """Image first, then the question, the options and "answer with the letter directly"."""
    return [Message.user(ImagePart(s.image_path), s.meta["default_prompt"])]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for a proportion, in percent."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


class TreeBench(Dataset):
    name = "treebench"
    categories = CATEGORIES
    prompts = {"default": default_prompt}
    report_groups = ("group",)
    grades_locally = True

    # ---- data
    def prepare(self, force: bool = False, tsv_path: Path | None = None) -> Path:
        """Download the TSV from the HF hub (or use ``tsv_path``) and extract images + an index."""
        index = self.data_dir / "questions.jsonl"
        if index.exists() and not force:
            return self.data_dir
        if tsv_path is None:
            from huggingface_hub import hf_hub_download
            tsv_path = Path(hf_hub_download(HF_DATASET, HF_FILE, repo_type="dataset"))

        img_dir = self.data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        csv.field_size_limit(sys.maxsize)
        tmp = index.with_suffix(".jsonl.tmp")
        with open(tsv_path, encoding="utf-8", newline="") as src, tmp.open("w") as f:
            for row in csv.DictReader(src, delimiter="\t"):
                qid = f"tb{int(row['index']):03d}"
                (img_dir / f"{qid}.jpg").write_bytes(base64.b64decode(row["image"]))
                category = row["l2-category"]
                options = _options(row)
                rec = {
                    "question_id": qid,
                    "image": f"images/{qid}.jpg",
                    "question": row["question"],
                    "options": options,
                    "category": category,
                    "default_prompt": _prompt(row["question"], options, category.endswith("/OCR")),
                    "answer": row["answer"].strip(),
                    "target_boxes": json.loads(row["target_instances"]),  # [x1, y1, x2, y2] in pixels
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(index)
        return self.data_dir

    def load_all(self) -> list[Sample]:
        samples = []
        with (self.data_dir / "questions.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                samples.append(Sample(
                    question_id=r["question_id"],
                    image_path=str((self.data_dir / r["image"]).resolve()),
                    question=r["question"],
                    category=r["category"],
                    answer=r["answer"],
                    meta={
                        "group": r["category"].split("/")[0],
                        "options": tuple(r["options"]),
                        "default_prompt": r["default_prompt"],
                        "target_boxes": tuple(tuple(b) for b in r.get("target_boxes") or ()),
                    },
                ))
        return samples

    # ---- grading
    def postprocess(self, sample: Sample, response: str | None) -> dict[str, Any]:
        letter = extract_choice_letter(response, sample.meta["options"])
        return {"extracted": letter, "parsed": letter is not None}

    def grade(self, sample: Sample, record: dict) -> bool:
        """Right only if the reply commits to the correct letter. Errors, unreadable replies and
        replies cut off by the token limit (whose last line may be an option list) are wrong."""
        if record.get("error") or record.get("finish_reason") == "length":
            return False
        return extract_choice_letter(record.get("response"), sample.meta["options"]) == sample.answer

    def summarize(self, samples: dict[str, Sample], records: dict[str, dict]) -> dict[str, Any]:
        def block(qids: list[str]) -> dict[str, Any]:
            k = sum(self.grade(samples[q], records[q]) for q in qids)
            return {"n": len(qids), "correct": k, "acc": round(100 * k / len(qids), 2), "ci95": wilson(k, len(qids))}

        by_cat: dict[str, list[str]] = defaultdict(list)
        by_group: dict[str, list[str]] = defaultdict(list)
        for q in records:
            by_cat[samples[q].category].append(q)
            by_group[samples[q].meta["group"]].append(q)
        letters = Counter(extract_choice_letter(r.get("response"), samples[q].meta["options"]) or "none"
                          for q, r in records.items())
        return {
            "accuracy": {
                "overall": block(list(records)) if records else None,
                "per_group": {g: block(qs) for g, qs in sorted(by_group.items())},
                "per_category": {c: block(qs) for c, qs in sorted(by_cat.items())},
            },
            "answer_letters": dict(sorted(letters.items())),
            "cut_off": sum(1 for r in records.values() if r.get("finish_reason") == "length"),
            "note": "Graded locally against TreeBench's public answers; `vlmbench report` works without a judge.",
        }
