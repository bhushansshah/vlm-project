"""TreeBench dataset preparation (Wang et al., arXiv 2507.07999; HF ``HaochenWang/TreeBench``).

405 multiple-choice questions on dense SA-1B photos (~2-7 MP). Unlike VisualOverload, answers
and target boxes are public, so runs are scored locally. The HF release is a single TSV with
base64 images; ``prepare`` extracts it into the same ``images/`` + ``questions.jsonl`` layout
that ``data.load_samples`` reads.

Two deliberate differences from the official ``inference_treebench.py``:

* "Perception/OCR" questions print the question and options *on the image* and the text
  question only says "Recognize the question and options in the image and answer it.". The
  official script means to withhold the options for these, but compares ``category == "OCR"``
  against values like "Perception/OCR", so it always appends them. We withhold them, as the
  benchmark intends; the options are still stored so answers can be extracted.
* For OCR rows the boxes include the printed-question banner; we keep all boxes as released.
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

HF_DATASET = "HaochenWang/TreeBench"
HF_FILE = "TreeBench.tsv"
LETTERS = "ABCDEFGHIJK"
DIRECT_ANSWER = "Answer with the option's letter from the given choices directly."


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
    # "multi-choice options" is what the official script puts in the prompt, and it is the
    # only source for OCR rows. The A-K columns disagree with it at least once (a dropped
    # option), so they are only a fallback.
    opts = _parse_option_text(row.get("multi-choice options") or "")
    return opts or [row[L].strip() for L in LETTERS if (row.get(L) or "").strip()]


def _prompt(question: str, options: list[str], options_in_image: bool) -> str:
    if options_in_image:
        return f"{question}\n{DIRECT_ANSWER}"
    listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\nOptions:\n{listed}\n{DIRECT_ANSWER}"


def prepare(data_dir: Path, force: bool = False, tsv_path: Path | None = None) -> Path:
    """Download the TSV from the HF hub (or use ``tsv_path``) and extract images + index."""
    data_dir = Path(data_dir)
    index = data_dir / "questions.jsonl"
    if index.exists() and not force:
        log.info("Dataset already prepared at %s", data_dir)
        return data_dir
    if tsv_path is None:
        from huggingface_hub import hf_hub_download
        tsv_path = Path(hf_hub_download(HF_DATASET, HF_FILE, repo_type="dataset"))

    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    tmp = index.with_suffix(".jsonl.tmp")
    n = 0
    with open(tsv_path, encoding="utf-8", newline="") as src, tmp.open("w") as f:
        for row in csv.DictReader(src, delimiter="\t"):
            qid = f"tb{int(row['index']):03d}"
            (img_dir / f"{qid}.jpg").write_bytes(base64.b64decode(row["image"]))
            category = row["l2-category"]
            options = _options(row)
            in_image = category.endswith("/OCR")
            rec = {
                "question_id": qid,
                "image": f"images/{qid}.jpg",
                "question": row["question"],
                "question_type": "choice",
                "options": options,
                "difficulty": "n/a",  # TreeBench has no difficulty split
                "category": category,
                "default_prompt": _prompt(row["question"], options, in_image),
                "answer": row["answer"].strip(),
                "target_boxes": json.loads(row["target_instances"]),  # [x1, y1, x2, y2] in pixels
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(index)
    log.info("Prepared %d TreeBench questions in %s", n, data_dir)
    return data_dir
