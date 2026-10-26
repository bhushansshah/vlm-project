"""VisualOverload (Gavrikov et al., CVPR 2026): 2,720 questions on 150 dense ~4K paintings.

Ground truth is private: accuracy comes from the official evaluation server, so this
module writes ``submission.json`` and reports local parse diagnostics only.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..types import ImagePart, Message
from .base import Dataset, Sample
from .visualoverload_extract import extract, normalize_response

log = logging.getLogger(__name__)

HF_DATASET = "paulgavrikov/visualoverload"
EVAL_SERVER = "https://huggingface.co/spaces/paulgavrikov/visualoverload-submit"


def _parse_options(raw) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return list(ast.literal_eval(raw))  # stored as a stringified Python list


def format_options(options) -> str:
    return "\n".join(f"{chr(65 + i)}. {o.strip()}" for i, o in enumerate(options))


# ----------------------------------------------------------------- prompts
def default_prompt(s: Sample) -> list[Message]:
    """The authors' prompt, e.g. "<question> Options:\\nA. yes\\nB. no\\nAnswer with the
    option's letter from the given choices directly." """
    return [Message.user(ImagePart(s.image_path), s.meta["default_prompt"])]


_FORMAT = {
    "choice": "the option's letter from the given choices",
    "counting": "a single number",
    "ocr": "the exact text as written in the image",
}


def think_then_answer(s: Sample) -> list[Message]:
    """Reason first, then the final answer in <answer></answer> tags (the judge reads these tags)."""
    body = s.question
    if s.meta["question_type"] == "choice":
        body += "\nOptions:\n" + format_options(s.meta["options"])
    body += (
        "\n\nLook carefully at the relevant parts of the image and think step by step. "
        f"Then give your final answer as {_FORMAT[s.meta['question_type']]} inside <answer></answer> tags."
    )
    return [Message.user(ImagePart(s.image_path), body)]


class VisualOverload(Dataset):
    name = "visualoverload"
    categories = ("activity", "attributes", "counting", "ocr", "reasoning", "scene")
    prompts = {"default": default_prompt, "think_then_answer": think_then_answer}
    report_groups = ("difficulty",)

    # ---- data
    def prepare(self, force: bool = False) -> Path:
        """Extract each painting once (the HF parquet repeats it for every question) + an index."""
        from datasets import Image as HFImage
        from datasets import load_dataset
        from tqdm import tqdm

        index = self.data_dir / "questions.jsonl"
        if index.exists() and not force:
            return self.data_dir
        img_dir = self.data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(HF_DATASET, split="test").cast_column("image", HFImage(decode=False))

        tmp = index.with_suffix(".jsonl.tmp")
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
                    "options": _parse_options(row["options"]),
                    "difficulty": row["difficulty"],
                    "category": row["category"],
                    "default_prompt": row["default_prompt"],
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
                    meta={
                        "question_type": r["question_type"],
                        "options": tuple(r["options"]),
                        "difficulty": r["difficulty"],
                        "default_prompt": r["default_prompt"],
                    },
                ))
        return samples

    # ---- hooks
    def postprocess(self, sample: Sample, response: str | None) -> dict[str, Any]:
        if response is None:
            return {"extracted": None, "parsed": False}
        ex = extract(response, sample.meta["question_type"], sample.meta["options"])
        return {"extracted": ex.value, "parsed": ex.parsed}

    def summarize(self, samples: dict[str, Sample], records: dict[str, dict]) -> dict[str, Any]:
        parsed: dict[str, Counter] = defaultdict(Counter)
        letters: dict[str, Counter] = defaultdict(Counter)
        for qid, r in records.items():
            cat = r["category"]
            parsed[cat]["n"] += 1
            parsed[cat]["ok"] += bool(r.get("parsed"))
            if samples[qid].meta["question_type"] == "choice" and r.get("parsed"):
                letters[cat][r["extracted"]] += 1
        n = sum(c["n"] for c in parsed.values())
        return {
            "parse_rate": round(sum(c["ok"] for c in parsed.values()) / n, 4) if n else None,
            "parse_rate_per_category": {k: round(c["ok"] / c["n"], 4) for k, c in sorted(parsed.items())},
            "choice_letter_distribution": {k: dict(sorted(v.items())) for k, v in sorted(letters.items())},
            "note": f"Ground truth is private: submit submission.json to {EVAL_SERVER} for accuracy.",
        }

    def write_outputs(self, run_dir: Path, samples: dict[str, Sample], records: dict[str, dict]) -> None:
        """Responses as returned (like the authors' eval.py) with markdown emphasis stripped;
        the server applies its own answer extraction on top."""
        sub = [{"question_id": qid, "response": normalize_response(r["response"]) or ""}
               for qid, r in records.items()]
        (run_dir / "submission.json").write_text(json.dumps(sub, ensure_ascii=False))
