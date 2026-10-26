"""VistaQA (Nasr Azadani et al., NeurIPS 2026 D&B): 1,157 free-form VQA samples over six task
types and six visual domains; 314 are "hallucination" samples asking about things that are
not in the image. Reference answers are public, so we can grade locally.

The benchmark also asks for segmentation masks (scored jointly with the answer as GROVE).
The base-VLM pipeline only produces text, so we report the paper's text-accuracy metric
(the 𝒯 column of their Table 2): an LLM judge decides if each answer is correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import ImagePart, Message
from .base import Dataset, Sample

HF_DATASET = "mozhina/VistaQA"

# Verbatim from the official evaluator (src/config.py in
# https://github.com/VistaQA/VistaQA-Evaluation, MIT License, (c) 2026 VistaQA).
JUDGE_PROMPT_TEMPLATE = """You are an evaluator for a visual question answering task. Your task is to determine whether the predicted answer is correct with respect to the ground-truth answer for the given question.

Evaluation rules:
1. Accept minor wording differences, paraphrases, singular/plural variation, and equivalent expressions.
2. Accept answers that are semantically equivalent to the ground-truth answer.
3. Reject answers that are incomplete, overly vague, or refer to the wrong object, attribute, count number, text, or relation.
4. For counting questions, the numeric value must match exactly unless the reference explicitly allows a range.
5. For OCR questions, ignore capitalization and minor punctuation differences, but do not ignore incorrect characters or different words.
6. For hallucination questions, accept only answers that clearly indicate the target does not exist or is not present or the answer cannot be determined from the image.
7. Be strict: the predicted answer should mean the same thing as the ground-truth answer in the context of the question.

Return your decision in JSON with the following fields:
{{
  "correct": 0 or 1,
  "reason": "short explanation"
}}

Question: {question}
Ground-truth answer: {gt_answer}
Predicted answer: {pred_answer}
Task type: {task_type}
Domain: {domain}
"""

# The paper does not publish its answer prompt, only that models get "standardized output
# format instructions". Ours asks for a short answer and allows "not present" answers, since
# ~27% of questions are about things missing from the image.
ANSWER_INSTRUCTION = (
    "Answer the question based on the image in one short sentence. "
    "If the object or information the question asks about is not in the image, "
    "or the question cannot be answered from the image, say so."
)


def _image_file(json_path: Path, file_name: str) -> str:
    """The image named in the JSON, or the one sharing its stem: in v1.1, 1079.json says
    "1079.png" but the file is 1079.jpg."""
    if (json_path.parent / file_name).exists():
        return file_name
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if json_path.with_suffix(ext).exists():
            return json_path.with_suffix(ext).name
    raise FileNotFoundError(f"No image for {json_path.name} (expected {file_name})")


def instance_group(n: int | None) -> str:
    """How many objects the answer is about, grouped like the official evaluator's breakdown.
    "none" is exactly the set of hallucination questions."""
    if not n:
        return "none"
    return "one" if n == 1 else "many"


def default_prompt(s: Sample) -> list[Message]:
    return [Message.user(ImagePart(s.image_path), f"{s.question}\n\n{ANSWER_INSTRUCTION}")]


def question_only(s: Sample) -> list[Message]:
    """Ablation: the bare question, no instructions (tests the effect of the 'say so' hint)."""
    return [Message.user(ImagePart(s.image_path), s.question)]


class VistaQA(Dataset):
    name = "vistaqa"
    categories = ("attribute", "comparison", "counting", "identification", "ocr", "spatial")
    prompts = {"default": default_prompt, "question_only": question_only}
    report_groups = ("domain", "hallucination", "instances")

    # ---- data
    def prepare(self, force: bool = False) -> Path:
        """Download the HF repo (one image + one JSON per sample) and write a compact index
        without the segmentation masks, which the text pipeline does not need."""
        from huggingface_hub import snapshot_download

        index = self.data_dir / "questions.jsonl"
        if index.exists() and not force:
            return self.data_dir
        raw = self.data_dir / "raw"
        snapshot_download(HF_DATASET, repo_type="dataset", local_dir=raw, max_workers=8)

        records = []
        for path in (raw / "VistaQA").glob("*.json"):
            r = json.loads(path.read_text())
            img = r["image"]
            records.append({
                "question_id": str(img["image_id"]),
                "image": f"raw/VistaQA/{_image_file(path, img['file_name'])}",
                "width": img["width"],
                "height": img["height"],
                "question": r["question"],
                "answer": r["answer"],
                "task_type": r["task_type"],
                "domain": r["task_domain"],
                "hallucination": int(r["hallucination"]),
                "num_instances": r.get("num_instances"),
            })
        records.sort(key=lambda r: int(r["question_id"]))
        tmp = index.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
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
                    category=r["task_type"],
                    answer=r["answer"],
                    meta={
                        "domain": r["domain"],
                        "hallucination": r["hallucination"],
                        "num_instances": r["num_instances"],
                        "instances": instance_group(r["num_instances"]),
                        "width": r["width"],
                        "height": r["height"],
                    },
                ))
        return samples

    # ---- hooks
    def write_outputs(self, run_dir: Path, samples: dict[str, Sample], records: dict[str, dict]) -> None:
        """predictions.json in the official evaluator's input format (text only, no masks)."""
        preds = [
            {
                "image": {"image_id": int(qid), "width": samples[qid].meta["width"],
                          "height": samples[qid].meta["height"]},
                "prediction": {"text": r["response"] or "", "masks": []},
            }
            for qid, r in records.items()
        ]
        (run_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False))

    def judge_prompt(self, sample: Sample, prediction: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            question=sample.question,
            gt_answer=sample.answer,
            pred_answer=prediction,
            task_type=sample.category,
            domain=sample.meta["domain"],
        )

    def summarize(self, samples: dict[str, Sample], records: dict[str, dict]) -> dict[str, Any]:
        return {"note": "Grade answers with `vlmbench judge <run_dir>`, then `vlmbench report <run_dir>...`."}
