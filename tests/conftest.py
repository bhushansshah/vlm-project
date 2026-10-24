import json

import pytest
from PIL import Image

from vlmbench.datasets import VistaQA, VisualOverload
from vlmbench.types import Completion


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (400, 200), "red").save(p, quality=95)
    return str(p)


def _write_index(data_dir, rows):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def vistaqa(tmp_path, image_path):
    """A tiny VistaQA: 3 task types x 4 questions, every 4th one a hallucination question."""
    rows = []
    for i, task in enumerate(["counting", "ocr", "spatial"] * 4):
        rows.append({
            "question_id": str(i + 1), "image": image_path, "width": 400, "height": 200,
            "question": f"q{i + 1}?", "answer": f"a{i + 1}", "task_type": task,
            "domain": ["av", "math"][i % 2], "hallucination": int(i % 4 == 0), "num_instances": int(i % 4 != 0),
        })
    _write_index(tmp_path / "vistaqa", rows)
    return VistaQA(tmp_path / "vistaqa")


@pytest.fixture
def visualoverload(tmp_path, image_path):
    rows = [
        {"question_id": "q1", "image": image_path, "question": "Is it day?", "question_type": "choice",
         "options": ["yes", "no"], "difficulty": "easy", "category": "scene",
         "default_prompt": "Is it day? Options:\nA. yes\nB. no\nAnswer with the option's letter from the given choices directly."},
        {"question_id": "q2", "image": image_path, "question": "How many cups?", "question_type": "counting",
         "options": [], "difficulty": "hard", "category": "counting",
         "default_prompt": "How many cups? Answer with a number directly."},
        {"question_id": "q3", "image": image_path, "question": "What year?", "question_type": "ocr",
         "options": [], "difficulty": "medium", "category": "ocr", "default_prompt": "What year? Answer directly."},
    ]
    _write_index(tmp_path / "visualoverload", rows)
    return VisualOverload(tmp_path / "visualoverload")


class Flaky:
    """A backend that fails `fail_times` times before answering."""
    provider = "flaky"
    supports_concurrency = False

    def __init__(self, fail_times, answer="A"):
        self.fail_times, self.answer, self.calls, self.model = fail_times, answer, 0, "flaky"

    def describe(self):
        return {"provider": "flaky"}

    def generate_batch(self, convs, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return [Completion(text=self.answer) for _ in convs]

    def generate(self, msgs, config):
        return self.generate_batch([msgs], config)[0]


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr("vlmbench.retry.time.sleep", lambda s: None)
