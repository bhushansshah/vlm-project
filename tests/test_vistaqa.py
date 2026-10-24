import json
from collections import Counter

import pytest

from vlmbench.datasets import Selection
from vlmbench.judge import Judge, judge_run, parse_judgment
from vlmbench.models import load_model
from vlmbench.report import build_report
from vlmbench.runner import RunConfig, Runner, read_records
from vlmbench.types import Completion


def test_balanced_sampling_is_reproducible(vistaqa):
    a = vistaqa.load(Selection(per_category=2, seed=0))
    b = vistaqa.load(Selection(per_category=2, seed=0))
    c = vistaqa.load(Selection(per_category=2, seed=1))
    assert Counter(s.category for s in a) == {"counting": 2, "ocr": 2, "spatial": 2}
    assert [s.question_id for s in a] == [s.question_id for s in b]
    assert [s.question_id for s in a] != [s.question_id for s in c]


def test_filters_and_ids(vistaqa):
    assert {s.meta["hallucination"] for s in vistaqa.load(Selection(filters={"hallucination": ["1"]}))} == {1}
    assert [s.question_id for s in vistaqa.load(Selection(question_ids=["3", "1"]))] == ["1", "3"]


def test_prompts(vistaqa):
    s = vistaqa.load()[0]
    text = vistaqa.build_messages("default", s)[0].content[1].text
    assert text.startswith("q1?") and "say so" in text
    assert vistaqa.build_messages("question_only", s)[0].content[1].text == "q1?"
    jp = vistaqa.judge_prompt(s, "my answer")
    assert "Ground-truth answer: a1" in jp and "Predicted answer: my answer" in jp and "Domain: av" in jp


@pytest.mark.parametrize("raw,expected", [
    ('{"correct": 1, "reason": "ok"}', 1), ('{"correct": "0"}', 0), ('Sure! {"correct": true}', 1),
    ("```json\n{\"correct\": 0, \"reason\": \"no\"}\n```", 0), ("I think it is right", None),
    ("{\"correct\": 0, \"reason\": \"x\"}\n{\"correct\": 1, \"reason\": \"changed my mind\"}", 1),
])
def test_parse_judgment(raw, expected):
    assert parse_judgment(raw)[0] == expected


class FakeJudge:
    """Says an answer is correct iff it equals the reference answer embedded in the prompt."""
    supports_concurrency = True
    calls = 0

    def generate(self, messages, config):
        FakeJudge.calls += 1
        prompt = messages[0].content[0].text
        gt = prompt.split("Ground-truth answer: ")[1].split("\n")[0]
        pred = prompt.split("Predicted answer: ")[1].split("\n")[0]
        return Completion(text=json.dumps({"correct": int(gt == pred), "reason": "fake"}))


class Oracle:
    """A backend answering correctly on counting questions and wrongly otherwise."""
    supports_concurrency = False
    model = "oracle"

    def __init__(self, dataset):
        self.answers = {s.question: s.answer for s in dataset.load()}
        self.categories = {s.question: s.category for s in dataset.load()}

    def describe(self):
        return {"provider": "oracle"}

    def generate(self, messages, config):
        q = messages[0].content[1].text.split("\n")[0]
        return Completion(text=self.answers[q] if self.categories[q] == "counting" else "no idea")

    def generate_batch(self, convs, config):
        return [self.generate(m, config) for m in convs]


def test_run_judge_report(tmp_path, vistaqa):
    cfg = RunConfig(model="oracle:x", dataset="vistaqa", data_dir=str(vistaqa.data_dir),
                    output_dir=str(tmp_path / "runs"), run_name="oracle")
    Runner(Oracle(vistaqa), vistaqa, cfg).run(vistaqa.load())
    preds = json.loads((cfg.run_dir() / "predictions.json").read_text())
    assert preds[0]["prediction"]["masks"] == [] and isinstance(preds[0]["image"]["image_id"], int)

    judge = Judge(FakeJudge(), "fake-judge")
    stats = judge_run(cfg.run_dir(), judge)
    assert stats == {"answers": 12, "judged_now": 12, "judge_errors": 0}
    calls = FakeJudge.calls
    assert judge_run(cfg.run_dir(), judge)["judged_now"] == 0 and FakeJudge.calls == calls  # cached

    md, data = build_report([cfg.run_dir()])
    d = data["oracle"]
    assert d["graded"] == 12 and d["correct"] == 4
    assert d["by"]["category"]["counting"]["accuracy"] == 100.0
    assert d["by"]["category"]["ocr"]["accuracy"] == 0.0
    assert set(d["by"]["hallucination"]) == {"0", "1"}
    assert "By task type" in md and "By domain" in md
    assert set(d["by"]["instances"]) == {"none", "one"}


def test_unanswered_questions_count_as_wrong_without_judge_call(tmp_path, vistaqa):
    from conftest import Flaky
    cfg = RunConfig(model="flaky:x", dataset="vistaqa", data_dir=str(vistaqa.data_dir),
                    output_dir=str(tmp_path / "runs"), run_name="broken", max_retries=0)
    Runner(Flaky(fail_times=99), vistaqa, cfg).run(vistaqa.load()[:2])
    FakeJudge.calls = 0
    judge_run(cfg.run_dir(), Judge(FakeJudge(), "fake-judge"))
    assert FakeJudge.calls == 0
    assert all(j["correct"] == 0 for j in read_records(cfg.run_dir() / "judgments.jsonl").values())


def test_image_file_falls_back_to_same_stem(tmp_path):
    from vlmbench.datasets.vistaqa import _image_file
    (tmp_path / "1079.json").write_text("{}")
    (tmp_path / "1079.jpg").write_bytes(b"x")  # JSON says .png, file is .jpg (VistaQA v1.1)
    assert _image_file(tmp_path / "1079.json", "1079.png") == "1079.jpg"
    with pytest.raises(FileNotFoundError):
        _image_file(tmp_path / "1079.json".replace("1079", "1080"), "1080.png")


def test_second_judge_keeps_its_own_file(tmp_path, vistaqa):
    cfg = RunConfig(model="oracle:x", dataset="vistaqa", data_dir=str(vistaqa.data_dir),
                    output_dir=str(tmp_path / "runs"), run_name="two-judges")
    Runner(Oracle(vistaqa), vistaqa, cfg).run(vistaqa.load())
    judge_run(cfg.run_dir(), Judge(FakeJudge(), "judge-a"))
    judge_run(cfg.run_dir(), Judge(FakeJudge(), "judge-b"), filename="judgments_b.jsonl")
    a = read_records(cfg.run_dir() / "judgments.jsonl")
    b = read_records(cfg.run_dir() / "judgments_b.jsonl")
    assert {j["judge_model"] for j in a.values()} == {"judge-a"}
    assert {j["judge_model"] for j in b.values()} == {"judge-b"}
    assert build_report([cfg.run_dir()], "judgments_b.jsonl")[1]["two-judges"]["graded"] == 12
