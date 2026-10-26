import base64
import csv
import json

import pytest

from vlmbench.datasets import get_dataset
from vlmbench.datasets.choice import extract_choice_letter
from vlmbench.datasets.treebench import TreeBench
from vlmbench.judge import judge_run
from vlmbench.models import load_model
from vlmbench.report import build_report
from vlmbench.runner import RunConfig, Runner, read_records


def _tsv(path, image_path):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    cols = ["index", "image", "question", "answer", "multi-choice options", "category", "l2-category",
            "target_instances", *"ABCDEFGHIJK"]
    rows = [
        {"index": "0", "image": b64, "question": "What color is the cup?", "answer": "B",
         "multi-choice options": "A. Red\nB. Blue", "category": "Perception/Attributes",
         "l2-category": "Perception/Attributes", "target_instances": "[[1, 2, 30, 40]]", "A": "Red", "B": "Blue"},
        {"index": "1", "image": b64, "question": "Recognize the question and options in the image and answer it.",
         "answer": "A", "multi-choice options": "A. Left\nB. Right\nC. Front", "category": "Perception/OCR",
         "l2-category": "Perception/OCR", "target_instances": "[[5, 5, 50, 20]]"},
        # the release has rows whose A-K columns drop an option; "multi-choice options" is right
        {"index": "2", "image": b64, "question": "What is he doing?", "answer": "D",
         "multi-choice options": "A. Standing. B. Lying down.\nC. Running.\nD. Sitting.",
         "category": "Reasoning/Ordering", "l2-category": "Reasoning/Ordering",
         "target_instances": "[[0, 0, 10, 10], [20, 20, 40, 40]]",
         "A": "Standing. B. Lying down.", "B": "Running.", "C": "Sitting."},
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


@pytest.fixture
def treebench(tmp_path, image_path):
    tsv = tmp_path / "TreeBench.tsv"
    _tsv(tsv, image_path)
    ds = TreeBench(tmp_path / "treebench")
    ds.prepare(tsv_path=tsv)
    return ds


def test_prepare_and_prompts(treebench):
    s = {x.question_id: x for x in treebench.load()}
    assert set(s) == {"tb000", "tb001", "tb002"}
    assert s["tb000"].answer == "B" and s["tb000"].meta["options"] == ("Red", "Blue")
    assert s["tb000"].meta["target_boxes"] == ((1, 2, 30, 40),) and s["tb000"].meta["group"] == "Perception"
    assert "B. Blue" in treebench.build_messages("default", s["tb000"])[0].content[1].text
    # OCR rows: options are printed on the image, so they stay out of the prompt (but are kept)
    assert s["tb001"].meta["options"] == ("Left", "Right", "Front")
    assert "Left" not in s["tb001"].meta["default_prompt"]
    assert s["tb002"].meta["options"] == ("Standing.", "Lying down.", "Running.", "Sitting.")
    assert get_dataset("treebench", treebench.data_dir).name == "treebench"


@pytest.mark.parametrize("resp,expected", [
    ("B", "B"), ("**C**", "C"), ("(D)", "D"), ("<answer>B</answer>", "B"),
    # the VisualOverload judge reads these as "A" (first letter anywhere in the text)
    ("Above the cart is an umbrella; the top stripe is yellow.\n\nTherefore, the correct option is B.\n\nB", "B"),
    ("Let's call this Person A. ... the direction is **Front right**.\n\nB", "B"),
    ("Looking closely, the answer is **C**.", "C"), ("Correct Option: **B**", "B"),
    ("A detailed look shows nothing conclusive.", None),
    ("The answer is F", None),  # not a valid option for 4 choices
    # letter + that option's own text on the last line (Gemma with thinking)
    ("... located in front of her and to her left.\n\nA. Front left", "A"),
    ("Therefore, the correct option is:\n**D. Back right**", "D"),
    ("so it is on her left.\n\nB. Front left", None),  # letter and text disagree: don't guess
])
def test_extract_choice_letter(resp, expected):
    assert extract_choice_letter(resp, ("Front left", "Front right", "Back left", "Back right")) == expected


def test_grade_counts_cut_off_and_errors_as_wrong(treebench):
    s = {x.question_id: x for x in treebench.load()}["tb000"]
    assert treebench.grade(s, {"response": "B", "finish_reason": "stop", "error": None})
    assert not treebench.grade(s, {"response": "A", "finish_reason": "stop", "error": None})
    # a cut-off reply's last line can be an option list, not a choice
    assert not treebench.grade(s, {"response": "Options:\nA. Red\nB. Blue", "finish_reason": "length",
                                   "error": None})
    assert not treebench.grade(s, {"response": None, "finish_reason": None, "error": "boom"})


def test_run_summary_and_report_without_judge(treebench, tmp_path):
    cfg = RunConfig(model="dummy:echo", dataset="treebench", data_dir=str(treebench.data_dir),
                    output_dir=str(tmp_path / "runs"), concurrency=1)
    summary = Runner(load_model(cfg.model), treebench, cfg).run(treebench.load())
    assert cfg.resolved_run_name().startswith("treebench__")
    assert summary["accuracy"]["overall"]["n"] == 3
    assert set(summary["accuracy"]["per_group"]) == {"Perception", "Reasoning"}
    recs = read_records(cfg.run_dir() / "responses.jsonl")
    assert all("extracted" in r and "parsed" in r for r in recs.values())
    md, data = build_report([cfg.run_dir()])  # no judgments.jsonl needed
    run = data[cfg.resolved_run_name()]
    assert run["graded"] == 3 and run["n_ungraded"] == 0 and "group" in run["by"]
    assert "By group" in md
    with pytest.raises(SystemExit):
        judge_run(cfg.run_dir(), judge=None)
