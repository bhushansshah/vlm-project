import json

import pytest

from vlmbench.datasets.visualoverload_extract import extract, normalize_response
from vlmbench.models import load_model
from vlmbench.runner import RunConfig, Runner
from vlmbench.types import ImagePart


@pytest.mark.parametrize("resp,expected", [
    ("B", "B"), ("B.", "B"), ("Answer: A", "A"), ("The answer is B. no", "B"),
    ("<answer>A</answer>", "A"), ("I think B. no is right", "B"),
])
def test_extract_choice(resp, expected):
    assert extract(resp, "choice", ("yes", "no")).value == expected


@pytest.mark.parametrize("resp,expected,parsed", [
    ("7", 7, True), ("seven.", 7, True), ("There are 3 cups, maybe 4", 4, True),
    ("<answer>12</answer>", 12, True), ("too many to count", 0, False), ("", 0, False),
])
def test_extract_counting(resp, expected, parsed):
    ex = extract(resp, "counting")
    assert (ex.value, ex.parsed) == (expected, parsed)


def test_extract_ocr():
    assert extract('The text reads "Anno 1650."', "ocr").value == "anno 1650"
    assert extract("VIVAT", "ocr").value == "uiuat"  # server maps v -> u
    assert extract("**1557**", "ocr").value == "1557"


@pytest.mark.parametrize("resp,expected", [
    ("**1557**", "1557"), ("`Anno`", "Anno"), ("*B*", "B"), ("**B. yes**", "B. yes"), ("plain", "plain"),
    ("**a** and **b**", "**a** and **b**"),
])
def test_normalize_markdown(resp, expected):
    assert normalize_response(resp) == expected


def test_default_prompt_is_paper_prompt(visualoverload):
    s = visualoverload.load()[0]
    msgs = visualoverload.build_messages("default", s)
    img, text = msgs[0].content
    assert isinstance(img, ImagePart) and text.text == s.meta["default_prompt"]


def test_run_writes_submission(tmp_path, visualoverload):
    cfg = RunConfig(model="dummy:random", dataset="visualoverload", output_dir=str(tmp_path / "runs"),
                    data_dir=str(visualoverload.data_dir))
    summary = Runner(load_model(cfg.model), visualoverload, cfg).run(visualoverload.load())
    sub = json.loads((cfg.run_dir() / "submission.json").read_text())
    assert {s["question_id"] for s in sub} == {"q1", "q2", "q3"}
    assert summary["parse_rate"] == 1.0
