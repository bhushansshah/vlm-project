import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vlmbench.data import Sample
from vlmbench.extract import extract
from vlmbench.images import ImagePolicy, encode_image
from vlmbench.models import load_model, parse_spec
from vlmbench.prompts import build_messages
from vlmbench.runner import RunConfig, Runner, read_records
from vlmbench.types import GenerationConfig, ImagePart, Message


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (400, 200), "red").save(p, quality=95)
    return str(p)


@pytest.fixture
def samples(image_path):
    return [
        Sample("q1", image_path, "Is it day?", "choice", ("yes", "no"), "easy", "scene",
               "Is it day? Options:\nA. yes\nB. no\nAnswer with the option's letter from the given choices directly."),
        Sample("q2", image_path, "How many cups?", "counting", (), "hard", "counting",
               "How many cups? Answer with a number directly."),
        Sample("q3", image_path, "What year?", "ocr", (), "medium", "ocr", "What year? Answer directly."),
    ]


# ------------------------------------------------------------------ extraction
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


# ------------------------------------------------------------------ images
def test_image_original_passthrough(image_path):
    enc = encode_image(ImagePart(image_path))
    assert enc.transform == "original" and enc.size == (400, 200)


def test_image_resize_policy(image_path):
    enc = encode_image(ImagePart(image_path), ImagePolicy(max_side=100))
    assert enc.size == (100, 50) and "->" in enc.transform


def test_image_byte_limit(tmp_path):
    import os
    p = tmp_path / "noise.png"
    Image.frombytes("RGB", (600, 600), os.urandom(600 * 600 * 3)).save(p)
    enc = encode_image(ImagePart(str(p)), ImagePolicy(max_b64_bytes=200_000))
    assert len(enc.b64) <= 200_000 and enc.mime_type == "image/jpeg"


# ------------------------------------------------------------------ prompts
def test_default_prompt_is_paper_prompt(samples):
    msgs = build_messages("default", samples[0])
    assert len(msgs) == 1 and msgs[0].role == "user"
    img, text = msgs[0].content
    assert isinstance(img, ImagePart) and text.text == samples[0].default_prompt


# ------------------------------------------------------------------ registry
def test_parse_spec_keeps_colons_in_model_name():
    assert parse_spec("ollama:qwen2.5vl:7b") == ("ollama", "qwen2.5vl:7b")
    with pytest.raises(ValueError):
        parse_spec("nope:model")


# ------------------------------------------------------------------ backends (no network)
def _msgs(image_path):
    return [Message.system("be brief"), Message.user(ImagePart(image_path), "What is this?")]


def test_openai_payload(image_path):
    m = load_model("openai:gpt-test", api_key="x", image_detail="high")
    seen = {}
    m.client.responses.create = lambda **kw: seen.update(kw) or SimpleNamespace(
        output_text="A", status="completed", incomplete_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=1,
                              output_tokens_details=SimpleNamespace(reasoning_tokens=0)))
    out = m.generate(_msgs(image_path), GenerationConfig(max_tokens=5, extra={"reasoning": {"effort": "low"}}))
    assert out.text == "A" and out.usage.input_tokens == 10
    assert seen["instructions"] == "be brief" and seen["max_output_tokens"] == 5
    assert seen["reasoning"] == {"effort": "low"}
    img = seen["input"][0]["content"][0]
    assert img["type"] == "input_image" and img["detail"] == "high"
    assert img["image_url"].startswith("data:image/jpeg;base64,")


def test_anthropic_payload(image_path):
    m = load_model("anthropic:claude-test", api_key="x")
    seen = {}
    m.client.messages.create = lambda **kw: seen.update(kw) or SimpleNamespace(
        content=[SimpleNamespace(type="text", text="B")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=7, output_tokens=1))
    out = m.generate(_msgs(image_path), GenerationConfig())
    assert out.text == "B" and seen["system"] == "be brief" and seen["max_tokens"] == 4096
    assert "extra_body" not in seen  # no sampling params unless requested
    assert seen["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_openrouter_payload(image_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    m = load_model("openrouter:qwen/qwen2.5-vl-72b-instruct")
    assert m.provider == "openrouter" and str(m.client.base_url).startswith("https://openrouter.ai")
    seen = {}
    m.client.chat.completions.create = lambda **kw: seen.update(kw) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="3"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, completion_tokens_details=None))
    out = m.generate(_msgs(image_path), GenerationConfig(temperature=0, extra={"top_k": 1}))
    assert out.text == "3" and seen["temperature"] == 0 and seen["extra_body"] == {"top_k": 1}
    assert seen["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen["messages"][1]["content"][0]["type"] == "image_url"


def test_vllm_custom_base_url():
    m = load_model("vllm:Qwen/Qwen2.5-VL-7B-Instruct", base_url="http://gpu-node:8000/v1")
    assert str(m.client.base_url).startswith("http://gpu-node:8000")


def test_gemini_payload(image_path):
    m = load_model("gemini:gemini-test", api_key="x", media_resolution="MEDIA_RESOLUTION_HIGH")
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return SimpleNamespace(
            text="C", prompt_feedback=None,
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value="STOP"))],
            usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=1, thoughts_token_count=None))
    m.client = SimpleNamespace(models=SimpleNamespace(generate_content=fake))
    out = m.generate(_msgs(image_path), GenerationConfig(max_tokens=8))
    assert out.text == "C" and out.finish_reason == "STOP"
    cfg = seen["config"]
    assert cfg.system_instruction == "be brief" and cfg.max_output_tokens == 8
    assert all(s.threshold.value == "BLOCK_NONE" for s in cfg.safety_settings)
    img = seen["contents"][0].parts[0]
    assert img.inline_data.mime_type == "image/jpeg" and img.media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"


# ------------------------------------------------------------------ runner
def test_runner_end_to_end_and_resume(tmp_path, samples):
    cfg = RunConfig(model="dummy:random", output_dir=str(tmp_path), concurrency=2)
    summary = Runner(load_model(cfg.model), cfg).run(samples)
    run_dir = tmp_path / cfg.resolved_run_name()
    assert summary["n_questions"] == 3 and summary["n_errors"] == 0
    sub = json.loads((run_dir / "submission.json").read_text())
    assert {s["question_id"] for s in sub} == {"q1", "q2", "q3"}

    # resume: nothing re-run
    lines_before = (run_dir / "responses.jsonl").read_text().count("\n")
    Runner(load_model(cfg.model), cfg).run(samples)
    assert (run_dir / "responses.jsonl").read_text().count("\n") == lines_before

    # changing the model under the same run name is refused
    with pytest.raises(RuntimeError):
        Runner(load_model("dummy:echo"), RunConfig(model="dummy:echo", output_dir=str(tmp_path),
                                                   run_name=cfg.resolved_run_name())).run(samples)


class Flaky:
    provider = "flaky"
    supports_concurrency = False

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0
        self.model = "flaky"

    def describe(self):
        return {"provider": "flaky"}

    def generate_batch(self, convs, config):
        from vlmbench.types import Completion
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return [Completion(text="A") for _ in convs]

    def generate(self, msgs, config):
        return self.generate_batch([msgs], config)[0]


def test_runner_retries_then_records_error(tmp_path, samples, monkeypatch):
    monkeypatch.setattr("vlmbench.runner.time.sleep", lambda s: None)
    cfg = RunConfig(model="dummy:x", output_dir=str(tmp_path), max_retries=2, batch_size=3)
    Runner(Flaky(fail_times=2), cfg).run(samples[:1])
    rec = read_records(tmp_path / cfg.resolved_run_name() / "responses.jsonl")["q1"]
    assert rec["error"] is None and rec["attempts"] == 3

    cfg2 = RunConfig(model="dummy:y", output_dir=str(tmp_path), max_retries=1)
    s = Runner(Flaky(fail_times=99), cfg2).run(samples[:1])
    assert s["n_errors"] == 1
    cfg2.retry_errors = True
    s = Runner(Flaky(fail_times=0), cfg2).run(samples[:1])
    assert s["n_errors"] == 0


@pytest.mark.parametrize("resp,expected", [
    ("**1557**", "1557"), ("`Anno`", "Anno"), ("*B*", "B"), ("**B. yes**", "B. yes"), ("plain", "plain"),
    ("**a** and **b**", "**a** and **b**"),
])
def test_normalize_markdown(resp, expected):
    from vlmbench.extract import normalize_response
    assert normalize_response(resp) == expected


def test_markdown_ocr_extracts_clean():
    assert extract("**1557**", "ocr").value == "1557"
