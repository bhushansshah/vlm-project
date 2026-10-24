import os
from types import SimpleNamespace

import pytest
from PIL import Image

from vlmbench.images import ImagePolicy, encode_image
from vlmbench.models import load_model, parse_spec
from vlmbench.types import GenerationConfig, ImagePart, Message


# ------------------------------------------------------------------ images
def test_image_original_passthrough(image_path):
    enc = encode_image(ImagePart(image_path))
    assert enc.transform == "original" and enc.size == (400, 200)


def test_image_resize_policy(image_path):
    enc = encode_image(ImagePart(image_path), ImagePolicy(max_side=100))
    assert enc.size == (100, 50) and "->" in enc.transform


def test_image_byte_limit(tmp_path):
    p = tmp_path / "noise.png"
    Image.frombytes("RGB", (600, 600), os.urandom(600 * 600 * 3)).save(p)
    enc = encode_image(ImagePart(str(p)), ImagePolicy(max_b64_bytes=200_000))
    assert len(enc.b64) <= 200_000 and enc.mime_type == "image/jpeg"


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
