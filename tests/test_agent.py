import json
from dataclasses import replace

import pytest

from vlmbench.agents import ToolAgent, build_agent
from vlmbench.agents.tool_agent import parse_step
from vlmbench.runner import RunConfig, Runner, read_records
from vlmbench.types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage

TOOLS = {"zoom", "grid"}


class Scripted:
    """A fake model that replies with a fixed script and records what it was sent."""
    model = "scripted"
    supports_concurrency = False

    def __init__(self, replies, native=False):
        self.replies, self.seen, self.configs = list(replies), [], []
        self.supports_native_tool_messages = native

    def describe(self):
        return {"provider": "scripted"}

    def generate(self, messages, config):
        self.seen.append([replace(m, content=list(m.content)) for m in messages])
        self.configs.append(config)
        reply = self.replies.pop(0)
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return Completion(text=text, finish_reason=finish, usage=Usage(10, 2, 0))

    def generate_batch(self, convs, config):
        return [self.generate(m, config) for m in convs]


def _question(image_path):
    return [Message.user(ImagePart(image_path), "How many cups? Answer in one short sentence.")]


@pytest.mark.parametrize("text,kind", [
    ('Let me zoom. <action>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</action>', "call"),
    ('<action>{"name": "zoom", "arguments": {"box": [0, 0, 5, 5]}}', "call"),  # stop sequence ate the tag
    ('<tool_call>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</tool_call>', "call"),
    ('<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 5, 5]}}', "call"),  # stop sequence ate the tag
    ('{"name": "zoom", "arguments": {"box": [0, 0, 5, 5]}}', "call"),  # bare JSON
    ("<tool_call>zoom please</tool_call>", "bad_call"),
    ("<answer>Three cups.</answer>", "answer"),
    ('<answer>Two.</answer> <tool_call>{"name": "zoom", "arguments": {}}</tool_call>', "answer"),
    ("There are two cups.", "none"),
])
def test_parse_step(text, kind):
    assert parse_step(text, TOOLS)[0] == kind


def test_call_without_tool_name():
    params = {"zoom": {"box", "grid"}, "grid": {"region", "step"}, "detect": {"query", "region"}}
    text = '<action>{"box": {"left": 1, "top": 2, "right": 30, "bottom": 40}}</action>'
    assert parse_step(text, set(params), params)[:2] == (
        "call", {"name": "zoom", "arguments": {"box": {"left": 1, "top": 2, "right": 30, "bottom": 40}}})
    assert parse_step('<action>{"region": [0, 0, 5, 5]}</action>', set(params), params)[0] == "bad_call"  # ambiguous


def test_arguments_without_wrapper():
    text = '<action>{"name": "zoom", "box": [0, 0, 5, 5]}</action>'
    assert parse_step(text, TOOLS)[1] == {"name": "zoom", "arguments": {"box": [0, 0, 5, 5]}}


def test_tool_call_then_answer(image_path):
    model = Scripted([
        'I need a closer look. <tool_call>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}',
        "<answer>There are 3 cups.</answer>",
    ])
    agent = ToolAgent(model, tools=["zoom", "grid"])
    out = agent.generate(_question(image_path), GenerationConfig(temperature=0))
    assert out.text == "There are 3 cups."
    assert out.usage.input_tokens == 20 and out.usage.output_tokens == 4
    # first turn: instructions + original image + the question; stop sequence added
    first = model.seen[0][0].content
    assert "zoom:" in first[0].text and isinstance(first[1], ImagePart) and "How many cups" in first[2].text
    assert "</action>" in model.configs[0].stop and model.configs[0].max_tokens == 1024
    # second turn sees: assistant call (closing tag restored), tool result with the zoomed image
    convo = model.seen[1]
    assert convo[1].role == "assistant" and convo[1].content[0].text.endswith("</tool_call>")
    assert "<tool_result>" in convo[2].content[0].text and isinstance(convo[2].content[1], ImagePart)
    assert [s.get("tool") for s in out.trace] == ["zoom", None] and out.trace[-1]["final"]


def test_errors_are_shown_to_the_model(image_path):
    model = Scripted([
        '<tool_call>{"name": "teleport", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "zoom", "arguments": {"box": [1, 2]}}</tool_call>',
        "<tool_call>not json</tool_call>",
        "<answer>Two cups.</answer>",
    ])
    out = ToolAgent(model, tools=["zoom"]).generate(_question(image_path), GenerationConfig())
    obs = [s["observation"] for s in out.trace if "observation" in s]
    assert "unknown tool" in obs[0] and "box must be" in obs[1] and "could not read" in obs[2]
    assert all(s["error"] for s in out.trace if "observation" in s) and out.text == "Two cups."


def test_step_limit_and_nudges(image_path):
    call = '<tool_call>{"name": "grid", "arguments": {}}</tool_call>'
    model = Scripted([call, call, call, "<answer>One cup.</answer>"])
    out = ToolAgent(model, tools=["grid"], max_steps=2).generate(_question(image_path), GenerationConfig())
    assert out.text == "One cup." and sum("observation" in s for s in out.trace) == 2
    assert "no tool calls left" in model.seen[3][-1].content[0].text.lower()

    model = Scripted(["I think two."])  # no tags, no tool call: that is the answer
    out = ToolAgent(model, tools=["grid"]).generate(_question(image_path), GenerationConfig())
    assert out.text == "I think two." and out.finish_reason == "no_answer_tag"

    model = Scripted(["", ""])  # empty twice: nudged once, then gives up with ""
    out = ToolAgent(model, tools=["grid"]).generate(_question(image_path), GenerationConfig())
    assert out.text == "" and out.trace[0]["note"] == "empty reply"


def test_keep_last_images(image_path):
    call = '<tool_call>{"name": "grid", "arguments": {}}</tool_call>'
    model = Scripted([call, call, call, "<answer>x</answer>"])
    ToolAgent(model, tools=["grid"], keep_last_images=1).generate(_question(image_path), GenerationConfig())
    last = model.seen[-1]
    images = [p for m in last[1:] for p in m.content if isinstance(p, ImagePart)]
    assert len(images) == 1 and isinstance(last[0].content[1], ImagePart)  # original always kept


def test_agent_run_saves_trace_and_tool_stats(tmp_path, vistaqa):
    replies = []
    for _ in range(12):
        replies += ['<tool_call>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</tool_call>',
                    "<answer>a1</answer>"]
    agent = build_agent("tools", Scripted(replies), tools=["zoom"])
    cfg = RunConfig(model="scripted:x", dataset="vistaqa", data_dir=str(vistaqa.data_dir),
                    output_dir=str(tmp_path / "runs"), agent="tools", agent_options={"tools": ["zoom"]})
    assert cfg.run_dir().name == "vistaqa__scripted__x__default__agent-tools-zoom"
    summary = Runner(agent, vistaqa, cfg).run(vistaqa.load())
    rec = read_records(cfg.run_dir() / "responses.jsonl")["1"]
    assert rec["response"] == "a1" and rec["trace"][0]["tool"] == "zoom"
    assert summary["tool_use"]["calls_per_tool"] == {"zoom": 12} and summary["tool_use"]["used_any_tool"] == 1.0
    assert json.loads((cfg.run_dir() / "config.json").read_text())["backend"]["provider"] == "agent:tools"


def test_repeat_calls_are_not_rerun(image_path):
    call = '<action>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</action>'
    model = Scripted([call, call, "<answer>x</answer>"])
    out = ToolAgent(model, tools=["zoom"]).generate(_question(image_path), GenerationConfig())
    assert "already made this exact call at step 1" in out.trace[1]["observation"]
    assert out.trace[1]["images"] == [] and not out.trace[1]["error"]


def test_native_tool_specs_are_declared(image_path):
    model = Scripted(["<answer>x</answer>"])
    ToolAgent(model, tools=["zoom", "grid"]).generate(_question(image_path), GenerationConfig())
    assert [t["name"] for t in model.configs[0].tools] == ["zoom", "grid"]
    model = Scripted(["<answer>x</answer>"])
    ToolAgent(model, tools=["zoom"], native_tools=False).generate(_question(image_path), GenerationConfig())
    assert model.configs[0].tools is None


def test_openai_compat_converts_native_tool_calls(monkeypatch):
    from types import SimpleNamespace
    from vlmbench.models import load_model
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    m = load_model("openrouter:some/model")
    seen = {}
    call = SimpleNamespace(function=SimpleNamespace(name="zoom", arguments='{"box": [1, 2, 3, 4]}'))
    m.client.chat.completions.create = lambda **kw: seen.update(kw) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]), finish_reason="tool_calls")],
        usage=None)
    out = m.generate([Message.user("hi")], GenerationConfig(tools=[{"name": "zoom", "description": "d", "parameters": {}}]))
    assert seen["tools"][0] == {"type": "function", "function": {"name": "zoom", "description": "d", "parameters": {}}}
    assert parse_step(out.text, {"zoom"}) [:2] == ("call", {"name": "zoom", "arguments": {"box": [1, 2, 3, 4]}})


def test_step_budget_and_cut_off_reply(image_path):
    # a reply that ran out of tokens before an answer gets one "answer now" nudge
    model = Scripted([("Let me count the cups one by one: first", "length"), "<answer>Three cups.</answer>"])
    out = ToolAgent(model, tools=["zoom"]).generate(_question(image_path), GenerationConfig(max_tokens=10000))
    assert out.text == "Three cups." and out.trace[0]["note"] == "cut off"
    assert "cut off" in model.seen[1][-1].content[0].text
    assert model.configs[0].max_tokens == 10000  # the run's budget applies to every step
    model = Scripted(["<answer>x</answer>"])
    ToolAgent(model, tools=["zoom"], step_max_tokens=300).generate(_question(image_path), GenerationConfig(max_tokens=10000))
    assert model.configs[0].max_tokens == 300  # unless the agent sets its own


def test_leaked_template_fragments_are_removed(image_path):
    from vlmbench.agents.tool_agent import clean_text
    assert clean_text("_calls> <ate I will zoom.") == "I will zoom."
    assert clean_text("essage|><answer>Blue.</answer>") == "<answer>Blue.</answer>"
    assert clean_text("<|im_end|>The value x|y stays") == "The value x|y stays"
    model = Scripted(['_calls> <ate<action>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</action>',
                      "essage|><answer>Blue.</answer>"])
    out = ToolAgent(model, tools=["zoom"]).generate(_question(image_path), GenerationConfig())
    assert out.text == "Blue." and not out.trace[0]["model_output"].startswith("_calls")


def test_native_protocol_sends_tool_turns(image_path):
    call = 'Look closer. <action>{"name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}</action>'
    model = Scripted([call, "<answer>x</answer>"], native=True)
    ToolAgent(model, tools=["zoom"], protocol="native").generate(_question(image_path), GenerationConfig())
    assistant, tool, user = model.seen[1][1:4]
    assert assistant.tool_calls == [{"id": "call_1", "name": "zoom", "arguments": {"box": [0, 0, 500, 500]}}]
    assert assistant.content[0].text == "Look closer."
    assert tool.role == "tool" and tool.tool_call_id == "call_1" and "Zoomed view" in tool.content[0].text
    assert isinstance(user.content[0], ImagePart) and "call(s) left" in user.content[-1].text

    # a backend without native tool turns keeps the text protocol
    model = Scripted([call, "<answer>x</answer>"], native=False)
    ToolAgent(model, tools=["zoom"], protocol="native").generate(_question(image_path), GenerationConfig())
    assert model.seen[1][1].tool_calls is None and "<tool_result>" in model.seen[1][2].content[0].text


def test_openai_compat_serializes_native_tool_turns(monkeypatch):
    from types import SimpleNamespace
    from vlmbench.models import load_model
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    m = load_model("openrouter:some/model")
    seen = {}
    m.client.chat.completions.create = lambda **kw: seen.update(kw) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None)
    msgs = [Message.user("hi"),
            Message("assistant", [], tool_calls=[{"id": "call_1", "name": "zoom", "arguments": {"box": [1, 2, 3, 4]}}]),
            Message("tool", [TextPart("Zoomed view")], tool_call_id="call_1")]
    m.generate(msgs, GenerationConfig())
    a, t = seen["messages"][1:]
    assert a == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "zoom", "arguments": '{"box": [1, 2, 3, 4]}'}}]}
    assert t == {"role": "tool", "tool_call_id": "call_1", "content": "Zoomed view"}


def test_evidence_parsing():
    from vlmbench.agents.tool_agent import parse_evidence
    box = {"left": 1, "top": 2, "right": 30, "bottom": 40}
    text = f'<evidence>[{{"label": "cup", "box": {json.dumps(box)}}}]</evidence><answer>A cup.</answer>'
    assert parse_evidence(text) == [{"label": "cup", "box": box}]
    assert parse_evidence(f'<evidence>{{"label": "cup", "box": {json.dumps(box)}}}<answer>x</answer>') == [
        {"label": "cup", "box": box}]  # a single object, no closing tag
    assert parse_evidence("<answer>No cup.</answer>") == [] and parse_evidence("<evidence>junk</evidence>") == []


def test_verify_sends_a_failed_check_back_once(image_path):
    ev = '<evidence>[{"label": "oxygen tank", "box": {"left": 0, "top": 0, "right": 500, "bottom": 500}}]</evidence>'
    model = Scripted([ev + "<answer>A fish.</answer>",
                      "no\nThere is only a snorkel, no tank.",           # the separate check
                      "<answer>There is no oxygen tank in the image.</answer>"])
    out = ToolAgent(model, tools=["zoom"], instructions="v4", verify=True).generate(
        _question(image_path), GenerationConfig())
    assert out.text == "There is no oxygen tank in the image."
    check = model.seen[1][0].content
    assert "oxygen tank" in check[0].text and sum(isinstance(p, ImagePart) for p in check) == 2
    assert model.configs[1].tools is None and model.configs[1].stop is None  # a plain question, no tools
    assert out.trace[0]["verify"][0]["ok"] is False and "did not confirm" in model.seen[2][-1].content[0].text
    assert out.usage.input_tokens == 30  # the check is billed to the answer

    model = Scripted([ev + "<answer>A fish.</answer>", "Yes, a tank."])  # confirmed: answer kept
    out = ToolAgent(model, tools=["zoom"], instructions="v4", verify=True).generate(
        _question(image_path), GenerationConfig())
    assert out.text == "A fish." and out.trace[-1]["final"] and out.trace[-1]["verify"][0]["ok"] is True


def test_v4_defaults_to_its_tools_and_old_runs_keep_their_name(tmp_path, vistaqa):
    agent = ToolAgent(Scripted([]), instructions="v4")
    assert set(agent.tools) == {"zoom", "grid", "detect", "count", "mark", "ocr", "depth", "distance"}
    assert "verify" not in agent.options and "protocol" not in agent.options
    cfg = RunConfig(model="scripted:x", dataset="vistaqa", data_dir=str(vistaqa.data_dir),
                    output_dir=str(tmp_path), agent="tools", agent_options={"instructions": "v2"})
    assert cfg.run_dir().name == "vistaqa__scripted__x__default__agent-tools-v2"
    cfg = replace(cfg, agent_options={"instructions": "v4", "protocol": "native", "verify": True})
    assert cfg.run_dir().name == "vistaqa__scripted__x__default__agent-tools-v4-native-verify"
    assert replace(cfg, tag="reason-medium").run_dir().name.endswith("-v4-native-verify__reason-medium")
