"""Tool-augmented VLM agent (ReAct-style): the model alternates reasoning and tool calls, then answers.

The protocol is plain text, so it works with every backend (API or local) and behaves the same
for all models: the model writes ``<action>{"name": ..., "arguments": {...}}</action>`` to
use a tool, gets the result (text + images) in the next message, and finishes with
``<answer>...</answer>``. (Not ``<tool_call>``: some API providers intercept that tag and
return an empty reply; it is still accepted if a model writes it.) The agent wraps any ``VLM`` and is itself a ``VLM``, so the runner,
judge and reports work unchanged; every step is kept in ``Completion.trace``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, replace
from typing import Any

from PIL import Image, ImageDraw

from ..models.base import VLM
from ..retry import call_with_retries
from ..tools import Tool, ToolContext, ToolError, build_tools
from ..tools.base import fmt_box, parse_box, to_pixels
from ..tools.image_ops import resize_for_view
from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage

# Versioned agent instructions: add a new version instead of editing one, so every run stays reproducible.
# Placeholders: {width} {height} {tools} {max_steps}.
INSTRUCTIONS: dict[str, str] = {}
INSTRUCTIONS["v1"] = """You are answering a question about an image. You can call tools to inspect the image more closely before you answer.

Coordinates: every box is {{"left": .., "top": .., "right": .., "bottom": ..}} on a 0-1000 scale of the ORIGINAL image. left and right are horizontal positions (0 = left edge, 1000 = right edge); top and bottom are vertical positions (0 = top edge, 1000 = bottom edge). Example: the top-right quarter of the image is {{"left": 500, "top": 0, "right": 1000, "bottom": 500}}. The original image is {width}x{height} pixels.

Tools:
{tools}

How to work:
- Think briefly about what you need to check, then call ONE tool by writing exactly:
<action>{{"name": "<tool name>", "arguments": {{...}}}}</action>
- Stop writing after the tool call. Its result comes back in the next message.
- You can make up to {max_steps} tool calls. Use tools when details are small, text must be read, objects must be counted or distances compared. Skip them when the answer is already clear.
- Tool results can be wrong. Check them against what you see.
- When you are ready, write your final answer inside <answer></answer>, following the question's instructions."""

# v2 (after tuning round 1 on analysis/tuning_ids.txt): think before each call, example calls,
# check the question's premise, how to use detector / depth / OCR results, commit to one answer.
INSTRUCTIONS["v2"] = """You are answering a question about an image. You can call tools to inspect the image more closely before you answer.

Coordinates: every box is {{"left": .., "top": .., "right": .., "bottom": ..}} on a 0-1000 scale of the ORIGINAL image. left and right are horizontal positions (0 = left edge, 1000 = right edge); top and bottom are vertical positions (0 = top edge, 1000 = bottom edge). Example: the top-right quarter of the image is {{"left": 500, "top": 0, "right": 1000, "bottom": 500}}. The original image is {width}x{height} pixels.

Tools:
{tools}

Example calls:
{examples}

How to work:
1. Look at the image and the question. Before each tool call, write one or two sentences: what you see so far and what the call will check. Then call ONE tool by writing:
<action>{{"name": "<tool name>", "arguments": {{...}}}}</action>
and stop. The result comes back in the next message. You can make up to {max_steps} tool calls; skip tools when the answer is already clear.
2. Check the question's premise. Some questions describe objects, people or text that are NOT in the image, or not as described. Before answering, make sure everything the question names is really there; zoom in if unsure. If something is missing, say that it is not in the image. Do not answer about a similar-looking object instead (a different sign, app, animal or person). If it is there, even small or partly hidden, answer normally.
3. Use tool results critically:
   - detect can miss objects, count one object twice, or label the wrong thing. When counting, look at the numbered image, fix the count, and zoom into crowded areas. Keep queries short (one to three nouns).
   - depth only tells how far things are from the camera. It does not measure distances between objects, or to a wall, table or robot.
   - ocr can misread small text; zoom in to confirm. In diagrams and charts, parts are identified by their text labels: read the labels (ocr or zoom) instead of saying the image lacks the information.
   - If a tool result disagrees with what you clearly see, look again (zoom) before deciding.
4. Give ONE final answer inside <answer></answer> that directly answers the question and follows its instructions. Do not hedge between options, and do not add details the question did not ask about."""

# v3 (after tuning round 2): v2 with a stricter premise rule. "There, but not as described" counts as
# missing, and then the answer only says so (no details about a look-alike object).
INSTRUCTIONS["v3"] = INSTRUCTIONS["v2"].replace(
    """2. Check the question's premise. Some questions describe objects, people or text that are NOT in the image, or not as described. Before answering, make sure everything the question names is really there; zoom in if unsure. If something is missing, say that it is not in the image. Do not answer about a similar-looking object instead (a different sign, app, animal or person). If it is there, even small or partly hidden, answer normally.""",
    """2. Check the question's premise. Some questions describe objects, people or text that are NOT in the image, or that are there but not as described (in a different place, with a different color, count or name). Before answering, make sure everything the question names is really there as described; zoom in if unsure. If not, your answer is only that: say what is not in the image (or not as described). Do not give the requested information for something else, such as a similar sign, app, animal or person, or an object in a different place. If everything is there, even small or partly hidden, answer normally.""",
)
assert INSTRUCTIONS["v3"] != INSTRUCTIONS["v2"]

# v4 (after the failure analysis of the full v2 runs): mark-and-verify counting (count, mark),
# distance between objects (distance), zoom limits, detector misses, and cited evidence boxes
# before the answer (checked when the agent runs with verify=True). Meant for V4_TOOLS.
INSTRUCTIONS["v4"] = """You are answering a question about an image. You can call tools to inspect the image more closely before you answer.

Coordinates: every box is {{"left": .., "top": .., "right": .., "bottom": ..}} and every point is {{"x": .., "y": ..}}, on a 0-1000 scale of the ORIGINAL image. left, right and x are horizontal positions (0 = left edge, 1000 = right edge); top, bottom and y are vertical positions (0 = top edge, 1000 = bottom edge). Example: the top-right quarter of the image is {{"left": 500, "top": 0, "right": 1000, "bottom": 500}}. The original image is {width}x{height} pixels.

Tools:
{tools}

Example calls:
{examples}

How to work:
1. Look at the image and the question. Before each tool call, write one or two sentences: what you see so far and what the call will check. Then call ONE tool by writing:
<action>{{"name": "<tool name>", "arguments": {{...}}}}</action>
and stop. The result comes back in the next message. You can make up to {max_steps} tool calls; skip tools when the answer is already clear.
2. Check the question's premise. Some questions describe objects, people or text that are NOT in the image, or not as described. Make sure everything the question names is really there; if something is missing, say that it is not in the image. Do not answer about a similar-looking object instead. Do not search until you "find" it: if you have to zoom into a tiny, blurry spot to make it out, it is probably not there. If it is clearly there, even small or partly hidden, answer normally.
3. Use tools critically:
   - zoom enlarges at most 3x. A region only a few dozen pixels wide in the original holds little detail, and zooming in further will not show more.
   - Counting: call count, then check every numbered mark (zoom into crowded parts) and say which numbers you accept; add real objects that have no mark. If you counted by eye, check your count with mark (one dot per object).
   - detect can miss objects, count one object twice, or label the wrong thing. "No confident detections" does NOT mean the object is absent: look yourself. Keep queries short (one to three nouns).
   - depth only tells how far things are from the camera. For distances between objects (closer to the robot, the gripper, a wall, the table edge), use distance with tight boxes.
   - ocr can misread small text; zoom in to confirm. In diagrams and charts, parts are identified by their text labels: read the labels instead of saying the image lacks the information.
   - If a tool result disagrees with what you clearly see, look again before deciding.
4. Finish with your evidence and ONE answer:
   - If your answer is about specific objects in the image, first give tight boxes for them (at most 3), e.g.
<evidence>[{{"label": "red cup", "box": {{"left": 410, "top": 520, "right": 470, "bottom": 600}}}}]</evidence>
     If your answer is that something is not in the image, give no evidence.
   - Then <answer></answer>: answer the question directly and follow its instructions. Give one specific answer (one color, one number, one name), not "X or Y" or "X/Y", and do not add details the question did not ask about."""

# Tools each instruction version is written for (used when the run names no tools).
V4_TOOLS = ("zoom", "grid", "detect", "count", "mark", "ocr", "depth", "distance")
INSTRUCTION_TOOLS = {"v4": V4_TOOLS}

# Fragments of a model's own chat/tool-call template that some providers leak into the text
# (e.g. "_calls> <ate", "essage|>"). Left in, the model sees them in its history and repeats them.
_TEMPLATE_JUNK = re.compile(r"<\|[^|<>\s]{0,40}\|>|(?<![\w|])[A-Za-z_]{0,12}\|>|_calls>|<ate\b")


def clean_text(text: str) -> str:
    """Model text without leaked template fragments."""
    return re.sub(r"[ \t]{2,}", " ", _TEMPLATE_JUNK.sub("", text)).strip() if text else text


VERIFY_PROMPT = """Check one detail of an image. The first image is the whole picture, with a red box. The second image shows the area of the red box enlarged.

Is there {label} inside the red box? Judge only from what you can clearly see; if you cannot make it out, the answer is no.
Reply with "yes" or "no" on the first line, then one short reason."""

_CALL = re.compile(r"<(action|tool_call)>(.*?)(?:</\1>|$)", re.S)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)
_EVIDENCE = re.compile(r"<evidence>(.*?)(?:</evidence>|(?=<answer>)|$)", re.S)


def _first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                return decoder.raw_decode(text, i)[0]
            except ValueError:
                continue
    return None


def _arguments(obj: dict) -> dict:
    """The call's arguments. Some models put them next to "name" instead of under "arguments"."""
    if "arguments" in obj:
        return obj["arguments"] or {}
    return {k: v for k, v in obj.items() if k != "name"}


def parse_evidence(text: str) -> list[dict[str, Any]]:
    """Boxes the model cited before its answer: [{"label", "box"}], unreadable items dropped."""
    m = _EVIDENCE.search(text)
    if not m:
        return []
    body = m.group(1)
    i = body.find("[")
    items: Any = None
    if i >= 0:
        try:
            items = json.JSONDecoder().raw_decode(body, i)[0]
        except ValueError:
            items = None
    if items is None:
        obj = _first_json_object(body)
        items = [obj] if obj is not None else []
    return [it for it in (items if isinstance(items, list) else [items])
            if isinstance(it, dict) and isinstance(it.get("label"), str) and "box" in it]


def _guess_tool(args: dict, tool_params: dict[str, set[str]]) -> str | None:
    """The tool a call meant when the model left out "name": the only tool taking all its arguments."""
    fits = [n for n, params in tool_params.items() if args and set(args) <= params]
    return fits[0] if len(fits) == 1 else None


def parse_step(text: str, tool_names: set[str],
               tool_params: dict[str, set[str]] | None = None) -> tuple[str, Any, int]:
    """Classify a model turn. Returns (kind, payload, end) with kind in
    {"call", "answer", "bad_call", "none"}; ``end`` = where to cut the turn (after the call).
    ``tool_params`` (tool -> argument names) lets a call without "name" be matched to its tool."""
    call = _CALL.search(text)
    answer = _ANSWER.search(text)
    if call and not (answer and answer.start() < call.start()):
        obj = _first_json_object(call.group(2))
        end = call.end()
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            return "call", {"name": obj["name"], "arguments": _arguments(obj)}, end
        if isinstance(obj, dict) and tool_params and (name := _guess_tool(obj, tool_params)):
            return "call", {"name": name, "arguments": obj}, end
        return "bad_call", f"could not read the tool call as JSON: {call.group(2).strip()[:200]}", end
    if answer:
        return "answer", answer.group(1).strip(), len(text)
    # Fallback: a bare JSON tool call without tags
    obj = _first_json_object(text)
    if isinstance(obj, dict) and obj.get("name") in tool_names and "arguments" in obj:
        return "call", {"name": obj["name"], "arguments": obj.get("arguments") or {}}, len(text)
    return "none", text.strip(), len(text)


class ToolAgent(VLM):
    """Options:
        tools: tool names (default: the instruction version's tools, else zoom, grid, detect, ocr, depth)
        tool_options: per-tool options, e.g. {"detect": {"box_threshold": 0.35}}
        max_steps: maximum tool calls per question (default 6)
        step_max_tokens: output budget per model turn (default: the run's max_tokens, else 1024)
        keep_last_images: keep only the newest N tool images in the conversation (default: all)
        instructions: which version of the agent instructions to use (default "v1"; see INSTRUCTIONS)
        native_tools: also declare the tools through the API's native tool-calling field (default
            True). Some models/providers only call tools natively; those calls are converted to
            <action> text, so the loop is the same either way. Backends without native tool
            support ignore it.
        protocol: "text" (default) sends the history as plain text (<action> blocks, results in user
            turns); "native" sends it as native tool calls and "tool" turns, as the model's own chat
            template expects (tool images follow in a user turn). "native" needs native_tools and a
            backend that takes tool turns (OpenAI-compatible: OpenRouter, vLLM, ...); others use text.
        verify: check the evidence boxes the model cites before its answer (<evidence>, see the v4
            instructions) with a separate yes/no call per box; if one fails, the model is told and
            gets one more chance (with verify_extra_calls more tool calls). Default False.
        verify_extra_calls: tool calls allowed after a failed check (default 2)
    """

    provider = "agent"

    def __init__(self, model: VLM, tools: list[str] | None = None,
                 tool_options: dict[str, dict[str, Any]] | None = None, max_steps: int = 6,
                 step_max_tokens: int | None = None, keep_last_images: int | None = None, max_retries: int = 3,
                 native_tools: bool = True, instructions: str = "v1", protocol: str = "text",
                 verify: bool = False, verify_extra_calls: int = 2):
        if instructions not in INSTRUCTIONS:
            raise ValueError(f"Unknown instructions {instructions!r}. Available: {sorted(INSTRUCTIONS)}")
        if protocol not in ("text", "native"):
            raise ValueError(f"protocol must be 'text' or 'native', got {protocol!r}")
        self.instructions = instructions
        self.inner = model
        tools = tools or list(INSTRUCTION_TOOLS.get(instructions, ())) or None
        self.tools: dict[str, Tool] = {t.name: t for t in build_tools(tools, tool_options)}
        self.max_steps = max_steps
        self.step_max_tokens = step_max_tokens
        self.keep_last_images = keep_last_images
        self.max_retries = max_retries
        self.native_tools = native_tools
        # native history only where the backend can send it; otherwise the plain-text protocol
        self.native = protocol == "native" and native_tools and model.supports_native_tool_messages
        self.verify = verify
        self.verify_extra_calls = verify_extra_calls
        self._tool_params = {n: set(t.parameters) for n, t in self.tools.items()}
        self.supports_concurrency = model.supports_concurrency
        opts = dict(tools=list(self.tools), tool_options=tool_options or {}, max_steps=max_steps,
                    step_max_tokens=step_max_tokens, keep_last_images=keep_last_images,
                    native_tools=native_tools, instructions=instructions)
        # new options only when used, so earlier runs keep the same identity (run name and config)
        if protocol != "text":
            opts["protocol"] = protocol
        if verify:
            opts.update(verify=True, verify_extra_calls=verify_extra_calls)
        super().__init__(model.model, **opts)

    def describe(self) -> dict[str, Any]:
        return {"provider": "agent:tools", "inner": self.inner.describe(), "options": self.options}

    # ------------------------------------------------------------ prompt
    def _tool_list(self) -> str:
        lines = []
        for t in self.tools.values():
            spec = t.spec()
            args = json.dumps(spec["parameters"]["properties"], separators=(", ", ": "))
            req = ", ".join(spec["parameters"]["required"]) or "none"
            lines.append(f"- {t.name}: {t.description}\n  arguments: {args}\n  required: {req}")
        return "\n".join(lines)

    def _examples(self) -> str:
        return "\n".join(f'<action>{json.dumps({"name": t.name, "arguments": t.example})}</action>'
                          for t in self.tools.values())

    def _start(self, messages: list[Message], ctx: ToolContext) -> list[Message]:
        """First user turn: tool instructions, the image, then the question as the dataset phrased it."""
        user = next(m for m in messages if m.role == "user")
        image = next(p for p in user.content if isinstance(p, ImagePart))
        question = [p for p in user.content if isinstance(p, TextPart)]
        w, h = ctx.size
        intro = INSTRUCTIONS[self.instructions].format(width=w, height=h, tools=self._tool_list(),
                                                       examples=self._examples(), max_steps=self.max_steps)
        system = [p for m in messages if m.role == "system" for p in m.content]
        return [Message("user", [TextPart(intro), *system, image, *question])]

    def _trim_images(self, convo: list[Message]) -> None:
        if self.keep_last_images is None:
            return
        # the original image lives in the first message and is always kept
        tool_images = [(i, j) for i, m in enumerate(convo) if i > 0
                       for j, p in enumerate(m.content) if isinstance(p, ImagePart)]
        excess = len(tool_images) - self.keep_last_images
        for i, j in tool_images[:max(0, excess)]:
            convo[i].content[j] = TextPart("[earlier tool image removed]")

    # -------------------------------------------------------------- loop
    def _ask(self, convo: list[Message], cfg: GenerationConfig) -> Completion:
        comp, err, _ = call_with_retries(self.inner.generate, convo, cfg, max_retries=self.max_retries)
        if err:
            raise RuntimeError(f"model call failed: {err}")
        return comp

    def _run_tool(self, ctx: ToolContext, call: dict[str, Any]):
        tool = self.tools.get(call["name"])
        if tool is None:
            return None, f"Error: unknown tool '{call['name']}'. Available: {', '.join(self.tools)}.", True
        try:
            return tool(ctx, call["arguments"]), None, False
        except ToolError as e:
            return None, f"Error: {e}", True
        except Exception as e:  # noqa: BLE001 - a broken tool must not end the episode
            return None, f"Error: the {tool.name} tool failed ({type(e).__name__}: {e}).", True

    def _add_result(self, convo: list[Message], said: str, call: dict[str, Any] | None, call_id: str,
                    obs: str, images: list, note: str) -> None:
        """Append the model's call and the tool result, as text or as native tool turns."""
        if self.native and call is not None:
            thought = _CALL.split(said, maxsplit=1)[0].strip()
            convo.append(Message("assistant", [TextPart(thought)] if thought else [],
                                 tool_calls=[{"id": call_id, **call}]))
            convo.append(Message("tool", [TextPart(obs)], tool_call_id=call_id))
            convo.append(Message("user", [*(ImagePart(img) for img in images), TextPart(note)]))
            return
        convo.append(Message.assistant(said))
        parts = [TextPart(f"<tool_result>\n{obs}\n</tool_result>"), *(ImagePart(img) for img in images),
                 TextPart(note)]
        convo.append(Message("user", parts))

    def _check_evidence(self, ctx: ToolContext, evidence: list[dict[str, Any]],
                        cfg: GenerationConfig, usage: Usage) -> list[dict[str, Any]]:
        """One yes/no call per cited box (whole image with the box drawn + the area enlarged)."""
        check_cfg = replace(cfg, stop=None, tools=None)
        results = []
        for ev in evidence[:3]:
            try:
                box = parse_box(ev["box"], "evidence box")
            except ToolError as e:
                results.append({"label": ev["label"], "box": ev["box"], "ok": None, "reason": str(e)})
                continue
            full, crop = evidence_views(ctx.image, box)
            msgs = [Message.user(VERIFY_PROMPT.format(label=ev["label"]), ImagePart(full), ImagePart(crop))]
            comp = self._ask(msgs, check_cfg)
            usage.add(comp.usage)
            reply = clean_text(comp.text or "")
            first = (reply.split() or [""])[0].strip(".,:*\"'").lower()
            ok = True if first == "yes" else False if first == "no" else None  # unclear: not held against it
            results.append({"label": ev["label"], "box": [round(v) for v in box], "ok": ok,
                            "reason": reply[:300], "usage": asdict(comp.usage)})
        return results

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        image = next(p for m in messages if m.role == "user" for p in m.content if isinstance(p, ImagePart))
        ctx = ToolContext(image=image.load().convert("RGB"))
        convo = self._start(messages, ctx)
        cfg = replace(config, max_tokens=self.step_max_tokens or config.max_tokens or 1024,
                      stop=[*(config.stop or []), "</action>"],
                      tools=[t.spec() for t in self.tools.values()] if self.native_tools else None)
        seen_calls: dict[str, int] = {}  # exact repeat calls -> the step that first made them
        usage, trace = Usage(), []
        calls = 0
        max_calls = self.max_steps
        nudged = cut_nudged = verified = False
        answer, finish = None, None
        last_text = ""

        while True:
            comp = self._ask(convo, cfg)
            usage.add(comp.usage)
            text = clean_text(comp.text or "")
            if text.strip():
                last_text = _ANSWER.sub(lambda m: m.group(1), text).strip()
            kind, payload, end = parse_step(text, set(self.tools), self._tool_params)
            step = {"step": len(trace) + 1, "model_output": text[:end], "usage": asdict(comp.usage)}

            if kind in ("call", "bad_call") and calls < max_calls:
                calls += 1
                said = text[:end]
                for tag in ("action", "tool_call"):  # the stop sequence ate the closing tag
                    if f"<{tag}>" in said and f"</{tag}>" not in said:
                        said += f"</{tag}>"
                if kind == "call":
                    key = json.dumps(payload, sort_keys=True)
                    if key in seen_calls:  # nothing new to see: don't pay for the same image again
                        result, failed = None, False
                        error_text = (f"You already made this exact call at step {seen_calls[key]}; "
                                      "its result is above. Try something different or answer.")
                    else:
                        seen_calls[key] = len(trace) + 1
                        result, error_text, failed = self._run_tool(ctx, payload)
                    step.update(tool=payload["name"], arguments=payload["arguments"])
                else:
                    result, error_text, failed = None, f"Error: {payload}", True
                    step.update(tool=None, arguments=None)
                obs = error_text if result is None else result.text
                left = max_calls - calls
                note = (f"You have {left} tool call(s) left." if left else
                        "You have no tool calls left. Give your final answer inside <answer></answer>.")
                images = result.images if result else []
                self._add_result(convo, said, payload if kind == "call" else None, f"call_{calls}", obs,
                                 images, note)
                self._trim_images(convo)
                step.update(observation=obs, error=failed,
                            data=result.data if result else None,
                            images=[list(img.size) for img in images])
                trace.append(step)
                continue

            if kind == "answer":
                evidence = parse_evidence(text)
                if self.verify and not verified and evidence:
                    verified = True
                    checks = self._check_evidence(ctx, evidence, cfg, usage)
                    failed_checks = [c for c in checks if c["ok"] is False]
                    trace.append({**step, "evidence": evidence, "verify": checks})
                    if failed_checks:
                        max_calls = calls + self.verify_extra_calls
                        bad = "; ".join(f"'{c['label']}' at {fmt_box(c['box'])}: {c['reason'][:150]}"
                                        for c in failed_checks)
                        convo.append(Message.assistant(text))
                        convo.append(Message.user(
                            f"A separate check of your evidence did not confirm it: {bad}. Look again (you have "
                            f"{self.verify_extra_calls} more tool call(s)). If something the question asks about "
                            "is not in the image, your answer should say so. Then give your final answer inside "
                            "<answer></answer>."))
                        continue
                    answer, finish = payload, comp.finish_reason
                    trace[-1]["final"] = True
                    break
                answer, finish = payload, comp.finish_reason
            elif kind in ("call", "bad_call"):  # out of tool calls: ask once more for the answer
                if nudged:
                    answer, finish = text.strip(), "no_answer_tag"
                else:
                    nudged = True
                    convo.append(Message.assistant(text[:end]))
                    convo.append(Message.user("You have no tool calls left. Give your final answer now inside <answer></answer>."))
                    trace.append({**step, "note": "tool call over the limit"})
                    continue
            elif comp.finish_reason == "length" and not cut_nudged:
                # the reply ran out of tokens before an answer: ask for the answer, not more reasoning
                cut_nudged = True
                convo.append(Message.assistant(text))
                convo.append(Message.user("Your reply was cut off before you gave an answer. Do not repeat your "
                                          "reasoning: give your final answer now inside <answer></answer>."))
                trace.append({**step, "note": "cut off"})
                continue
            else:  # neither a tool call nor an answer tag
                if payload:  # a reply without a tool call is the answer, tagged or not
                    answer, finish = payload, "no_answer_tag"
                elif nudged:  # still nothing: keep the last thing the model actually said
                    answer, finish = last_text, "no_answer_tag"
                else:
                    nudged = True
                    convo.append(Message.assistant(text))
                    convo.append(Message.user("Reply with either one <action>...</action> or your final "
                                              "answer inside <answer></answer>."))
                    trace.append({**step, "note": "empty reply"})
                    continue
            trace.append({**step, "final": True})
            break

        return Completion(text=answer, finish_reason=finish, usage=usage, trace=trace)


def evidence_views(image: Image.Image, box) -> tuple[Image.Image, Image.Image]:
    """For an evidence check: the whole image with the box in red, and the box's area (with a
    margin) enlarged at most 3x, so the checker sees as much as the image really holds."""
    w, h = image.size
    x1, y1, x2, y2 = to_pixels(box, (w, h))
    full = resize_for_view(image, 512 * 512, 1024 * 1024).convert("RGB")
    s = full.width / w
    d = ImageDraw.Draw(full)
    d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=(255, 0, 0), width=max(2, full.width // 300))
    mx, my = (x2 - x1) // 4 + 2, (y2 - y1) // 4 + 2
    crop = image.crop((max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)))
    return full, resize_for_view(crop, 384 * 384, 1024 * 1024, max_upscale=3.0)


def summarize_traces(records: dict[str, dict]) -> dict[str, Any]:
    """Tool-use statistics for summary.json."""
    with_trace = [r for r in records.values() if r.get("trace") is not None]
    if not with_trace:
        return {}
    per_tool, errors = Counter(), Counter()
    n_calls = []
    for r in with_trace:
        n_calls.append(sum(1 for s in r["trace"] if "observation" in s))
        for s in r["trace"]:
            if "observation" in s:
                per_tool[s.get("tool") or "(unreadable call)"] += 1
                errors[s.get("tool") or "(unreadable call)"] += bool(s.get("error"))
    n = len(with_trace)
    return {"tool_use": {
        "questions": n,
        "used_any_tool": round(sum(c > 0 for c in n_calls) / n, 4),
        "mean_tool_calls": round(sum(n_calls) / n, 2),
        "calls_per_tool": dict(per_tool.most_common()),
        "failed_calls_per_tool": {k: v for k, v in errors.items() if v},
        "no_answer_tag": sum(1 for r in with_trace if r.get("finish_reason") == "no_answer_tag"),
    }}
