"""Tool-call SFT data from dumped agent conversations (arm B1; plan, section 4, agent column).

    python training/make_agent_sft.py --dumps data/agent_dumps/teacher-27b-nothink \
        --run runs/train_v1__teacher-27b-agent-nothink__sft --out data/sft_arms/b1_teacher_agent.jsonl

Keeps conversations whose final answer is correct (graded exactly as the run grades it) and whose agent ended
normally. Each assistant turn (every tool call and the final answer) becomes one example:
- chat_prompt: the conversation up to that turn, rendered with Qwen3.5's chat template, the tool definitions and
  enable_thinking off, with the generation prompt, i.e. exactly what vLLM rendered for the model at that step;
- images: every image in that prefix (the photo, then tool-result images), in order;
- target: what the model wrote at that step (text and, for a call, the tool call in the template's own format).
Tool results are part of the prompt, never of the target, so they are not trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoProcessor

from vlmbench.datasets import get_dataset
from vlmbench.runner import read_records


def to_hf(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Dumped messages -> HF chat messages (image placeholders) and the image paths in order."""
    conv, images = [], []
    for m in messages:
        if m["role"] == "assistant":
            text = "".join(p["text"] for p in m["content"] if p["type"] == "text")
            msg = {"role": "assistant", "content": text}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in m["tool_calls"]]
            conv.append(msg)
        elif m["role"] == "tool":
            conv.append({"role": "tool", "content": "".join(p["text"] for p in m["content"] if p["type"] == "text")})
        else:
            parts = []
            for p in m["content"]:
                if p["type"] == "image":
                    parts.append({"type": "image"})
                    images.append(p["path"])
                else:
                    parts.append({"type": "text", "text": p["text"]})
            conv.append({"role": m["role"], "content": parts})
    return conv, images


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/train_v1/sft"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    proc = AutoProcessor.from_pretrained(args.model)
    ds = get_dataset("tbstyle", args.data_dir)
    samples = {s.question_id: s for s in ds.load_all()}
    records = read_records(args.run / "responses.jsonl")

    def render(conv, tools, gen):
        return proc.apply_chat_template(conv, tools=tools, add_generation_prompt=gen, tokenize=False, enable_thinking=False)

    n_conv, n_ex, skipped = 0, 0, Counter()
    cats = Counter()
    with args.out.open("w") as f:
        for qid, rec in records.items():
            s = samples.get(qid)
            if s is None or not ds.grade(s, rec):
                skipped["wrong or missing"] += 1
                continue
            key = hashlib.sha1(f"{s.image_path}\n{s.meta['default_prompt']}".encode()).hexdigest()[:16]
            path = args.dumps / f"{key}.json"
            if not path.exists():
                skipped["no dump"] += 1
                continue
            d = json.loads(path.read_text())
            if d["finish"] not in ("stop", None):
                skipped[f"finish={d['finish']}"] += 1
                continue
            msgs = d["messages"] + [{"role": "assistant", "content": [{"type": "text", "text": d["final_text"]}],
                                     "tool_calls": None, "tool_call_id": None}]
            conv, images = to_hf(msgs)
            tools = [{"type": "function", "function": t} for t in (d["tools"] or [])]
            ok = True
            rows = []
            for k, m in enumerate(conv):
                if m["role"] != "assistant":
                    continue
                prefix = render(conv[:k], tools, True)
                hist, full = render(conv[:k], tools, False), render(conv[: k + 1], tools, False)
                if not full.startswith(hist):
                    ok = False
                    break
                seg = full[len(hist):]
                head = "<|im_start|>assistant\n"
                if not seg.startswith(head):
                    ok = False
                    break
                target = seg[len(head):]
                if target.startswith("<think>\n\n</think>\n\n"):
                    target = target[len("<think>\n\n</think>\n\n"):]
                target = target.rstrip("\n").removesuffix("<|im_end|>")
                n_img = sum(1 for mm in conv[:k] if isinstance(mm.get("content"), list)
                            for p in mm["content"] if p.get("type") == "image")
                rows.append({"question_id": qid, "turn": len(rows), "chat_prompt": prefix, "images": images[:n_img],
                             "target": target, "thinking": False, "category": s.category})
            if not ok:
                skipped["template mismatch"] += 1
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_conv += 1
            n_ex += len(rows)
            cats[s.category] += 1
    report = {"conversations": n_conv, "examples": n_ex, "skipped": dict(skipped), "per_category": dict(cats)}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
