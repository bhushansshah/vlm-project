"""Write a short referring description for each object a v2 question names ("the man in a blue jacket").

TreeBench names its objects with descriptions, not boxes. Each object is shown to a VLM (Qwen3.5-4B, thinking
off, served by vLLM) as a crop with some context and the object outlined in red; the reply is one noun phrase.
Only appearance is asked for: the VLM never sees the question or its answer. Replies that are not a short
phrase starting with "the" are dropped (that question is then left out). Resumable.

    python training/data_v2/describe.py --objects data/train_v2/candidates.objects.jsonl \
        --out data/train_v2/descriptions.jsonl --base-url http://127.0.0.1:18431/v1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw

PROMPT = ("The red box marks one {label} in a photo. If what is in the box is not a real {label} (for example a "
          "picture or poster of one, a statue, a mannequin, a toy, or a reflection), reply exactly: NOT REAL. "
          "Otherwise write a short noun phrase (at most 10 words) that describes that {label} by how it looks, so "
          "someone could pick it out in the full photo: colours, clothing, what it is holding or doing. Start with "
          "\"the\". Do not mention the red box, the image or its position. Reply with the phrase only.")
# The "NOT REAL" check was added after the hand check of early SA-1B candidates: the detector calls a photo of a
# person on an easel, and a shop-window mannequin, "person"; questions about them have no real answer.
BAD = re.compile(r"\b(red box|box|image|photo|picture|left|right|center|centre|corner)\b", re.I)


def crop(path: str, box) -> str:
    img = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = max(w, h) * 0.6 + 40
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(img.width, x2 + pad), min(img.height, y2 + pad)
    c = img.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
    d = ImageDraw.Draw(c)
    lw = max(2, int(min(c.size) / 150))
    d.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline=(255, 0, 0), width=lw)
    if max(c.size) > 1024:
        c.thumbnail((1024, 1024))
    buf = io.BytesIO()
    c.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def clean(text: str) -> str | None:
    t = text.strip().strip('"').strip().rstrip(".").strip()
    t = t.split("\n")[0]
    if not t.lower().startswith("the ") or len(t.split()) > 12 or len(t.split()) < 2 or BAD.search(t):
        return None
    return "the " + t[4:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18431/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    todo = [json.loads(line) for line in args.objects.open()]
    done = set()
    if args.out.exists():
        done = {(d["image"], d["object"]) for d in map(json.loads, args.out.open())}
    todo = [o for o in todo if (o["image"], o["object"]) not in done]
    client = OpenAI(base_url=args.base_url, api_key="none", timeout=600)
    lock = threading.Lock()
    f = args.out.open("a")

    def one(o):
        try:
            r = client.chat.completions.create(
                model=args.model, temperature=0.2, max_tokens=40,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + crop(o["image"], o["box"])}},
                    {"type": "text", "text": PROMPT.format(label=o["label"])}]}])
            raw = r.choices[0].message.content or ""
        except Exception as e:  # keep going; the object just gets no description
            raw = f"ERROR {e!r}"
        rec = {"image": o["image"], "object": o["object"], "label": o["label"], "raw": raw, "description": clean(raw)}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    f.close()
    print(f"described {len(todo)} objects -> {args.out}")


if __name__ == "__main__":
    main()
