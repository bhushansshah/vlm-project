"""Pass rate of a model on each RL-pool question, over k samples, for prompt filtering (plan, section 5).

    python training/passrate.py --base-url http://localhost:18432/v1 --model Qwen/Qwen3.5-4B --k 8 \
        --out data/train_v1/passrate_base4b_nothink

Uses a running vLLM server (n=k samples per request), the same prompt as every other run (TreeBench format) and
the RL stage's sampling (temperature 1.0, top_p 1.0). Writes <out>.jsonl (question_id, category, correct of k)
and <out>.ids (questions with 0 < pass rate < 1: the ones a group of k samples can learn from).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from vlmbench.datasets import get_dataset
from vlmbench.datasets.choice import extract_choice_letter
from vlmbench.images import ImagePolicy, encode_image
from vlmbench.types import ImagePart


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/train_v1/rl"))
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 1024 (no thinking) / 4096 (thinking)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    max_tokens = args.max_tokens or (4096 if args.thinking else 1024)
    samples = get_dataset("tbstyle", args.data_dir).load_all()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    policy = ImagePolicy()

    def one(s):
        url = encode_image(ImagePart(s.image_path), policy).data_url
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": s.meta["default_prompt"]}]}]
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, messages=msgs, n=args.k, max_tokens=max_tokens, temperature=1.0, top_p=1.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}})
                break
            except Exception as e:  # noqa: BLE001 - retried, then recorded
                err = str(e)
        else:
            return {"question_id": s.question_id, "category": s.category, "error": err}
        ok = sum(c.finish_reason != "length" and extract_choice_letter(c.message.content, s.meta["options"]) == s.answer
                 for c in r.choices)
        return {"question_id": s.question_id, "category": s.category, "correct": ok, "k": args.k}

    with ThreadPoolExecutor(args.concurrency) as pool:
        res = list(pool.map(one, samples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    mid = [r for r in res if "correct" in r and 0 < r["correct"] < args.k]
    args.out.with_suffix(".ids").write_text("\n".join(r["question_id"] for r in mid) + "\n")
    hist = Counter(r.get("correct", "error") for r in res)
    print(json.dumps({"n": len(res), "kept_0<p<1": len(mid), "correct_of_k_histogram": dict(sorted(hist.items(), key=str)),
                      "kept_per_category": dict(Counter(r["category"] for r in mid))}, indent=1))


if __name__ == "__main__":
    main()
