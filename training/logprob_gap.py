"""Diagnose the gap between vLLM's sampling log-probs and the trainer's (transformers) log-probs for the same
tokens, which GRPO/GSPO's importance ratios depend on.

    CUDA_VISIBLE_DEVICES=4 python training/logprob_gap.py --n 6

For text-only prompts and for RL-pool image prompts: generate with vLLM (temperature 1, sampled-token
log-probs), re-score the exact prompt + completion token ids with transformers, and report the mean and max
|logp_vllm - logp_hf| per token, plus whether both engines produced the same prompt tokens (image token count).
A healthy gap is ~0.01 or less; a gap only on image prompts points at image preprocessing, a gap on text too
points at the model kernels (e.g. Qwen3.5's linear attention).
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", default="data/train_v1/rl/questions.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--sync", action="store_true", help="first push the transformers weights into vLLM the way TRL does")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(args.data)][: args.n]
    text_prompts = ["Explain in three sentences why the sky is blue.", "List five prime numbers and say why each is prime.",
                    "Write a short paragraph about trees in a city park."][: max(1, args.n // 2)]

    def chat(text, with_image):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
        return proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False, enable_thinking=args.thinking)

    llm = LLM(model=args.model, gpu_memory_utilization=0.35, max_model_len=16384, max_num_seqs=16, seed=0,
              limit_mm_per_prompt={"image": 1})
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.sync:  # TRL's colocate sync: one (name, tensor) pair at a time into the vLLM model
        vm = llm.llm_engine.model_executor.driver_worker.model_runner.model
        for name, param in model.named_parameters():
            vm.load_weights([(name, param.data)])
        llm.reset_prefix_cache()
        print("synced weights into vLLM the way TRL does")
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=0, seed=0)
    reqs = [{"prompt": chat(t, False)} for t in text_prompts]
    reqs += [{"prompt": chat(r["default_prompt"], True), "multi_modal_data": {"image": Image.open(r["image"]).convert("RGB")}}
             for r in rows]
    outs = llm.generate(reqs, sp)

    for i, (req, out) in enumerate(zip(reqs, outs)):
        comp = out.outputs[0]
        v_lp = torch.tensor([list(d.values())[0].logprob if comp.token_ids[j] in d else float("nan")
                             for j, d in enumerate(comp.logprobs)])
        img = req.get("multi_modal_data", {}).get("image")
        enc = proc(text=[req["prompt"]], images=[img] if img is not None else None, return_tensors="pt")
        hf_prompt = enc["input_ids"][0].tolist()
        same_prompt = hf_prompt == list(out.prompt_token_ids)
        ids = torch.tensor(hf_prompt + list(comp.token_ids))[None].cuda()
        kw = {k: v.cuda() for k, v in enc.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        if "mm_token_type_ids" in enc:
            kw["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"][0],
                                                 torch.zeros(len(comp.token_ids), dtype=enc["mm_token_type_ids"].dtype)])[None].cuda()
        with torch.no_grad():
            raw = model(input_ids=ids, attention_mask=torch.ones_like(ids), **kw).logits[0]
        tok = torch.tensor(comp.token_ids).cuda()[:, None]
        hf_lp = torch.log_softmax(raw.float()[len(hf_prompt) - 1:-1], -1).gather(1, tok)[:, 0].cpu()
        hf_lp_bf16 = torch.log_softmax(raw.bfloat16()[len(hf_prompt) - 1:-1], -1).gather(1, tok)[:, 0].float().cpu()
        d = (v_lp - hf_lp).abs()
        d16 = (v_lp - hf_lp_bf16).abs()
        print(json.dumps({"kind": "image" if img is not None else "text", "prompt_tokens_vllm": len(out.prompt_token_ids),
                          "prompt_tokens_hf": len(hf_prompt), "same_prompt_ids": same_prompt, "completion_tokens": len(comp.token_ids),
                          "mean_abs_gap": round(float(d.nanmean()), 4), "max_abs_gap": round(float(d.nan_to_num(0).max()), 3),
                          "mean_abs_gap_bf16_logsoftmax": round(float(d16.nanmean()), 4), "logits_dtype": str(raw.dtype)}))


if __name__ == "__main__":
    main()
