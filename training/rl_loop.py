"""Minimal GRPO / GSPO trainer for Qwen3.5-4B on TreeBench-format multiple choice, no tools (plan, section 5).

    CUDA_VISIBLE_DEVICES=4 python training/rl_loop.py --algo gspo --out checkpoints/rl_gspo_base

Why not TRL: TRL 1.19 mis-aligns Qwen3.5's multimodal positions when images of different sizes share a batch
(its transformers generation path crashes in get_rope_index; its vLLM path shows a 0.13-0.37 per-token gap
between sampling and training log-probs). This loop avoids both problems by construction:
- one forward pass = the G samples of ONE prompt (same image, same prompt length; only right padding after the
  completions, which cannot affect earlier tokens in a causal model, linear-attention layers included);
- the same transformers model samples and scores, so importance ratios start at exactly 1.

Shared by both arms: start model, prompt order, G = 8 samples per prompt, P prompts per optimizer step,
R optimizer steps per generation round (updates 2..R are off-policy, which is where the objectives differ),
group-normalised advantages (r - mean) / (std + 1e-4), KL 0, LoRA rank 64 / lr 1e-5 / AdamW, temperature 1.0,
rewards (1 for the gold letter + 0.1 for a readable letter; truncated completions are masked out).
- grpo: token ratio r_t = exp(lp - lp_old); loss_t = -min(r_t A, clip(r_t, 1-0.2, 1+0.28) A); token-mean over the
  step's tokens (DAPO aggregation).
- gspo: sequence ratio s = exp(mean_t(lp - lp_old)); loss = -min(s A, clip(s, 1-3e-4, 1+4e-4) A); mean over
  the step's sequences (GSPO paper).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

LORA_TARGETS = (r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def reward(text: str, options, gold: str, thinking: bool) -> tuple[float, float]:
    from vlmbench.datasets.choice import extract_choice_letter

    if thinking and "</think>" not in text:
        return 0.0, 0.0
    letter = extract_choice_letter(text.split("</think>")[-1].strip(), options)
    return (1.0 if letter == gold else 0.0), (0.1 if letter is not None else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["grpo", "gspo"], required=True)
    ap.add_argument("--init", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", type=Path, default=Path("data/train_v1/rl/questions.jsonl"))
    ap.add_argument("--prompt-ids", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--steps", type=int, default=100, help="optimizer steps")
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--steps-per-round", type=int, default=2, help="optimizer steps per generation round")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in args.data.open()]
    if args.prompt_ids:
        keep = set(args.prompt_ids.read_text().split())
        rows = [r for r in rows if r["question_id"] in keep]
    random.Random(args.seed).shuffle(rows)

    proc = AutoProcessor.from_pretrained(args.init)
    model = AutoModelForImageTextToText.from_pretrained(args.init, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    eos_ids = {proc.tokenizer.convert_tokens_to_ids("<|im_end|>"), proc.tokenizer.eos_token_id}
    pad_id = proc.tokenizer.pad_token_id or 0
    lo, hi = (0.2, 0.28) if args.algo == "grpo" else (3e-4, 4e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    log_f = (args.out / "log.jsonl").open("w")

    def encode(r):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["default_prompt"]}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=args.thinking)
        return proc(text=[text], images=[Image.open(r["image"]).convert("RGB")], return_tensors="pt")

    def seq_logps(enc, comp_ids, comp_mask):
        """Per-token log-probs of the completions (G x T) for one prompt's G samples."""
        G, T = comp_ids.shape
        ids = torch.cat([enc["input_ids"].expand(G, -1), comp_ids], 1)
        att = torch.cat([enc["attention_mask"].expand(G, -1), comp_mask], 1)
        mm = torch.cat([enc["mm_token_type_ids"].expand(G, -1), torch.zeros_like(comp_ids)], 1)
        out = model(input_ids=ids, attention_mask=att, mm_token_type_ids=mm,
                    pixel_values=enc["pixel_values"].repeat(G, 1), image_grid_thw=enc["image_grid_thw"].repeat(G, 1),
                    logits_to_keep=T + 1)
        logits = out.logits[:, :-1].float()  # predictions for the T completion tokens
        return torch.log_softmax(logits, -1).gather(2, comp_ids[..., None])[..., 0]

    step, cursor, t0 = 0, 0, time.time()
    rounds = math.ceil(args.steps / args.steps_per_round)
    for rnd in range(rounds):
        # ---- generate one round: steps_per_round x prompts_per_step prompts, G samples each
        model.eval()
        groups = []
        for _ in range(args.steps_per_round * args.prompts_per_step):
            r = rows[cursor % len(rows)]
            cursor += 1
            enc = {k: v.cuda() for k, v in encode(r).items()}
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     max_new_tokens=args.max_new, num_return_sequences=args.G, use_cache=True,
                                     pad_token_id=pad_id)
            comp = out[:, enc["input_ids"].shape[1]:]
            mask = torch.zeros_like(comp)
            keep_seq = torch.zeros(args.G, dtype=torch.bool)
            texts, rew = [], []
            for i in range(args.G):
                ends = [j for j, t in enumerate(comp[i].tolist()) if t in eos_ids]
                if ends:
                    mask[i, : ends[0] + 1] = 1
                    keep_seq[i] = True
                    texts.append(proc.tokenizer.decode(comp[i, : ends[0]]))
                else:  # truncated: no reward signal, masked out
                    texts.append(proc.tokenizer.decode(comp[i]))
                c, f = reward(texts[-1], r["options"], r["answer"], args.thinking)
                rew.append(c + f)
            rew_t = torch.tensor(rew)
            adv = (rew_t - rew_t.mean()) / (rew_t.std() + 1e-4)
            mask[~keep_seq.cuda()] = 0
            comp = comp[:, : int(mask.sum(1).max().item()) or 1]
            mask = mask[:, : comp.shape[1]]
            with torch.no_grad():
                old = seq_logps(enc, comp, mask)
            groups.append({"enc": enc, "comp": comp, "mask": mask, "old": old, "adv": adv.cuda(),
                           "correct": float(sum(reward(t, r["options"], r["answer"], args.thinking)[0] for t in texts)) / args.G,
                           "len": float(mask.sum(1).float().mean()), "trunc": float((~keep_seq).float().mean())})
        # ---- R optimizer steps over the round (step 1 on-policy, the rest off-policy)
        model.train()
        for s in range(args.steps_per_round):
            chunk = groups[s * args.prompts_per_step:(s + 1) * args.prompts_per_step]
            n_tok = sum(float(g["mask"].sum()) for g in chunk) or 1.0
            n_seq = sum(int((g["mask"].sum(1) > 0).sum()) for g in chunk) or 1
            clipped, ratio_dev, used = 0.0, 0.0, 0
            for g in chunk:
                if float(g["adv"].abs().sum()) == 0 or float(g["mask"].sum()) == 0:
                    continue  # all samples got the same reward: no signal
                lp = seq_logps(g["enc"], g["comp"], g["mask"])
                m = g["mask"].float()
                A = g["adv"][:, None]
                if args.algo == "grpo":
                    ratio = torch.exp(lp - g["old"])
                    surr = torch.minimum(ratio * A, torch.clamp(ratio, 1 - lo, 1 + hi) * A)
                    loss = -(surr * m).sum() / n_tok
                    clipped += float((((ratio < 1 - lo) | (ratio > 1 + hi)).float() * m).sum()) / n_tok
                    ratio_dev += float(((ratio - 1).abs() * m).sum()) / n_tok
                else:
                    valid = m.sum(1) > 0
                    log_s = ((lp - g["old"]) * m).sum(1) / m.sum(1).clamp(min=1)
                    s_i = torch.exp(log_s)
                    surr = torch.minimum(s_i * g["adv"], torch.clamp(s_i, 1 - lo, 1 + hi) * g["adv"])
                    loss = -(surr * valid).sum() / n_seq
                    clipped += float((((s_i < 1 - lo) | (s_i > 1 + hi)) & valid).float().sum()) / n_seq
                    ratio_dev += float(((s_i - 1).abs() * valid).sum()) / n_seq
                loss.backward()
                used += 1
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "round": rnd, "off_policy": s > 0, "reward_correct": sum(g["correct"] for g in chunk) / len(chunk),
                   "groups_with_signal": used, "clip_frac": clipped, "mean_abs_ratio_minus_1": ratio_dev,
                   "completion_len": sum(g["len"] for g in chunk) / len(chunk),
                   "truncated": sum(g["trunc"] for g in chunk) / len(chunk), "grad_norm": gnorm,
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(json.dumps(rec), flush=True)
            if step >= args.steps:
                break

    model.save_pretrained(args.out / "adapter")
    (args.out / "run_args.json").write_text(json.dumps({**vars(args), "data": str(args.data), "out": str(args.out),
                                                        "prompt_ids": str(args.prompt_ids), "n_prompts": len(rows),
                                                        "clip": [lo, hi]}, indent=1))
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out / "merged", safe_serialization=True)
    proc.save_pretrained(args.out / "merged")
    from sft import copy_unloaded_weights  # same folder

    copy_unloaded_weights(args.init, args.out / "merged")


if __name__ == "__main__":
    main()
