"""GRPO vs GSPO for Qwen3.5-4B on TreeBench-format multiple choice, no tools (plan, section 5).

    CUDA_VISIBLE_DEVICES=4 python training/grpo.py --algo gspo --init Qwen/Qwen3.5-4B --out checkpoints/rl_gspo_base

The two arms differ ONLY in the policy-gradient objective:
- grpo: token-level importance ratios, clip 0.2 / 0.28 (clip-higher), token-mean loss over the batch (DAPO
  aggregation; TRL's default "dapo" loss type).
- gspo: sequence-level ratio (length-normalised), clip 3e-4 / 4e-4, sequence-mean loss (GSPO paper).
Everything else is shared: start checkpoint, prompts and their order, G = 8 samples per prompt, 32 prompts per
optimizer step, 4 optimizer steps per generated batch (so updates are off-policy and the ratios/clipping matter;
with 1 step per batch every ratio is 1 and the two arms nearly coincide), KL 0, LoRA rank 64, lr 1e-5,
temperature 1.0, rewards. Generation runs in vLLM inside the trainer process (colocate, sleep mode).

Reward = 1 if the extracted option letter is the gold letter, + 0.1 if the reply ends with a readable letter
(format). In thinking mode the letter is read after </think>; a reply without </think> (cut off) gets 0.

Prompts: data/train_v1/rl (disjoint from the SFT pool and dev, by image), optionally restricted to the ids in
--prompt-ids (e.g. those the start model gets right sometimes but not always: training/passrate.py).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def extract_letter(text: str, options) -> str | None:
    """The option letter a reply commits to, read exactly as evaluation reads it (vlmbench's extractor), from
    the text after </think> when the reply has one."""
    from vlmbench.datasets.choice import extract_choice_letter

    return extract_choice_letter(text.split("</think>")[-1].strip(), options)


def make_rewards(thinking: bool):
    def text_of(c):
        return c[0]["content"] if isinstance(c, list) else c

    def correct(completions, answer, options, **_):
        out = []
        for c, a, o in zip(completions, answer, options):
            t = text_of(c)
            if thinking and "</think>" not in t:
                out.append(0.0)
                continue
            out.append(1.0 if extract_letter(t, o) == a else 0.0)
        return out

    def fmt(completions, options, **_):
        out = []
        for c, o in zip(completions, options):
            t = text_of(c)
            ok = (not thinking or "</think>" in t) and extract_letter(t, o) is not None
            out.append(0.1 if ok else 0.0)
        return out

    return [correct, fmt]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["grpo", "gspo"], required=True)
    ap.add_argument("--init", default="Qwen/Qwen3.5-4B", help="base model or a merged SFT checkpoint")
    ap.add_argument("--data", type=Path, default=Path("data/train_v1/rl/questions.jsonl"))
    ap.add_argument("--prompt-ids", type=Path, help="file with one question_id per line to train on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thinking", action="store_true", help="thinking on (default off)")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--steps-per-generation", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=None, help="default 1024 (no thinking) / 4096 (thinking)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=8, help="completions per forward/backward pass")
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sleep", action="store_true", help="keep vLLM awake between generation rounds")
    ap.add_argument("--max-pixels", type=int, default=None, help="cap image size during RL (pixels), e.g. 1000000")
    ap.add_argument("--no-vllm", action="store_true",
                    help="generate with the transformers model (slower, but sampling and training log-probs then match)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset, Image as HFImage
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(line) for line in args.data.open()]
    if args.prompt_ids:
        keep = set(args.prompt_ids.read_text().split())
        rows = [r for r in rows if r["question_id"] in keep]
    random.Random(args.seed).shuffle(rows)
    ds = Dataset.from_list([{
        "prompt": [{"role": "user", "content": r["default_prompt"]}],
        "image": r["image"], "answer": r["answer"], "options": r["options"], "question_id": r["question_id"],
        "category": r["category"]} for r in rows]).cast_column("image", HFImage())

    max_completion = args.max_completion or (4096 if args.thinking else 1024)
    G, P = args.num_generations, args.prompts_per_step
    algo = (dict(importance_sampling_level="sequence", loss_type="grpo", epsilon=3e-4, epsilon_high=4e-4)
            if args.algo == "gspo" else
            dict(importance_sampling_level="token", loss_type="dapo", epsilon=0.2, epsilon_high=0.28))
    cfg = GRPOConfig(
        output_dir=str(args.out), seed=args.seed, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=10, beta=0.0, bf16=True,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=P * G // args.micro_batch,
        num_generations=G, num_iterations=1,
        # TRL counts steps_per_generation in micro-batches: one generation round = args.steps_per_generation optimizer steps
        steps_per_generation=(P * G // args.micro_batch) * args.steps_per_generation,
        max_completion_length=max_completion, temperature=1.0, top_p=1.0,
        chat_template_kwargs={"enable_thinking": args.thinking},
        use_vllm=not args.no_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_enable_sleep_mode=not args.no_sleep, vllm_max_model_length=max_completion + 8192,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        mask_truncated_completions=True, reward_weights=[1.0, 1.0],
        logging_steps=1, save_strategy="no", report_to=[], log_completions=False,
        model_init_kwargs={"dtype": torch.bfloat16}, **algo)
    peft = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                                     r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.init)
    if args.max_pixels:  # Qwen's image processor: longest_edge is the pixel budget
        processor.image_processor.size = {"longest_edge": args.max_pixels, "shortest_edge": 65536}
    trainer = GRPOTrainer(model=args.init, args=cfg, reward_funcs=make_rewards(args.thinking), train_dataset=ds,
                          peft_config=peft, processing_class=processor)
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.out / "adapter")
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (args.out / "run_args.json").write_text(json.dumps({**vars(args), "data": str(args.data), "out": str(args.out),
                                                        "prompt_ids": str(args.prompt_ids), "n_prompts": len(rows),
                                                        "trl_algo": algo}, indent=1))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out / "merged", safe_serialization=True)
    trainer.processing_class.save_pretrained(args.out / "merged")
    from sft import copy_unloaded_weights  # same folder

    copy_unloaded_weights(args.init, args.out / "merged")


if __name__ == "__main__":
    main()
