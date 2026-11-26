"""LoRA SFT of Qwen3.5-4B on TreeBench-format examples (plan: docs/plans/treebench_sft_rl_plan.md, section 4).

    CUDA_VISIBLE_DEVICES=4 python training/sft.py --data data/sft_arms/a1_bare.jsonl --out checkpoints/a1_bare

Each example is {"image", "prompt", "target", "thinking": bool}. The model sees exactly what vLLM serves at
evaluation: the chat template's generation prompt (with enable_thinking on or off), then the target, then
<|im_end|>. Loss is on the target tokens only. One example per micro-batch and no sequence packing: without
the right kernels, Qwen3.5's linear-attention layers carry state across packed samples.

After training, the LoRA adapter is merged into a full bf16 checkpoint (<out>/merged) that vLLM can serve.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments,
                          set_seed)

LORA_TARGETS = (r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], processor, max_len: int):
        self.rows, self.proc, self.max_len = rows, processor, max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        if "chat_prompt" in r:  # multi-turn agent step: prompt already rendered (system, tools, earlier turns)
            images = [Image.open(p).convert("RGB") for p in r["images"]]
            prompt = r["chat_prompt"]
        else:
            images = [Image.open(r["image"]).convert("RGB")]
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
            prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                   enable_thinking=bool(r["thinking"]))
        full = prompt + r["target"] + "<|im_end|>\n"
        enc = self.proc(text=[full], images=images, return_tensors="pt")
        n_prompt = self.proc(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
        ids = enc["input_ids"][0]
        labels = ids.clone()
        labels[:n_prompt] = -100
        mask, mm = enc["attention_mask"][0], enc["mm_token_type_ids"][0]
        if ids.shape[0] > self.max_len:  # cuts only the target's tail (the prompt is shorter than max_len)
            ids, labels, mask, mm = (x[: self.max_len] for x in (ids, labels, mask, mm))
        out = {"input_ids": ids, "labels": labels, "attention_mask": mask, "mm_token_type_ids": mm}
        for k in ("pixel_values", "image_grid_thw"):
            out[k] = enc[k]
        return out


def collate(batch: list[dict]) -> dict:
    assert len(batch) == 1, "one example per micro-batch (no padding, no packing)"
    b = batch[0]
    return {"input_ids": b["input_ids"][None], "labels": b["labels"][None],
            "attention_mask": b["attention_mask"][None], "mm_token_type_ids": b["mm_token_type_ids"][None],
            "pixel_values": b["pixel_values"],
            "image_grid_thw": b["image_grid_thw"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=32, help="effective batch size (micro-batch is 1)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="first N examples only (smoke test)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    rows = [json.loads(line) for line in args.data.open()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps = math.ceil(len(rows) * args.epochs / args.grad_accum)
    targs = TrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * steps)), weight_decay=0.0, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to=[], seed=args.seed, dataloader_num_workers=4,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=SFTData(rows, processor, args.max_len),
                      data_collator=collate)
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter")
    (args.out / "train_args.json").write_text(json.dumps({**vars(args), "data": str(args.data), "out": str(args.out),
                                                          "n_examples": len(rows), "optimizer_steps": steps}, indent=1))
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out / "merged", safe_serialization=True)
        processor.save_pretrained(args.out / "merged")
        copy_unloaded_weights(args.model, args.out / "merged")
        print(f"merged model: {args.out / 'merged'}")


def copy_unloaded_weights(base: str, merged_dir: Path) -> None:
    """transformers does not load Qwen3.5's multi-token-prediction head (mtp.*), so the merged checkpoint
    lacks it. Copy those tensors, unchanged (LoRA never touches them), from the base checkpoint so the merged
    folder has the release's full set of weights."""
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    have = set()
    for f in glob.glob(str(merged_dir / "*.safetensors")):
        with safe_open(f, "pt") as s:
            have |= set(s.keys())
    extra = {}
    base_dir = base if Path(base).is_dir() else snapshot_download(base)  # a local merged checkpoint or a Hub id
    for f in glob.glob(str(base_dir) + "/*.safetensors"):
        with safe_open(f, "pt") as s:
            extra.update({k: s.get_tensor(k) for k in s.keys() if k not in have})
    if extra:
        save_file(extra, str(merged_dir / "model-extra.safetensors"), metadata={"format": "pt"})
        index = {"metadata": {}, "weight_map": {}}
        for f in sorted(glob.glob(str(merged_dir / "*.safetensors"))):
            with safe_open(f, "pt") as s:
                index["weight_map"].update({k: Path(f).name for k in s.keys()})
        (merged_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
        print(f"copied {len(extra)} tensors the merge dropped (e.g. {sorted(extra)[0]})")


if __name__ == "__main__":
    main()
