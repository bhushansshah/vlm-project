"""In-process HuggingFace transformers backend.

Uses the model-agnostic ``AutoModelForImageTextToText`` + processor chat template path,
which covers most current open VLMs without per-model code, e.g.

    Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct, OpenGVLab/InternVL3-8B-hf,
    google/gemma-3-12b-it, llava-hf/llava-onevision-qwen2-7b-ov-hf,
    HuggingFaceTB/SmolVLM-Instruct, LiquidAI/LFM2-VL-1.6B, ...

Defaults follow the VisualOverload authors' setup: greedy decoding, max_new_tokens=512,
bf16, and the processor's default image resolution. Models whose remote code does not
implement the standard processor interface need a subclass overriding ``_prepare``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM

log = logging.getLogger(__name__)


class HFLocalVLM(VLM):
    """Options:
        device_map: passed to from_pretrained (default "auto": spread over visible GPUs)
        dtype: "bfloat16" (default) | "float16" | "float32" | "auto"
        attn_implementation: e.g. "flash_attention_2" or "sdpa"
        trust_remote_code: bool (default False)
        processor_kwargs: dict for AutoProcessor.from_pretrained, e.g. Qwen's
            {"max_pixels": 3584*3584} or {"min_pixels": ..., "max_pixels": ...}
        model_kwargs: extra dict for the model's from_pretrained
        default_max_new_tokens: used when GenerationConfig.max_tokens is None (default 512)
    """

    provider = "local"
    supports_concurrency = False  # one model on the GPU; use batch_size for throughput

    def __init__(
        self,
        model: str,
        device_map: str | dict = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        default_max_new_tokens: int = 512,
        **options: Any,
    ):
        super().__init__(
            model, device_map=device_map, dtype=dtype, attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code, processor_kwargs=processor_kwargs,
            model_kwargs=model_kwargs, default_max_new_tokens=default_max_new_tokens, **options,
        )
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=trust_remote_code, **(processor_kwargs or {})
        )
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"  # required for batched generation with decoder-only LMs

        mk: dict[str, Any] = {"device_map": device_map, "trust_remote_code": trust_remote_code}
        mk["dtype"] = dtype if dtype == "auto" else getattr(torch, dtype)
        if attn_implementation:
            mk["attn_implementation"] = attn_implementation
        mk.update(model_kwargs or {})
        self.hf_model = AutoModelForImageTextToText.from_pretrained(model, **mk).eval()
        self.default_max_new_tokens = default_max_new_tokens
        log.info("Loaded %s (device map: %s)", model, getattr(self.hf_model, "hf_device_map", None))

    @staticmethod
    def _to_hf(messages: list[Message]) -> list[dict[str, Any]]:
        conv = []
        for m in messages:
            content = []
            for p in m.content:
                if isinstance(p, TextPart):
                    content.append({"type": "text", "text": p.text})
                elif isinstance(p, ImagePart):
                    content.append({"type": "image", "image": p.load().convert("RGB")})
            conv.append({"role": m.role, "content": content})
        return conv

    def _prepare(self, conversations: list[list[Message]]):
        return self.processor.apply_chat_template(
            [self._to_hf(c) for c in conversations],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    def _gen_kwargs(self, config: GenerationConfig) -> dict[str, Any]:
        kw: dict[str, Any] = {"max_new_tokens": config.max_tokens or self.default_max_new_tokens}
        if config.temperature:  # None or 0 -> greedy
            kw.update(do_sample=True, temperature=config.temperature)
            if config.top_p is not None:
                kw["top_p"] = config.top_p
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        kw.update(config.extra)  # e.g. {"num_beams": 3, "repetition_penalty": 1.05}
        return kw

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        return self.generate_batch([messages], config)[0]

    def generate_batch(self, conversations: list[list[Message]], config: GenerationConfig) -> list[Completion]:
        torch = self.torch
        if config.seed is not None:
            torch.manual_seed(config.seed)
        inputs = self._prepare(conversations).to(self.hf_model.device)
        n_in = inputs["input_ids"].shape[1]
        gen_kwargs = self._gen_kwargs(config)
        with torch.inference_mode():
            out = self.hf_model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, n_in:]
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        attn = inputs.get("attention_mask")
        results = []
        for i, text in enumerate(texts):
            row = new_tokens[i]
            n_out = int((row != pad_id).sum()) if pad_id is not None else int(row.numel())
            n_prompt = int(attn[i].sum()) if attn is not None else n_in
            finish = "length" if n_out >= gen_kwargs["max_new_tokens"] else "stop"
            results.append(Completion(text=text.strip(), finish_reason=finish,
                                      usage=Usage(input_tokens=n_prompt, output_tokens=n_out)))
        return results
