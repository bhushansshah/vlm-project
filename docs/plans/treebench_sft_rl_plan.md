# Plan: SFT and RL of Qwen3.5-4B for dense VQA (TreeBench)

Written 2026-11-26 (Md. Tanvir Arafat, drafted with Claude). Status: **draft for review; Tanvir asked for the first
runs to start overnight once the plan was written (2026-11-26), see "Run log" at the end.** Deadline 2026-12-10.
Background and sources: [teacher and data note](../research_notes/treebench_sft_teacher.md),
[training-data survey](../reports/Dense%20scene%20VQA%20training%20datasets.md). Facts below that come from
papers carry a link; everything else is a design decision.

## 1. The question, and the answer to "where do we test?"

We want to know (a) whether SFT on **thinking traces** beats SFT on **bare answers**, and (b) whether **GSPO**
beats **GRPO** in the RL stage that follows, for Qwen3.5-4B, with and without the tb3 tools.

TreeBench has only 405 questions, so it cannot be split into a useful train and test set. We follow
every paper that reports TreeBench (TreeVGR, DeepEyes, DeepEyesV2, Mini-o3): **train only on outside data,
and test on all 405 TreeBench questions** ([TreeVGR](https://arxiv.org/html/2507.07999),
[DeepEyes](https://arxiv.org/html/2505.14362), [Mini-o3](https://arxiv.org/html/2509.07969v1)). None of them
trains on TreeBench or cross-validates on it.

| Set | What | Used for | Never used for |
|---|---|---|---|
| **Train (SFT pool, RL pool)** | Outside questions converted to TreeBench's format (image, question, lettered options), split by image; SFT and RL pools disjoint | training | – |
| **Dev** (~500) | Held-out questions from the same outside sources, split by image, with TreeBench's category mix | choosing checkpoints, learning rates, number of steps | training |
| **Test: TreeBench** (405) | run once per final model, all 405 | the headline result | anything else: no filtering, prompt tuning or checkpoint choice on it |
| **Generalization** | V\* (191), HR-Bench 4K (200 questions × 4 option orders) | does it transfer | training |

Rules: training images are checked against TreeBench and V\* images (exact and perceptual hash) and any match
is dropped; TreeBench is SA-1B, so SA-1B-heavy training sets (ZwZ, PixelReasoner, Chain-of-Focus) are left out
of the first version. Results are reported with Wilson 95% CIs and paired exact McNemar tests against the
starting model; at n = 405 and ~50% accuracy the CI is about ±4.9 points, so **gaps under ~5 points are not
significant** and we say so.

**Secondary, clearly labelled experiment (week 2, if time):** an in-domain upper bound with 5-fold
cross-validation on TreeBench, folds stratified by category and **grouped by image** (395 distinct images;
10 images carry two questions). Each fold trains on ~324 questions and tests on ~81; the 405 out-of-fold
answers are pooled. Only the bare-answer vs trace comparison, with LoRA. It answers "how much can 324
in-domain examples teach?", never the headline.

## 2. Starting point (measured 2026-11-26, zero-shot, all 405)

| | 4B | 27B |
|---|---|---|
| No tools, thinking off | 42.7% | 50.1% |
| No tools, thinking on | 49.4% | 57.8% |
| tb3 agent, thinking off | 51.6% | 63.2% |
| tb3 agent, thinking on | 56.3% | 64.2% |

## 3. Data

**Version 1 (built first; all non-SA-1B, all convertible to multiple choice without a model):**

| Source | TreeBench skill | Conversion | Boxes |
|---|---|---|---|
| [SpatialMQA](https://huggingface.co/datasets/liuziyan/SpatialMQA) (3,780 train, COCO) | perspective ("if you were the man, where is X?") | options from its fixed direction set | no |
| [STVQA-7K](https://huggingface.co/datasets/OX-PIXL/STVQA-7K) (6,895 train, VG) | orientation, depth, distance, size, relation | already MC | yes |
| [TreeVGR-RL-37K](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K) VisDrone part (6,338) | counting / ordering in dense high-res scenes | already MC | yes |
| TreeVGR-RL-37K V\*-train part (COCO) | attributes, left/right | left-or-right → 2 options; colour → 4 options from a colour vocabulary | yes |
| Rendered copies (~10%) | OCR (TreeBench prints the question on the image) | draw question and options onto the image | inherited |

Target sizes: SFT pool ~3,000, RL pool ~3,000, dev ~500, balanced toward perspective, ordering and comparison.
Version 2 (week 2): box-generated ordering questions from COCO/LVIS, InstaOrder occlusion and depth order,
PACO materials, FineRS-4k attributes; ZwZ2 only after the SA-1B dedup.

## 4. SFT stage: bare answers vs thinking traces

Same training questions in every arm; only the targets differ.

| Arm | No-tools target | Agent target (week 2) |
|---|---|---|
| A0 | none (zero-shot) | none |
| A1 bare answer | empty think block + letter (thinking off) | tool calls + answer, no reasoning text |
| A2 teacher traces | Qwen3.5-27B thinking traces, correct only, shortest correct kept, cap 4K tokens | correct 27B tb3 trajectories, tool outputs masked from the loss |
| A3 self-distillation | the 4B's own correct thinking traces (rejection sampling) | the 4B's own correct trajectories |

Why A3: it tells whether a gain comes from the teacher or just from in-domain practice (rejection-sampling
fine-tuning helps weaker models most: [Yuan et al.](https://arxiv.org/abs/2308.01825)). Why traces may lose:
small models learn worse from long traces of strong teachers ([Li et al.](https://arxiv.org/abs/2502.12143))
and trace SFT can hurt later RL ([VLAA-Thinker](https://arxiv.org/abs/2504.11468)); that is what the
experiment tests.

Teacher traces are generated **without showing the answer**; the answer is used only to keep correct ones.
The 27B's identical tokenizer and chat template mean no reformatting.

Settings: TRL `SFTTrainer`, LoRA rank 64, alpha 128, all linear layers of the language model, vision tower
frozen, lr 1e-4, 2 epochs, effective batch 32, bf16, loss on assistant tokens only, max length 8K, images
capped at ~2 MP, **no sequence packing** (Qwen3.5's linear-attention layers leak state across packed samples
without the fla kernels: see [verl #8629](https://github.com/verl-project/verl/issues/8629)). LoRA matches
full fine-tuning at this data size ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)). Each arm
is evaluated in its own mode (A1 thinking off; A2/A3 thinking on) and also in the other mode.

## 5. RL stage: GRPO vs GSPO

Evidence: GSPO and GRPO tie on a 7B VLM without tools (54.7 vs 54.3, [Vero](https://arxiv.org/pdf/2604.04917));
in multi-turn tool RL plain GSPO did much worse than GRPO (V\* 53.9 vs 67.5) and only batch-normalised GSPO won
([SenseNova-MARS](https://arxiv.org/html/2512.24330v1)). GRPO on Qwen3.5-4B left TreeBench unchanged in one
paper ([ZoomGain](https://arxiv.org/html/2611.14076v1)). So the comparison is worth running, and the honest
expectation is a small difference.

| Held fixed in both arms | Differs |
|---|---|
| start checkpoint (A0 and the best SFT arm), prompt pool and order, prompts × G × steps, rewards, temperature 1.0, max lengths, LoRA rank and lr, 1 update per batch, eval | GRPO: token-level ratio, clip 0.2 / 0.28, token-mean loss. GSPO: sequence-level ratio, clip 3e-4 / 4e-4, sequence-mean loss |

Settings: TRL `GRPOTrainer` with vLLM colocated, LoRA rank 64, lr 1e-5 (≈10× the 1e-6 used for full
fine-tuning by Vero, DAPO and Mini-o3), G = 8, 32 prompts per step, ~150 steps, KL 0, reward = 1 for the right
letter + 0.1 for a well-formed answer. Prompts kept only if the starting model's pass rate over 8 samples is
strictly between 0 and 1 (DeepEyes). No-tools RL first with thinking off (max 1,024 new tokens, cheap), then
thinking on (max 4,096 with an overlong penalty). Agent RL (max 6 turns, tool tokens masked, a tool bonus
only when correct) in week 2 if the no-tools runs work; there GSPO's batch-normalised variant joins.
Logged every step: reward, clip fraction, entropy, response length, and the trainer/vLLM log-prob gap at
step 0 (Qwen3.5's linear attention uses different kernels in training and in vLLM; a large gap collapsed
GRPO in [NeMo-RL #5173](https://github.com/NVIDIA-NeMo/RL/issues/5173)). Two seeds per arm if time.

## 6. Stack

New venv `~/venvs/train-qwen35`: vLLM 0.34 + torch 2.14 + transformers 5.22.2 + TRL 1.19 + PEFT +
flash-linear-attention. TRL is the only framework found that supports Qwen3.5, GSPO
(`importance_sampling_level="sequence"`) and multi-turn tools returning images in one environment that works
with our vLLM. Fallbacks: ms-swift, then verl. Teacher traces and evaluation reuse `vlmbench` (same prompts as
the TreeBench runs) through a generic multiple-choice dataset.

## 7. Schedule

| When | What |
|---|---|
| Night of 11-26 | env; data v1 + dev split + dedup; 27B no-tools thinking traces on the SFT pool; 4B zero-shot on dev; SFT A1 and A2; evaluate on dev, then TreeBench |
| 11-27 – 11-30 | A3; RL pilot (10 steps, measure time and the log-prob gap); GRPO vs GSPO from A0 and best SFT arm, thinking off |
| 12-01 – 12-05 | agent SFT (A1/A2 agent targets); agent RL if time; thinking-on RL |
| 12-06 – 12-08 | final TreeBench + V\* + HR-Bench runs; optional TreeBench 5-fold upper bound |
| 12-09 – 12-10 | write-up |

## 8. Risks

- Trainer/vLLM kernel mismatch on linear attention breaks RL: check the step-0 log-prob gap; use fla.
- TRL multi-turn tools with images are not verified on Qwen3.5: test on a few samples before agent RL.
- Small gains are expected: GRPO on Qwen3.5-4B left TreeBench flat in ZoomGain; we report CIs and McNemar.
- Only GPUs 2 and 4 (shared machine): runs are sized for one GPU each.

## Run log

Times are ET.

**2026-11-26, night run (GPUs 2 and 4)**
- Environment `~/venvs/train-qwen35` built: vLLM 0.34, torch 2.14, transformers 5.22.2, TRL 1.19, PEFT 0.22,
  flash-linear-attention 0.6 (causal-conv1d not installed: torch fallback, slower but correct).
- Data v1 built (`training/build_data_v1.py` → `data/train_v1/`): SFT pool 2,950, RL pool 2,941, dev 447;
  disjoint by image (0 shared images); 4 training images dropped as TreeBench look-alikes (pHash ≤ 10).
  No material or physical-state items yet (v2).
- Pipeline verified end to end on 8 examples: LoRA SFT → merge (+ the `mtp.*` weights transformers drops) →
  vLLM serves the merged checkpoint → `vlmbench` grades it.
- Dev baselines, Qwen3.5-4B zero-shot: thinking **65.6%** (61.0–69.8), no thinking **61.5%** (56.9–65.9).
- Running: 27B teacher traces on the SFT pool (~25 questions/min, done ≈ 09:25); 4B self-traces; 4B thinking
  TreeBench baseline on H100 (the earlier 49.4% was on Quadro/float16).
- Queued (`training/overnight_v1.sh`): lane 4 = train + evaluate A3 (self), then A1 (bare); lane 2 = train +
  evaluate A2 (teacher). Each SFT model: dev and TreeBench, thinking on and off.
- RL scripts written, not yet run: `training/passrate.py`, `training/grpo.py` (GRPO vs GSPO; both use 4
  optimizer steps per generated batch so the ratio and clipping actually differ).

**SFT results (finished 10:13; LoRA r64, 2 epochs, lr 1e-4; each model evaluated once on all 405 TreeBench
questions, H100 bf16; p = exact McNemar vs the base model in the same mode)**

| Model | Train examples | TreeBench, thinking | TreeBench, no thinking | Dev, thinking | Dev, no thinking |
|---|---|---|---|---|---|
| Base Qwen3.5-4B | – | 52.3% | 42.7% | 65.6% | 61.5% |
| A1 bare answers | 1,688 | 0.0%* | 46.4% (p 0.15) | 0.0%* | **70.7%** |
| A2 27B teacher traces | 1,688 (same questions) | 51.6% (p 0.80) | **48.6% (p 0.006)** | 66.9% | 66.4% |
| A3 self-distillation | 1,592 | 51.9% (p 0.89) | 44.4% (p 0.46) | 68.2% | 60.9% |

\* After bare-answer SFT the model no longer thinks in thinking mode: it writes the letter and stops without
closing `<think>`, so the reply has no answer. The letter left in the open think block is right 47.2%
(TreeBench) / 69.6% (dev) of the time.

- Training mode ≠ test mode. A1 is trained with thinking off; A2 and A3 with thinking on. Every model is
  tested in both modes: the cell matching the training mode is "same mode", the other "cross-mode".
- Same mode, no arm beats the base model significantly on TreeBench: A1 (off/off) +3.7 (p 0.15); A2 and A3
  (on/on) −0.7 and −0.5 (p > 0.8). Dev rose in all three same-mode cells: in-domain gains do not transfer.
- The one significant TreeBench gain is cross-mode: **A2, trained with thinking on, tested with thinking off,
  +5.9** (48.6% vs 42.7%, 48 fixed, 24 broken). With the think block closed it reasons in the visible reply
  (91% of replies, right 48% of the time; base: 65%, right 33%), so this is cheaper visible reasoning, not
  answering without reasoning.
- A1 cross-mode (trained off, tested on) breaks: it writes the letter and never closes `<think>` (0%).
- Coverage gap: only 37 of 450 ordering questions survived in A2 (41 in A3): both models think too long on
  the dense VisDrone counting items (543 of 2,950 teacher answers hit the 8K cap). Data v2 needs ordering
  items with shorter reasoning, plus material and physical-state items.
- RL not started: the session ended after the SFT lanes finished; GPUs 2 and 4 idle from ~10:15.

**2026-11-26 afternoon: RL setup (GPU 4 only; GPU 2 taken by another user)**
- TRL 1.19 is not usable for Qwen3.5 image RL as is: its vLLM path shows a 0.13–0.37 per-token gap between
  sampling and training log-probs (healthy: 0.01–0.05, which vLLM and transformers reach outside TRL; kernels,
  image preprocessing, padding, weight sync, sleep mode and logit precision were each ruled out), and its
  transformers generation path crashes in Qwen3.5's `get_rope_index` when images of different sizes share a
  batch. Also fixed: `steps_per_generation` counts micro-batches, so the first config was fully on-policy.
- Replaced by `training/rl_loop.py`: one forward pass per prompt's G samples, sampling and scoring with the same
  transformers model (ratio exactly 1 on the on-policy step, 0.015–0.04 off-policy), GRPO and GSPO objectives
  written out. Settings: base model, thinking off, 8 prompts × 8 samples per step, 2 optimizer steps per
  generation round, 100 steps, max 256 new tokens (truncated samples masked), LoRA r64, lr 1e-5, KL 0.
- 16:44: `training/rl_v1.sh grpo gspo` started (≈2–2.5 h per arm + evaluation). Watch: 15–33% of samples hit the
  256-token cap early on.
- Queued after it: `training/tool_sft_v1.sh` (arm B1, tool-call SFT): 27B tb3 agent with thinking off on the SFT
  pool with conversation dumps (`agent_options.dump_dir`), one example per assistant turn from correct
  conversations (`training/make_agent_sft.py`), LoRA SFT with thinking off, evaluation as the tb3 agent.
