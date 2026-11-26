# Teacher model and data for SFT + RL of Qwen3.5-4B on dense VQA (TreeBench)

As of 2026-11-26. Goal: improve Qwen3.5-4B (thinking on/off, with and without the tb3 tools) by SFT on
"gold" thinking traces from a stronger teacher, then RL. **TreeBench is the test set: no TreeBench question,
image or trace is used for training or for picking the teacher.** Model choices below use other benchmarks;
the TreeBench runs in this note only describe how the candidate behaves.

Tags: **[V]** checked against the primary page (HF card/API, arXiv), linked; **[est.]** our own estimate;
**[J]** judgement. Training-data background for the whole project:
[Dense scene VQA training datasets](../reports/Dense%20scene%20VQA%20training%20datasets.md).

## Recommendation [J]

1. **No-tools thinking traces: Qwen3.5-27B as the main teacher.** Its tokenizer, chat template and image
   preprocessor are byte-identical to Qwen3.5-4B's [V: sha256 of `chat_template.jinja`, `tokenizer.json`,
   `tokenizer_config.json`, `preprocessor_config.json` from the two HF repos], so its traces need no
   reformatting and its thinking style is the closest to the student's. Small students learn worse from
   much stronger long-CoT teachers ([Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143)).
   Use Qwen3.8-27B (or 3.6-27B) only as a second teacher for questions 3.5-27B gets wrong.
2. **Tool trajectories: Qwen3.5-27B (or 3.6-27B)** with the same tb3 tools and native `qwen3_coder` calls,
   so trajectories drop straight into the student's format.
3. **Filtering:** rejection sampling (keep only correct final answers; take the shortest of k correct
   samples; drop traces over ~6K tokens), plus a box check where the data has target boxes.
4. **Perspective questions will not be fixed by a teacher alone:** o3 scores 22.4% and Gemini-2.5-Pro 20.0%
   on TreeBench's perspective category ([TreeBench Table 2](https://arxiv.org/html/2507.07999v1)). Add
   allocentric left/right data (ViewSpatial, AlloEgo-View) and teach the agent to use `viewpoint`.

## Teacher candidates

| Model | Params | License | Serves on 1–2 H100 | Same tokenizer/template as 3.5-4B | Published scores |
|---|---|---|---|---|---|
| Qwen3.5-4B (student) | 4B | Apache-2.0 | 1 | – | V\* 84.3 (with CI) / 86.4 [V card] |
| **Qwen3.5-27B** | 27B dense | Apache-2.0 | 1 (bf16, tight) or 2; FP8 repo on 1 | **identical [V]** | V\* 93.7 / 89.0, CountBench 97.8, RefSpatial 67.7, ERQA 60.5 [V card]; TreeBench not published |
| Qwen3.6-27B | 27B dense | Apache-2.0 | same | tokenizer identical; template adds preserve_thinking | V\* 94.7, RefSpatial 70.0, ERQA 62.5 [V card] |
| Qwen3.8-27B | 27B dense | Apache-2.0 | same | same vocab; template injects a reasoning-effort system line (strip it) | ERQA 65.5, RealWorldQA 85.9; V\* not reported [V card] |
| Qwen3.5-35B-A3B | 35B / 3B active | Apache-2.0 | 1 (FP8) | same family | V\* 92.7 / 89.5 [V card] |
| Qwen3.5-122B-A10B | 122B / 10B active | Apache-2.0 | 2 (FP8, tight) [est.] | same family | V\* 93.2 / 90.1 [V card] |
| Qwen3.5-397B-A17B | 397B / 17B active | Apache-2.0 | no | same family | V\* 95.8 / 91.1 [V card] |
| o3 / Gemini-2.5-Pro | closed | API | – | no | TreeBench 54.8 / 54.1 [V paper] |

Model cards: [3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B).

## Training data that is not TreeBench

TreeBench and V\*Bench (test) images are SA-1B. Any training set built on SA-1B must be deduplicated against
TreeBench images (exact and perceptual hash) before use.

| Dataset | Use | Images | Contamination risk |
|---|---|---|---|
| [TreeVGR-SFT-35K](https://huggingface.co/datasets/HaochenWang/TreeVGR-SFT-35K) [V exists] | SFT, box-grounded traces | COCO, AI2D | low |
| [TreeVGR-RL-37K](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K) [V exists] | RL, with target boxes (V\* train + VisDrone counting) | COCO, VisDrone | low |
| [DeepEyes-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) [V exists] | RL for zoom agents | COCO and others | low |
| [SAT](https://huggingface.co/datasets/array/SAT) [V exists] | spatial QA | simulated (ProcTHOR) | low |
| ViewSpatial / AlloEgo-View | perspective (allocentric left/right) | ScanNet, COCO | low |
| PixelReasoner-SFT, Chain-of-Focus, ZwZ-RL-VQA | zoom traces / RL | **SA-1B** | **medium: dedup first** |

## Recipes reported on small Qwen VLMs (from the papers; not re-run)

- **TreeVGR** (Qwen2.5-VL-7B): SFT on ~35K box-grounded traces, then GRPO with reward = accuracy + format +
  box IoU (precision + recall); TreeBench +13.4 to 50.4 ([arXiv 2507.07999](https://arxiv.org/html/2507.07999v2)).
- **DeepEyes**: RL only; tool bonus paid only when the answer is correct; V\* 90.1 but TreeBench 37.5.
- **Mini-o3**: ~6K correct-only cold-start trajectories, then GRPO (group 16, no KL, over-turn masking).

## Measured on our setup

All 405 TreeBench questions, zero-shot, same prompt and tb3 agent (six tools) for both models. Qwen's
recommended sampling per mode (thinking: T 1.0, top_p 0.95; no thinking: T 0.7, top_p 0.8). H100, bf16,
vLLM 0.34, 2026-11-26, except the two 4B thinking runs (Quadro, float16, vLLM 0.32;
`docs/results/qwen35_4b_treebench.md`). These describe the models; they are not used to tune anything.

| Accuracy (95% CI) | Qwen3.5-4B | Qwen3.5-27B |
|---|---|---|
| No tools, thinking off | 42.7% (38.0–47.6) | 50.1% (45.3–55.0) |
| No tools, thinking on | 49.4% (44.5–54.2), Quadro | 57.8% (52.9–62.5) |
| tb3 agent, thinking off | 51.6% (46.7–56.4) | 63.2% (58.4–67.8) |
| tb3 agent, thinking on | 56.3% (51.4–61.0), Quadro | **64.2% (59.4–68.7)** |

| Category (n) | 4B no tools, think | 27B no tools, think | 4B agent, think | 27B agent, think | 27B agent, no think |
|---|---|---|---|---|---|
| Perspective transform (85) | 17.6 | 24.7 | 55.3 | 58.8 | 55.3 |
| Ordering (57) | 26.3 | **52.6** | 36.8 | 36.8 | 36.8 |
| Comparison (44) | 47.7 | 52.3 | 38.6 | 43.2 | 54.5 |
| Contact and occlusion (41) | 53.7 | 61.0 | 41.5 | 53.7 | 51.2 |
| OCR (68) | 82.4 | 86.8 | 80.9 | 86.8 | 91.2 |

What this says for distillation [J]:
- Tools fix perspective (left/right right 35% → 72% for the 27B) but **hurt ordering** (27B 52.6% → 36.8%:
  15 questions right only without tools, 6 only with) and comparison. Traces should be chosen per question
  type, not by mode: no-tools thinking traces for ordering/counting, tool trajectories for perspective.
- With tools, thinking adds 1 point for the 27B (63.2 → 64.2) at ~1.8× the output tokens; without tools it
  adds 7.7. For the 4B, thinking adds 6.7 (no tools) and 4.7 (agent).
- The 4B agent without thinking is the weakest tool user: 5.6 calls per question (27B: 3.4), 28 calls to a
  made-up "answer" tool, 18 replies without an answer tag, 17 failed viewpoint calls; perspective 35.3%
  against 55.3% with thinking. Tool-format SFT is most needed there.

## Not verified

TreeBench scores of any Qwen3.x-27B; HR-Bench/MME-RealWorld scores of the teachers; whether SAT covers
perspective-taking; the image sources of VisualProbe and Thyme; GPU fit figures marked [est.].
