# Do tools help VLMs on TreeBench? A tool study

TreeBench: 405 multiple-choice questions on dense photos, 10 categories (5 perception, 5 reasoning),
public answers and target boxes. Models via OpenRouter (bf16 providers, greedy). Runs: 26-27 Oct 2026.
Full paired tables: [results/treebench_agent_analysis.md](results/treebench_agent_analysis.md).

## Summary

- **Tools help all three models, by a large margin.** With the TreeBench agent (tb2 instructions and
  tools), accuracy rises from 59.8 to 68.9 (Qwen3.8-27B, thinking low), 54.3 to 65.2 (Muse Glimmer 30B)
  and 43.7 to 62.0 (Gemma 4 26B-A4B). All three gains are significant on paired tests (p < 0.001) and
  hold without the 40 tuning questions.
- **Purpose-built tools beat generic ones.** Bhushan's VistaQA agent (v2: zoom, grid, detect, ocr, depth)
  also helps, but less: +5.9, +2.5 (not significant) and +8.1. tb2 beats v2 for Muse (+8.4, p < 0.001) and
  Gemma (+10.1, p < 0.001); for Qwen the difference (+3.2) is not significant.
- **The viewpoint tools fix perspective questions.** Perspective transform (85 questions, the largest
  category) goes from 16.5 to 52.9 (Muse), 18.8 to 49.4 (Gemma) and 48.2 to 64.7 (Qwen); v2 barely moves
  it. The left/right swaps and mirror errors from the failure analysis drop sharply.
- **For perception, zoom does the work.** Gemma, which gets a heavily compressed image, gains +23 points
  on perception with either agent. A zoom-only Qwen agent matches the full tool set on perception
  questions (79.2 vs 81.2, not significant).
- **Cost:** a full 405-question agent run cost $2.33 (Qwen), $2.12 (Muse) and $0.72 (Gemma). All tool
  models ran on CPU.

## 1. What the dataset needs

From the base-model failure analysis and a keyword study of all 405 questions:

| category (n) | what the questions ask | how base models failed | tool designed for it |
|---|---|---|---|
| Perspective transform (85) | 80 ask "from X's viewpoint, where is Y" (front/rear x left/right); ~70 reference a person | answered from the camera's view: left/right swaps and mirrors were ~90% of errors | `orient` + `viewpoint` |
| Ordering (57) | 49 count positions "from left/right"; 26 right-to-left | wrong answers mostly off by one position | `order` |
| Comparison (44) | 25 closer/farther, 27 colour, 12 size | weak (43-56%) | `depth`, `distance`, `colour` |
| Contact & occlusion (41) | 17 touching/blocking, 13 depth | weak | `depth`, `zoom` |
| Perception (149) | small-target colour, material, state, printed questions | Gemma saw a compressed image | `zoom`, `enhance`, `colour` |

## 2. Tools

Bhushan's tools: `zoom`, `grid`, `detect` (Grounding DINO tiny), `depth` (Depth Anything V2 small),
`distance`, `ocr` (EasyOCR). New for TreeBench:

- **`viewpoint`**: moves the camera to a reference object. Places the reference and targets on a top-down
  map (sideways position from the image, distance from estimated depth), rotates it so the reference
  faces up, and reports each target as front/rear and left/right *of the reference*, with the map image
  and a camera-position check.
- **`orient`**: which way a person faces, from ViTPose keypoints (shoulder order and width, nose
  position), as an angle for `viewpoint`.
- **`order`**: detects every instance and numbers them left to right, right to left, top/bottom or
  near/far, drawing "1st, 2nd, ..." on the image.
- **`enhance`**: Swin2SR x4 super-resolution of small regions (up to 320 px).
- **`colour`**: named dominant colours of up to 4 boxes.

Considered and not built: novel-view synthesis (too heavy and unreliable for cluttered scenes; the
top-down map does the viewpoint change geometrically) and segmentation for contact (2D mask contact is
not 3D contact; 17 questions).

## 3. Setup

| condition | instructions | tools |
|---|---|---|
| base | the benchmark prompt, answer with the letter | none |
| v2 (Bhushan) | his VistaQA v2 | zoom, grid, detect, ocr, depth |
| tb2 (TreeBench agent) | no "not in the image" answers, a procedure per question type, evidence boxes then one letter | zoom, grid, enhance, detect, order, depth, distance, colour, orient, viewpoint |

Each agent uses exactly its model's baseline settings: Qwen thinking low, Muse thinking low (required by
its provider), Gemma no thinking (its thinking loops). Up to 6 tool calls per question. tb1/tb2 were
developed on 40 tuning questions (4 per category, `analysis/treebench_tuning_ids.txt`); one tuning round
tied (57.5 vs 57.5) and tb2 was frozen before the full runs.

## 4. Results

Accuracy (%), all 405 questions; p = exact McNemar test against the base model on the same questions.

| model | base | v2 agent | tb2 agent | tb2 vs base | v2 vs base |
|---|---|---|---|---|---|
| Qwen3.8-27B (think low) | 59.8 | 65.7 | **68.9** | +9.1 (66 fixed / 29 broken, p = 0.0002) | +5.9 (p = 0.008) |
| Muse Glimmer 30B (think low) | 54.3 | 56.8 | **65.2** | +10.9 (69 / 25, p < 0.0001) | +2.5 (p = 0.30) |
| Gemma 4 26B-A4B (no think) | 43.7 | 51.9 | **62.0** | +18.3 (105 / 31, p < 0.0001) | +8.1 (p = 0.002) |

By category (Q = Qwen, M = Muse, G = Gemma):

| category | n | Q base | Q v2 | Q tb2 | M base | M v2 | M tb2 | G base | G v2 | G tb2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Attributes | 29 | 62.1 | 72.4 | 79.3 | 55.2 | 65.5 | 69.0 | 51.7 | 58.6 | 75.9 |
| Material | 13 | 69.2 | 61.5 | 69.2 | 61.5 | 61.5 | 69.2 | 53.8 | 53.8 | 61.5 |
| OCR (printed question) | 68 | 83.8 | 91.2 | 92.6 | 86.8 | 85.3 | 89.7 | 50.0 | 86.8 | 86.8 |
| Object retrieval | 16 | 75.0 | 81.2 | 81.2 | 81.2 | 75.0 | 93.8 | 75.0 | 68.8 | 75.0 |
| Physical state | 23 | 56.5 | 60.9 | 56.5 | 65.2 | 73.9 | 73.9 | 73.9 | 87.0 | 78.3 |
| Comparison | 44 | 47.7 | 56.8 | 54.5 | 47.7 | 50.0 | 54.5 | 45.5 | 36.4 | 50.0 |
| Contact & occlusion | 41 | 53.7 | 58.5 | 56.1 | 56.1 | 51.2 | 48.8 | 43.9 | 58.5 | 53.7 |
| Ordering | 57 | 42.1 | 49.1 | 52.6 | 45.6 | 49.1 | 45.6 | 28.1 | 28.1 | 38.6 |
| **Perspective transform** | 85 | 48.2 | 54.1 | **64.7** | 16.5 | 22.4 | **52.9** | 18.8 | 21.2 | **49.4** |
| Spatial containment | 29 | 86.2 | 86.2 | 89.7 | 86.2 | 89.7 | 93.1 | 75.9 | 75.9 | 82.8 |

Categories have 13-85 questions: per-category differences under ~10-15 points are within noise.

### Perspective questions

| run | correct | left/right swap | mirror | front/back swap |
|---|---|---|---|---|
| Muse base | 14 | 40 | 22 | 7 |
| Muse tb2 | 45 | 22 | 2 | 15 |
| Gemma base | 16 | 34 | 25 | 5 |
| Gemma tb2 | 42 | 18 | 7 | 17 |
| Qwen base (think) | 41 | 20 | 9 | 15 |
| Qwen tb2 | 55 | 13 | 4 | 13 |

The viewpoint tools remove most mirror errors (the camera's view of a person facing it) and halve
left/right swaps. The remaining errors shift to front/back swaps, from wrong facing estimates or rough depth.

### Ablations (Qwen, tb2 agent, on the relevant category)

| ablation | questions | tb2 (all tools) | ablated | change | p |
|---|---|---|---|---|---|
| A1: without orient + viewpoint | perspective (85) | 64.7 | 56.5 | -8.2 | 0.31 |
| A2: without order | ordering (57) | 52.6 | 43.9 | -8.8 | 0.18 |
| A3: zoom only | perception (149) | 81.2 | 79.2 | -2.0 | 0.51 |

### Where the agent looked

TreeBench box mIoU of the tb2 agents' cited evidence: 34.8 (Qwen), 42.8 (Muse), 39.4 (Gemma); the TreeBench
paper reports 30.0 (DeepEyes-7B), 35.7 (Pixel-Reasoner-7B) and 44.0 (TreeVGR-7B). A tool call covered a
target on 75-86% of questions, but that did not predict a right answer (Qwen: 68.9% vs 69.0%).

### What the agent breaks

tb2 broke 29 (Qwen), 25 (Muse) and 31 (Gemma) questions the base model got right, mostly perspective,
ordering, comparison and contact: a wrong facing estimate, a missed or doubled detection, or a rough
depth/distance estimate followed over what the image shows.

## 5. Cost (provider-reported, 405 questions)

| model | v2 agent | tb2 agent |
|---|---|---|
| Qwen3.8-27B | $1.78 | $2.33 |
| Muse Glimmer 30B | $1.31 | $2.12 |
| Gemma 4 26B-A4B | $0.40 | $0.72 |

## 6. Limitations

- One run per setting; hosted models can vary slightly between calls.
- The tuning set is small (40) and tb1/tb2 tied on it; results are reported with and without it.
- Gemma receives a compressed image (~280 tokens) from its provider: part of its gain is zoom
  recovering detail the base model never saw.
- Agents differ from the base runs in more than tools (step-by-step prompting, more tokens); the
  ablations isolate specific tools only for perspective, ordering and perception.
- The perspective error classes assume the four direction options; they do not check the true facing.

## Reproduce

```bash
vlmbench run --config configs/treebench_agent/qwen3.8-27b.yaml                      # tb2 agent
vlmbench run --config configs/treebench_agent/qwen3.8-27b.yaml -a instructions=v2    # Bhushan's v2 agent
python analysis/treebench_agent.py --pair "label=<base run>:<agent run>" --out runs/treebench_agent_analysis
```
